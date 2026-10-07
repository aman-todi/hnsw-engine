#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

#include "hnsw/index.hpp"
#include "test_util.hpp"

using namespace hnsw;

namespace {

std::string tmp_path(const std::string& name) {
  return (std::filesystem::temp_directory_path() / ("hnsw_persist_" + name)).string();
}

std::vector<char> read_all(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
void write_all(const std::string& path, const std::vector<char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void expect_same_results(const Index& a, const Index& b, const std::vector<float>& queries, std::size_t nq,
                         std::size_t dim) {
  for (std::size_t i = 0; i < nq; ++i) {
    const auto ra = a.search(queries.data() + i * dim, 10, 50);
    const auto rb = b.search(queries.data() + i * dim, 10, 50);
    ASSERT_EQ(ra.size(), rb.size());
    for (std::size_t j = 0; j < ra.size(); ++j) {
      EXPECT_EQ(ra[j].label, rb[j].label);
      EXPECT_EQ(ra[j].distance, rb[j].distance);
    }
  }
}

class Persist : public ::testing::TestWithParam<Metric> {};

TEST_P(Persist, RoundTripBothModes) {
  const std::size_t n = 3000, dim = 37, nq = 50;
  const auto data = test::clustered_vectors(n, dim, 1);
  const auto queries = test::clustered_vectors(nq, dim, 2);
  Index index(Params{dim, GetParam(), 12, 100});
  index.add_batch(data.data(), test::iota_labels(n, 500).data(), n);
  index.mark_deleted(600);
  index.set_ef(33);
  const auto path = tmp_path("roundtrip.bin");
  index.save(path);

  for (bool mmap : {false, true}) {
    Index loaded = Index::load(path, mmap);
    EXPECT_EQ(loaded.size(), n);
    EXPECT_EQ(loaded.dim(), dim);
    EXPECT_EQ(loaded.metric(), GetParam());
    EXPECT_EQ(loaded.ef(), 33u);
    EXPECT_EQ(loaded.read_only(), mmap);
    EXPECT_TRUE(loaded.is_deleted(600));
    expect_same_results(index, loaded, queries, nq, dim);
    if (mmap) {
      EXPECT_THROW(loaded.add(data.data(), 1), std::runtime_error);
      loaded.mark_deleted(700);  // tombstones are kept in memory, allowed
      EXPECT_TRUE(loaded.is_deleted(700));
    } else {
      // A loaded (non-mmap) index stays fully mutable.
      const auto extra = test::clustered_vectors(10, dim, 3);
      loaded.add_batch(extra.data(), test::iota_labels(10, 100000).data(), 10);
      EXPECT_EQ(loaded.size(), n + 10);
      EXPECT_TRUE(loaded.contains(100009));
      if (GetParam() != Metric::InnerProduct) {  // IP: self is not necessarily nearest
        EXPECT_EQ(loaded.search(extra.data(), 1, 50)[0].label, 100000u);
      }
    }
    // Re-saving a loaded index reproduces the identical file.
    if (mmap) {
      const auto path2 = tmp_path("resave.bin");
      Index again = Index::load(path, true);
      again.save(path2);
      EXPECT_EQ(read_all(path), read_all(path2));
      std::remove(path2.c_str());
    }
  }
  std::remove(path.c_str());
}

INSTANTIATE_TEST_SUITE_P(Metrics, Persist,
                         ::testing::Values(Metric::L2, Metric::InnerProduct, Metric::Cosine));

TEST(PersistEdge, EmptyIndexRoundTrip) {
  const auto path = tmp_path("empty.bin");
  Index index(Params{5});
  index.save(path);
  for (bool mmap : {false, true}) {
    Index loaded = Index::load(path, mmap);
    EXPECT_EQ(loaded.size(), 0u);
    std::vector<float> q(5, 1.0f);
    EXPECT_TRUE(loaded.search(q.data(), 3).empty());
  }
  std::remove(path.c_str());
}

TEST(PersistEdge, MissingFile) {
  EXPECT_THROW(Index::load(tmp_path("nope.bin"), false), std::runtime_error);
  EXPECT_THROW(Index::load(tmp_path("nope.bin"), true), std::runtime_error);
}

TEST(PersistFuzz, TruncationAlwaysThrows) {
  const std::size_t n = 400, dim = 9;
  const auto data = test::random_vectors(n, dim, 4);
  Index index(Params{dim, Metric::L2, 8, 50});
  index.add_batch(data.data(), test::iota_labels(n).data(), n);
  const auto path = tmp_path("trunc_src.bin");
  index.save(path);
  const auto bytes = read_all(path);
  const auto bad = tmp_path("trunc.bin");
  std::mt19937 rng(7);
  std::vector<std::size_t> cuts = {0, 1, 7, 8, 100, 255, 256, 257, bytes.size() - 1};
  for (int i = 0; i < 40; ++i) cuts.push_back(rng() % bytes.size());
  for (std::size_t cut : cuts) {
    write_all(bad, std::vector<char>(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(cut)));
    for (bool mmap : {false, true}) {
      EXPECT_THROW(Index::load(bad, mmap), std::runtime_error) << "cut=" << cut << " mmap=" << mmap;
    }
  }
  // Appending garbage is also rejected.
  auto longer = bytes;
  longer.push_back(0);
  write_all(bad, longer);
  EXPECT_THROW(Index::load(bad, false), std::runtime_error);
  std::remove(path.c_str());
  std::remove(bad.c_str());
}

TEST(PersistFuzz, ByteFlipsAlwaysThrow) {
  const std::size_t n = 300, dim = 6;
  const auto data = test::random_vectors(n, dim, 5);
  Index index(Params{dim, Metric::Cosine, 6, 40});
  index.add_batch(data.data(), test::iota_labels(n).data(), n);
  index.mark_deleted(3);
  const auto path = tmp_path("flip_src.bin");
  index.save(path);
  const auto bytes = read_all(path);
  const auto bad = tmp_path("flip.bin");
  std::mt19937 rng(11);
  // Every header byte, then random positions over the whole file.
  std::vector<std::size_t> positions;
  for (std::size_t i = 0; i < 256; ++i) positions.push_back(i);
  for (int i = 0; i < 400; ++i) positions.push_back(rng() % bytes.size());
  for (std::size_t pos : positions) {
    auto corrupted = bytes;
    corrupted[pos] = static_cast<char>(corrupted[pos] ^ static_cast<char>(1u << (rng() % 8)));
    write_all(bad, corrupted);
    for (bool mmap : {false, true}) {
      EXPECT_THROW(Index::load(bad, mmap), std::runtime_error) << "pos=" << pos << " mmap=" << mmap;
    }
  }
  std::remove(path.c_str());
  std::remove(bad.c_str());
}

}  // namespace
