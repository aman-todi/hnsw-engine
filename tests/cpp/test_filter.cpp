#include <gtest/gtest.h>

#include <set>

#include "hnsw/filter.hpp"
#include "hnsw/index.hpp"
#include "test_util.hpp"

using namespace hnsw;

namespace {

TEST(Filter, BitsetBasics) {
  BitsetFilter f(130);
  f.set(0);
  f.set(64);
  f.set(129);
  EXPECT_TRUE(f.allowed(0));
  EXPECT_TRUE(f.allowed(64));
  EXPECT_TRUE(f.allowed(129));
  EXPECT_FALSE(f.allowed(1));
  EXPECT_FALSE(f.allowed(130));
  EXPECT_FALSE(f.allowed(~uint64_t{0}));
  EXPECT_EQ(f.count(), 3u);
  f.set(64, false);
  EXPECT_FALSE(f.allowed(64));
  EXPECT_THROW(f.set(500), std::out_of_range);
  const uint64_t labels[] = {5, 9};
  const auto g = BitsetFilter::from_labels(labels, 2);
  EXPECT_TRUE(g.allowed(9));
  EXPECT_FALSE(g.allowed(6));
}

class Selectivity : public ::testing::TestWithParam<double> {};

TEST_P(Selectivity, ResultsSatisfyFilterAndHaveGoodRecall) {
  const double sel = GetParam();
  const std::size_t n = 6000, dim = 24, nq = 100, k = 10;
  const auto data = test::clustered_vectors(n, dim, 77);
  const auto queries = test::clustered_vectors(nq, dim, 78);
  const auto labels = test::iota_labels(n);
  Index index(Params{dim});
  index.add_batch(data.data(), labels.data(), n);
  BruteForceIndex bf(dim, Metric::L2);
  bf.add(data.data(), labels.data(), n);

  // Pseudo-random allow-list with the requested selectivity.
  BitsetFilter allow(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (static_cast<double>((i * 2654435761u) % 1000) < sel * 1000) allow.set(i);
  }
  const FunctionFilter fn([&](uint64_t l) { return allow.allowed(l); });

  for (const Filter* f : {static_cast<const Filter*>(&allow), static_cast<const Filter*>(&fn)}) {
    std::vector<Neighbor> got(nq * k), truth(nq * k);
    index.search_batch(queries.data(), nq, k, 100, got.data(), 2, f);
    bf.search_batch(queries.data(), nq, k, truth.data(), 2, f);
    for (const auto& nb : got) {
      if (nb.label != kInvalidLabel) {
        EXPECT_TRUE(allow.allowed(nb.label)) << nb.label;
      }
    }
    const double r = test::recall(got, truth, nq, k);
    // Very selective filters are harder for graph search; see BENCHMARKS.md.
    EXPECT_GE(r, sel <= 0.01 ? 0.85 : 0.93) << "selectivity=" << sel;
  }
}

INSTANTIATE_TEST_SUITE_P(Sweep, Selectivity, ::testing::Values(0.01, 0.1, 0.5, 0.9));

TEST(SoftDelete, DeletedLabelsNeverReturned) {
  const std::size_t n = 3000, dim = 16;
  const auto data = test::clustered_vectors(n, dim, 55);
  Index index(Params{dim});
  index.add_batch(data.data(), test::iota_labels(n).data(), n);
  std::set<uint64_t> deleted;
  for (uint64_t l = 0; l < n; l += 3) {
    index.mark_deleted(l);
    deleted.insert(l);
  }
  index.mark_deleted(0);  // idempotent
  EXPECT_THROW(index.mark_deleted(999999), std::invalid_argument);
  EXPECT_EQ(index.size(), n);  // soft delete keeps nodes
  for (std::size_t i = 0; i < 200; ++i) {
    // Query exactly at a deleted vector: it must not come back.
    const auto res = index.search(data.data() + (i * 3) * dim, 10, 64);
    EXPECT_EQ(res.size(), 10u);
    for (auto& r : res) EXPECT_EQ(deleted.count(r.label), 0u) << r.label;
  }
  // Filter + deletion combined.
  BitsetFilter allow(n);
  for (uint64_t l = 0; l < n; l += 2) allow.set(l);
  const auto res = index.search(data.data(), 20, 100, &allow);
  for (auto& r : res) {
    EXPECT_TRUE(allow.allowed(r.label));
    EXPECT_EQ(deleted.count(r.label), 0u);
  }
}

TEST(SoftDelete, AllDeleted) {
  const auto data = test::random_vectors(20, 4, 1);
  Index index(Params{4});
  index.add_batch(data.data(), test::iota_labels(20).data(), 20);
  for (uint64_t l = 0; l < 20; ++l) index.mark_deleted(l);
  EXPECT_TRUE(index.search(data.data(), 5).empty());
}

}  // namespace
