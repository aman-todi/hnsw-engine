#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <set>
#include <tuple>

#include "hnsw/index.hpp"
#include "test_util.hpp"

using namespace hnsw;

namespace {

struct RecallCase {
  std::size_t dim;
  Metric metric;
  std::size_t ef = 100;
  double min_recall = 0.95;
};

class IndexRecall : public ::testing::TestWithParam<RecallCase> {};

TEST_P(IndexRecall, MatchesBruteForceOracle) {
  const auto [dim, metric, ef, min_recall] = GetParam();
  const std::size_t n = 4000, nq = 100, k = 10;
  const auto data = test::clustered_vectors(n, dim, 11);
  const auto queries = test::clustered_vectors(nq, dim, 12);
  const auto labels = test::iota_labels(n, 1000);

  Index index(Params{dim, metric, 16, 200});
  index.add_batch(data.data(), labels.data(), n, 1);
  BruteForceIndex bf(dim, metric);
  bf.add(data.data(), labels.data(), n);

  std::vector<Neighbor> got(nq * k), truth(nq * k);
  index.search_batch(queries.data(), nq, k, ef, got.data(), 2);
  bf.search_batch(queries.data(), nq, k, truth.data(), 2);
  EXPECT_GE(test::recall(got, truth, nq, k), min_recall)
      << "dim=" << dim << " metric=" << metric_name(metric);

  // Distances reported by HNSW must equal the true distance to that label.
  for (std::size_t i = 0; i < nq * k; i += 17) {
    const auto v = index.get_vector(got[i].label);
    BruteForceIndex one(dim, metric);
    one.add(data.data() + (got[i].label - 1000) * dim, nullptr, 1);
    const float expect = one.search(queries.data() + (i / k) * dim, 1)[0].distance;
    EXPECT_NEAR(expect, got[i].distance, 1e-5 * std::max(1.0f, std::fabs(expect)));
  }
}

INSTANTIATE_TEST_SUITE_P(DimsAndMetrics, IndexRecall,
                         ::testing::Values(RecallCase{3, Metric::L2}, RecallCase{16, Metric::L2},
                                           RecallCase{100, Metric::L2}, RecallCase{128, Metric::L2},
                                           // Max inner product on un-normalized clustered data
                                           // is much harder for graph search (hnswlib reaches
                                           // 0.81 at ef=100 on this data too).
                                           RecallCase{33, Metric::InnerProduct, 400, 0.93},
                                           RecallCase{64, Metric::Cosine}, RecallCase{100, Metric::Cosine}));

TEST(Index, RandomUniform10kRecall) {
  // M2 acceptance: 10k random vectors, recall@10 >= 0.95 at modest ef.
  const std::size_t n = 10000, dim = 32, nq = 200, k = 10;
  const auto data = test::random_vectors(n, dim, 21);
  const auto queries = test::random_vectors(nq, dim, 22);
  const auto labels = test::iota_labels(n);
  Index index(Params{dim, Metric::L2, 16, 200});
  index.add_batch(data.data(), labels.data(), n);
  BruteForceIndex bf(dim, Metric::L2);
  bf.add(data.data(), labels.data(), n);
  std::vector<Neighbor> got(nq * k), truth(nq * k);
  bf.search_batch(queries.data(), nq, k, truth.data());
  index.search_batch(queries.data(), nq, k, 128, got.data());
  EXPECT_GE(test::recall(got, truth, nq, k), 0.95);
}

TEST(Index, EmptyIndex) {
  Index index(Params{8});
  std::vector<float> q(8, 0.0f);
  EXPECT_TRUE(index.search(q.data(), 5).empty());
  EXPECT_EQ(index.size(), 0u);
  std::vector<Neighbor> out(3);
  index.search_batch(q.data(), 1, 3, 10, out.data());
  for (auto& nb : out) {
    EXPECT_EQ(nb.label, kInvalidLabel);
    EXPECT_TRUE(std::isinf(nb.distance));
  }
}

TEST(Index, KLargerThanSizeAndEfClamp) {
  const auto data = test::random_vectors(5, 4, 1);
  Index index(Params{4});
  index.add_batch(data.data(), test::iota_labels(5).data(), 5);
  const auto res = index.search(data.data(), 50, 1);  // ef < k: clamped to k
  ASSERT_EQ(res.size(), 5u);
  for (std::size_t i = 1; i < res.size(); ++i) EXPECT_LE(res[i - 1].distance, res[i].distance);
  EXPECT_EQ(res[0].label, 0u);
  EXPECT_THROW(index.search(data.data(), 0), std::invalid_argument);
}

TEST(Index, InvalidParams) {
  EXPECT_THROW(Index(Params{0}), std::invalid_argument);
  Params p{4};
  p.M = 1;
  EXPECT_THROW(Index{p}, std::invalid_argument);
  p = Params{4};
  p.ef_construction = 0;
  EXPECT_THROW(Index{p}, std::invalid_argument);
  Index index(Params{4});
  EXPECT_THROW(index.set_ef(0), std::invalid_argument);
}

TEST(Index, DuplicateLabels) {
  const auto data = test::random_vectors(4, 4, 1);
  Index index(Params{4});
  index.add(data.data(), 7);
  EXPECT_THROW(index.add(data.data() + 4, 7), std::invalid_argument);
  const uint64_t dup[2] = {9, 9};
  EXPECT_THROW(index.add_batch(data.data(), dup, 2), std::invalid_argument);
  const uint64_t clash[2] = {10, 7};
  EXPECT_THROW(index.add_batch(data.data(), clash, 2), std::invalid_argument);
  EXPECT_EQ(index.size(), 1u);  // failed batches insert nothing
  EXPECT_FALSE(index.contains(10));
}

TEST(Index, DimOne) {
  Index index(Params{1});
  for (int i = 0; i < 200; ++i) {
    const float v = static_cast<float>(i);
    index.add(&v, static_cast<uint64_t>(i));
  }
  const float q = 42.2f;
  const auto res = index.search(&q, 3, 50);
  ASSERT_EQ(res.size(), 3u);
  EXPECT_EQ(res[0].label, 42u);
  EXPECT_EQ(res[1].label, 43u);
  EXPECT_EQ(res[2].label, 41u);
}

TEST(Index, IdenticalVectors) {
  const std::size_t n = 300, dim = 12;
  std::vector<float> data(n * dim, 0.5f);
  Index index(Params{dim});
  index.add_batch(data.data(), test::iota_labels(n).data(), n, 1);
  const auto res = index.search(data.data(), 10, 64);
  ASSERT_EQ(res.size(), 10u);
  std::set<uint64_t> uniq;
  for (auto& r : res) {
    EXPECT_EQ(r.distance, 0.0f);
    uniq.insert(r.label);
  }
  EXPECT_EQ(uniq.size(), 10u);
}

TEST(Index, DeterministicForFixedSeedSingleThread) {
  const std::size_t n = 2000, dim = 24;
  const auto data = test::clustered_vectors(n, dim, 5);
  const auto labels = test::iota_labels(n);
  auto build = [&](uint64_t seed) {
    Params p{dim};
    p.seed = seed;
    Index index(p);
    index.add_batch(data.data(), labels.data(), n, 1);
    return index;
  };
  Index a = build(42), b = build(42);
  // add() one by one must produce the same graph as add_batch with 1 thread.
  Params p{dim};
  p.seed = 42;
  Index c(p);
  for (std::size_t i = 0; i < n; ++i) c.add(data.data() + i * dim, labels[i]);
  EXPECT_EQ(a.max_level(), b.max_level());
  for (std::size_t qi = 0; qi < 50; ++qi) {
    const auto ra = a.search(data.data() + qi * dim * 7, 10, 40);
    const auto rb = b.search(data.data() + qi * dim * 7, 10, 40);
    const auto rc = c.search(data.data() + qi * dim * 7, 10, 40);
    ASSERT_EQ(ra.size(), rb.size());
    for (std::size_t j = 0; j < ra.size(); ++j) {
      EXPECT_EQ(ra[j].label, rb[j].label);
      EXPECT_EQ(ra[j].distance, rb[j].distance);
      EXPECT_EQ(ra[j].label, rc[j].label);
    }
  }
}

TEST(Index, GrowsGeometricallyAndRespectsMaxElements) {
  const auto data = test::random_vectors(3000, 8, 9);
  Index grow(Params{8});
  for (std::size_t i = 0; i < 3000; ++i) grow.add(data.data() + i * 8, i);
  EXPECT_EQ(grow.size(), 3000u);
  EXPECT_GE(grow.capacity(), 3000u);
  const auto res = grow.search(data.data() + 8 * 1234, 1, 50);
  EXPECT_EQ(res[0].label, 1234u);

  Params p{8};
  p.max_elements = 10;
  Index fixed(p);
  fixed.add_batch(data.data(), test::iota_labels(10).data(), 10);
  EXPECT_THROW(fixed.add(data.data(), 99), std::runtime_error);
}

TEST(Index, CosineDoesNotMutateCallerBuffers) {
  auto data = test::random_vectors(50, 10, 3, 1.0f, 5.0f);
  const auto copy = data;
  Index index(Params{10, Metric::Cosine});
  index.add_batch(data.data(), test::iota_labels(50).data(), 50);
  auto q = test::random_vectors(1, 10, 4, 1.0f, 5.0f);
  const auto qcopy = q;
  index.search(q.data(), 5);
  EXPECT_EQ(data, copy);
  EXPECT_EQ(q, qcopy);
  // Stored vectors are unit-normalized.
  const auto v = index.get_vector(3);
  double norm = 0;
  for (float x : v) norm += double(x) * x;
  EXPECT_NEAR(norm, 1.0, 1e-5);
}

TEST(Index, ScalarAndSimdIndexesAgree) {
  const std::size_t n = 1500, dim = 100;
  const auto data = test::clustered_vectors(n, dim, 31);
  const auto labels = test::iota_labels(n);
  const auto prev = simd::active().isa;
  simd::set_active(simd::Isa::Scalar);
  Index scalar(Params{dim});
  scalar.add_batch(data.data(), labels.data(), n, 1);
  simd::set_active(prev);
  Index fast(Params{dim});
  fast.add_batch(data.data(), labels.data(), n, 1);
  std::size_t same = 0, total = 0;
  for (std::size_t qi = 0; qi < 50; ++qi) {
    const auto a = scalar.search(data.data() + qi * dim, 10, 64);
    const auto b = fast.search(data.data() + qi * dim, 10, 64);
    std::set<uint64_t> sa;
    for (auto& r : a) sa.insert(r.label);
    for (auto& r : b) same += sa.count(r.label);
    total += 10;
  }
  // Rounding differences may perturb ties, but results must be nearly identical.
  EXPECT_GE(double(same) / double(total), 0.98);
}

}  // namespace
