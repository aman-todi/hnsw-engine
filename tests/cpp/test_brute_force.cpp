#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "hnsw/index.hpp"
#include "test_util.hpp"

using namespace hnsw;

namespace {

TEST(BruteForce, MatchesNaiveScan) {
  const std::size_t n = 500, dim = 19;
  const auto data = test::random_vectors(n, dim, 3);
  const auto q = test::random_vectors(1, dim, 4);
  BruteForceIndex bf(dim, Metric::L2);
  bf.add(data.data(), nullptr, n);
  const auto res = bf.search(q.data(), 10);
  ASSERT_EQ(res.size(), 10u);
  std::vector<std::pair<double, uint64_t>> naive;
  for (std::size_t i = 0; i < n; ++i) {
    double d = 0;
    for (std::size_t j = 0; j < dim; ++j) d += std::pow(double(data[i * dim + j]) - q[j], 2);
    naive.emplace_back(d, i);
  }
  std::sort(naive.begin(), naive.end());
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(res[i].label, naive[i].second);
    EXPECT_NEAR(res[i].distance, naive[i].first, 1e-4);
  }
}

TEST(BruteForce, CosineIgnoresScale) {
  const std::size_t dim = 8;
  std::vector<float> data = {1, 0, 0, 0, 0, 0, 0, 0,   // label 0
                             0, 5, 0, 0, 0, 0, 0, 0,   // label 1
                             3, 3, 0, 0, 0, 0, 0, 0};  // label 2
  BruteForceIndex bf(dim, Metric::Cosine);
  bf.add(data.data(), nullptr, 3);
  std::vector<float> q = {0, 100, 0, 0, 0, 0, 0, 0};
  const auto res = bf.search(q.data(), 3);
  EXPECT_EQ(res[0].label, 1u);
  EXPECT_NEAR(res[0].distance, 0.0f, 1e-6);
  EXPECT_EQ(res[1].label, 2u);
  EXPECT_NEAR(res[1].distance, 1.0f - std::sqrt(0.5f), 1e-6);
}

TEST(BruteForce, KLargerThanSize) {
  const auto data = test::random_vectors(3, 4, 1);
  BruteForceIndex bf(4, Metric::InnerProduct);
  bf.add(data.data(), nullptr, 3);
  EXPECT_EQ(bf.search(data.data(), 10).size(), 3u);
  EXPECT_THROW(bf.search(data.data(), 0), std::invalid_argument);
}

}  // namespace
