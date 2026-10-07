#pragma once

#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "hnsw/index.hpp"

namespace hnsw::test {

inline std::vector<float> random_vectors(std::size_t n, std::size_t dim, uint64_t seed, float lo = -1.0f,
                                         float hi = 1.0f) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n * dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

/// Gaussian-mixture data: more realistic than uniform noise for ANN tests.
inline std::vector<float> clustered_vectors(std::size_t n, std::size_t dim, uint64_t seed,
                                            std::size_t clusters = 32) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> g(0.0f, 1.0f);
  std::vector<float> centers(clusters * dim);
  for (auto& x : centers) x = 4.0f * g(rng);
  std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
  std::vector<float> v(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = pick(rng);
    for (std::size_t d = 0; d < dim; ++d) v[i * dim + d] = centers[c * dim + d] + g(rng);
  }
  return v;
}

inline std::vector<uint64_t> iota_labels(std::size_t n, uint64_t start = 0) {
  std::vector<uint64_t> l(n);
  for (std::size_t i = 0; i < n; ++i) l[i] = start + i;
  return l;
}

/// Mean recall@k of `got` (nq*k rows) against `truth` (nq*k rows).
inline double recall(const std::vector<Neighbor>& got, const std::vector<Neighbor>& truth, std::size_t nq,
                     std::size_t k) {
  std::size_t hit = 0;
  for (std::size_t i = 0; i < nq; ++i) {
    std::set<uint64_t> t;
    for (std::size_t j = 0; j < k; ++j) t.insert(truth[i * k + j].label);
    for (std::size_t j = 0; j < k; ++j) hit += t.count(got[i * k + j].label);
  }
  return static_cast<double>(hit) / static_cast<double>(nq * k);
}

}  // namespace hnsw::test
