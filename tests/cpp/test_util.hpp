#pragma once

#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include "hnsw/index.hpp"

namespace hnsw::test {

/// Portable RNG for test data. std::*_distribution output is
/// implementation-defined (libstdc++ and libc++ differ), which made recall
/// thresholds pass on Linux and fail on macOS for the "same" seed. splitmix64
/// + Box-Muller produce identical data on every platform.
class Rng {
 public:
  explicit Rng(uint64_t seed) : state_(seed) {}
  uint64_t next() noexcept {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  /// Uniform in [0, 1).
  double uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
  float uniform(float lo, float hi) noexcept { return lo + static_cast<float>(uniform()) * (hi - lo); }
  /// Standard normal (Box-Muller; the second value is discarded for simplicity).
  float normal() noexcept {
    const double u1 = 1.0 - uniform();  // (0, 1]
    const double u2 = uniform();
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
  }
  std::size_t below(std::size_t n) noexcept { return static_cast<std::size_t>(next() % n); }

 private:
  uint64_t state_;
};

inline std::vector<float> random_vectors(std::size_t n, std::size_t dim, uint64_t seed, float lo = -1.0f,
                                         float hi = 1.0f) {
  Rng rng(seed);
  std::vector<float> v(n * dim);
  for (auto& x : v) x = rng.uniform(lo, hi);
  return v;
}

/// Gaussian-mixture data: more realistic than uniform noise for ANN tests.
inline std::vector<float> clustered_vectors(std::size_t n, std::size_t dim, uint64_t seed,
                                            std::size_t clusters = 32) {
  Rng rng(seed);
  std::vector<float> centers(clusters * dim);
  for (auto& x : centers) x = 4.0f * rng.normal();
  std::vector<float> v(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = rng.below(clusters);
    for (std::size_t d = 0; d < dim; ++d) v[i * dim + d] = centers[c * dim + d] + rng.normal();
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
