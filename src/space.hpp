#pragma once

// Metric helpers shared by the HNSW and brute-force indexes.

#include <cmath>
#include <cstddef>

#include "hnsw/metric.hpp"

namespace hnsw::detail {

/// Inner product and cosine both use the "1 - dot" distance (cosine on
/// unit-normalized vectors). Squared L2 for L2.
struct Space {
  Metric metric = Metric::L2;
  simd::DistanceFn kernel = nullptr;
  bool is_l2 = true;

  Space() = default;
  Space(Metric m, const simd::Kernels& k)
      : metric(m), kernel(m == Metric::L2 ? k.l2_sq : k.dot), is_l2(m == Metric::L2) {}

  float operator()(const float* a, const float* b, std::size_t dim) const noexcept {
    const float r = kernel(a, b, dim);
    return is_l2 ? r : 1.0f - r;
  }
};

/// Copy `src` into `dst`, normalizing to unit length if the metric is cosine.
/// A zero vector is copied unchanged (its cosine distance to anything is 1).
inline void prepare_vector(Metric metric, const float* src, float* dst, std::size_t dim) noexcept {
  if (metric != Metric::Cosine) {
    for (std::size_t i = 0; i < dim; ++i) dst[i] = src[i];
    return;
  }
  double norm = 0.0;
  for (std::size_t i = 0; i < dim; ++i) norm += static_cast<double>(src[i]) * src[i];
  const float inv = norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 1.0f;
  for (std::size_t i = 0; i < dim; ++i) dst[i] = src[i] * inv;
}

}  // namespace hnsw::detail
