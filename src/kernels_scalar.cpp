// Scalar reference kernels. Deliberately a plain sequential loop: this is the
// numerical reference the SIMD kernels are tested against and the baseline of
// the ablation benchmark.

#include "hnsw/metric.hpp"

namespace hnsw::simd {

float l2_sq_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float dot_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
  return sum;
}

}  // namespace hnsw::simd
