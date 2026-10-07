// AVX-512F kernels (optional ISA). Compiled with -mavx512f -mfma only; the
// tail is handled with a masked load instead of a scalar loop.

#include <immintrin.h>

#include "kernels.hpp"

namespace hnsw::simd::detail {

float l2_sq_avx512(const float* a, const float* b, std::size_t dim) noexcept {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps();
  __m512 acc3 = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 64 <= dim; i += 64) {
    const __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    const __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
    const __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32));
    const __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48));
    acc0 = _mm512_fmadd_ps(d0, d0, acc0);
    acc1 = _mm512_fmadd_ps(d1, d1, acc1);
    acc2 = _mm512_fmadd_ps(d2, d2, acc2);
    acc3 = _mm512_fmadd_ps(d3, d3, acc3);
  }
  for (; i + 16 <= dim; i += 16) {
    const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    acc0 = _mm512_fmadd_ps(d, d, acc0);
  }
  if (i < dim) {
    const __mmask16 m = static_cast<__mmask16>((1u << (dim - i)) - 1u);
    const __m512 d = _mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i));
    acc1 = _mm512_fmadd_ps(d, d, acc1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
}

float dot_avx512(const float* a, const float* b, std::size_t dim) noexcept {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps();
  __m512 acc3 = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 64 <= dim; i += 64) {
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
    acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
    acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
  }
  for (; i + 16 <= dim; i += 16) {
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
  }
  if (i < dim) {
    const __mmask16 m = static_cast<__mmask16>((1u << (dim - i)) - 1u);
    acc1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), acc1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
}

}  // namespace hnsw::simd::detail
