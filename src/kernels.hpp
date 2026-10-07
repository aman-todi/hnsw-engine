#pragma once

// Internal declarations of every compiled-in kernel. Each implementation lives
// in its own translation unit compiled with only its own ISA flags.

#include <cstddef>

namespace hnsw::simd::detail {

float l2_sq_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float dot_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float l2_sq_avx512(const float* a, const float* b, std::size_t dim) noexcept;
float dot_avx512(const float* a, const float* b, std::size_t dim) noexcept;
float l2_sq_neon(const float* a, const float* b, std::size_t dim) noexcept;
float dot_neon(const float* a, const float* b, std::size_t dim) noexcept;

}  // namespace hnsw::simd::detail
