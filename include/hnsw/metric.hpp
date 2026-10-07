#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace hnsw {

enum class Metric { L2, InnerProduct, Cosine };

/// Parse "l2" / "ip" / "inner_product" / "cosine" (case-sensitive). Throws std::invalid_argument.
Metric parse_metric(std::string_view name);
std::string_view metric_name(Metric m) noexcept;

namespace simd {

/// Instruction sets the distance kernels are implemented for.
enum class Isa { Scalar, AVX2, AVX512, NEON };

using DistanceFn = float (*)(const float* a, const float* b, std::size_t dim) noexcept;

struct Kernels {
  Isa isa;
  DistanceFn l2_sq;  ///< sum_i (a_i - b_i)^2
  DistanceFn dot;    ///< sum_i a_i * b_i
};

/// Kernels selected at startup (best ISA supported by the CPU, unless the
/// HNSW_SIMD environment variable names a lower one: scalar|avx2|avx512|neon).
const Kernels& active() noexcept;

/// Kernels for a specific ISA. Throws std::runtime_error if the ISA is not
/// compiled in or not supported by this CPU.
const Kernels& get(Isa isa);

/// True if `isa` is compiled in and supported by the running CPU.
bool supported(Isa isa) noexcept;

/// Override the active kernels (for tests/ablation benchmarks). Not
/// thread-safe with respect to concurrently constructed indexes; an Index
/// captures the active kernels when it is constructed or loaded.
void set_active(Isa isa);

std::string_view isa_name(Isa isa) noexcept;
Isa parse_isa(std::string_view name);

/// Scalar reference implementations (always available).
float l2_sq_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float dot_scalar(const float* a, const float* b, std::size_t dim) noexcept;

}  // namespace simd

/// Convenience wrappers that use the active kernels.
inline float l2_sq(const float* a, const float* b, std::size_t dim) noexcept {
  return simd::active().l2_sq(a, b, dim);
}
inline float dot(const float* a, const float* b, std::size_t dim) noexcept {
  return simd::active().dot(a, b, dim);
}

}  // namespace hnsw
