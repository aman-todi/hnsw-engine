// Runtime selection of distance kernels. The choice is made once (first call
// to active()) and stored in an atomic pointer; callers that need maximum
// speed (Index) cache the function pointers themselves.

#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "hnsw/metric.hpp"
#include "kernels.hpp"

namespace hnsw {

Metric parse_metric(std::string_view name) {
  if (name == "l2" || name == "L2" || name == "euclidean") return Metric::L2;
  if (name == "ip" || name == "inner_product" || name == "dot") return Metric::InnerProduct;
  if (name == "cosine" || name == "angular") return Metric::Cosine;
  throw std::invalid_argument("unknown metric '" + std::string(name) + "' (expected l2, ip or cosine)");
}

std::string_view metric_name(Metric m) noexcept {
  switch (m) {
    case Metric::L2:
      return "l2";
    case Metric::InnerProduct:
      return "ip";
    case Metric::Cosine:
      return "cosine";
  }
  return "unknown";
}

namespace simd {
namespace {

constexpr Kernels kScalar{Isa::Scalar, &l2_sq_scalar, &dot_scalar};
#if defined(HNSW_HAVE_X86_KERNELS)
constexpr Kernels kAvx2{Isa::AVX2, &detail::l2_sq_avx2, &detail::dot_avx2};
constexpr Kernels kAvx512{Isa::AVX512, &detail::l2_sq_avx512, &detail::dot_avx512};
#endif
#if defined(HNSW_HAVE_NEON_KERNELS)
constexpr Kernels kNeon{Isa::NEON, &detail::l2_sq_neon, &detail::dot_neon};
#endif

const Kernels* table_for(Isa isa) noexcept {
  switch (isa) {
    case Isa::Scalar:
      return &kScalar;
#if defined(HNSW_HAVE_X86_KERNELS)
    case Isa::AVX2:
      return &kAvx2;
    case Isa::AVX512:
      return &kAvx512;
#endif
#if defined(HNSW_HAVE_NEON_KERNELS)
    case Isa::NEON:
      return &kNeon;
#endif
    default:
      return nullptr;
  }
}

const Kernels* select_best() noexcept {
  const Kernels* best = &kScalar;
  if (supported(Isa::NEON)) best = table_for(Isa::NEON);
  if (supported(Isa::AVX2)) best = table_for(Isa::AVX2);
  if (supported(Isa::AVX512)) best = table_for(Isa::AVX512);
  // Allow forcing a lower ISA (ablation, debugging). Unsupported requests are ignored.
  // Read once, during static initialization of the dispatch table.
  if (const char* env = std::getenv("HNSW_SIMD");  // NOLINT(concurrency-mt-unsafe)
      env != nullptr && *env != '\0') {
    try {
      const Isa want = parse_isa(env);
      if (supported(want)) best = table_for(want);
    } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch)
      // Unknown value: keep the best supported ISA.
    }
  }
  return best;
}

std::atomic<const Kernels*>& active_ptr() noexcept {
  static std::atomic<const Kernels*> ptr{select_best()};
  return ptr;
}

}  // namespace

bool supported(Isa isa) noexcept {
  switch (isa) {
    case Isa::Scalar:
      return true;
#if defined(HNSW_HAVE_X86_KERNELS)
    case Isa::AVX2:
      return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    case Isa::AVX512:
      return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
#endif
#if defined(HNSW_HAVE_NEON_KERNELS)
    case Isa::NEON:
      return true;
#endif
    default:
      return false;
  }
}

const Kernels& active() noexcept {
  return *active_ptr().load(std::memory_order_acquire);
}

const Kernels& get(Isa isa) {
  if (!supported(isa)) {
    throw std::runtime_error("SIMD ISA '" + std::string(isa_name(isa)) +
                             "' is not available on this build/CPU");
  }
  return *table_for(isa);
}

void set_active(Isa isa) {
  active_ptr().store(&get(isa), std::memory_order_release);
}

std::string_view isa_name(Isa isa) noexcept {
  switch (isa) {
    case Isa::Scalar:
      return "scalar";
    case Isa::AVX2:
      return "avx2";
    case Isa::AVX512:
      return "avx512";
    case Isa::NEON:
      return "neon";
  }
  return "unknown";
}

Isa parse_isa(std::string_view name) {
  if (name == "scalar") return Isa::Scalar;
  if (name == "avx2") return Isa::AVX2;
  if (name == "avx512") return Isa::AVX512;
  if (name == "neon") return Isa::NEON;
  throw std::invalid_argument("unknown ISA '" + std::string(name) + "'");
}

}  // namespace simd
}  // namespace hnsw
