// Google Benchmark microbenchmarks: distance kernels per ISA across dims, and
// the search hot loop on a small in-memory index.

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "hnsw/index.hpp"
#include "hnsw/metric.hpp"

using namespace hnsw;

namespace {

std::vector<float> rand_vec(std::size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);  // exact values irrelevant for timing
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

void BM_L2(benchmark::State& state, simd::Isa isa) {
  if (!simd::supported(isa)) {
    state.SkipWithError("ISA not supported");
    return;
  }
  const auto& k = simd::get(isa);
  const auto dim = static_cast<std::size_t>(state.range(0));
  const auto a = rand_vec(dim, 1), b = rand_vec(dim, 2);
  for (auto _ : state) {
    benchmark::DoNotOptimize(k.l2_sq(a.data(), b.data(), dim));
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["GFLOP/s"] =
      benchmark::Counter(static_cast<double>(state.iterations()) * 3.0 * static_cast<double>(dim),
                         benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

void BM_Dot(benchmark::State& state, simd::Isa isa) {
  if (!simd::supported(isa)) {
    state.SkipWithError("ISA not supported");
    return;
  }
  const auto& k = simd::get(isa);
  const auto dim = static_cast<std::size_t>(state.range(0));
  const auto a = rand_vec(dim, 1), b = rand_vec(dim, 2);
  for (auto _ : state) {
    benchmark::DoNotOptimize(k.dot(a.data(), b.data(), dim));
  }
  state.SetItemsProcessed(state.iterations());
}

#define KERNEL_DIMS ->Arg(16)->Arg(100)->Arg(128)->Arg(384)->Arg(768)->Arg(960)->Arg(1024)
BENCHMARK_CAPTURE(BM_L2, scalar, simd::Isa::Scalar) KERNEL_DIMS;
BENCHMARK_CAPTURE(BM_L2, avx2, simd::Isa::AVX2) KERNEL_DIMS;
BENCHMARK_CAPTURE(BM_L2, avx512, simd::Isa::AVX512) KERNEL_DIMS;
BENCHMARK_CAPTURE(BM_L2, neon, simd::Isa::NEON) KERNEL_DIMS;
BENCHMARK_CAPTURE(BM_Dot, scalar, simd::Isa::Scalar) KERNEL_DIMS;
BENCHMARK_CAPTURE(BM_Dot, avx2, simd::Isa::AVX2) KERNEL_DIMS;
BENCHMARK_CAPTURE(BM_Dot, avx512, simd::Isa::AVX512) KERNEL_DIMS;
BENCHMARK_CAPTURE(BM_Dot, neon, simd::Isa::NEON) KERNEL_DIMS;

/// The search hot loop (search_layer on layer 0 dominates).
void BM_Search(benchmark::State& state) {
  const std::size_t n = 20000, dim = 128;
  static Index* index = [] {
    auto* idx = new Index(Params{dim});
    const auto data = rand_vec(n * dim, 3);
    std::vector<uint64_t> labels(n);
    for (std::size_t i = 0; i < n; ++i) labels[i] = i;
    idx->add_batch(data.data(), labels.data(), n);
    return idx;
  }();
  const auto queries = rand_vec(256 * dim, 4);
  const auto ef = static_cast<std::size_t>(state.range(0));
  std::size_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(index->search(queries.data() + (i++ % 256) * dim, 10, ef));
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Search)->Arg(16)->Arg(64)->Arg(256);

}  // namespace

BENCHMARK_MAIN();
