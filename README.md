# hnsw-engine

[![CI](https://github.com/aman-todi/hnsw-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aman-todi/hnsw-engine/actions/workflows/ci.yml)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

An approximate nearest-neighbor search library written from scratch: a
Hierarchical Navigable Small World (HNSW) index in C++20, implemented from the
Malkov & Yashunin paper, with hand-written AVX2 / AVX-512 / NEON distance
kernels behind runtime CPU dispatch, a cache-friendly flat graph layout with
software prefetching, parallel build and batch search, a versioned and
checksummed on-disk format with memory-mapped loading, filtered search and soft
deletes, and a NumPy-first Python package that releases the GIL. Every
performance number below comes from a script in this repository and is
benchmarked against **hnswlib** and **FAISS `IndexHNSWFlat`** under identical
parameters.

> **Benchmarks** are on the real ann-benchmarks datasets (SIFT-1M,
> GloVe-100, GIST-1M at 200k) on two laptops: an Apple M5 MacBook Pro (ARM,
> NEON) and an Intel i7-13800H HP ZBook under WSL2 (x86, AVX2). On ARM, FAISS
> is the like-for-like competitor (hnswlib has no NEON kernels); see
> [Results](#results) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Architecture

```
          Python: hnsw_engine.Index  (NumPy in/out, GIL released, filters)
                              │ pybind11
 ┌────────────────────────────▼─────────────────────────────────────────────┐
 │ hnsw::Index            std::shared_mutex: searches shared, writes exclusive│
 │  build  ── Alg.1 insert + Alg.4 heuristic ── 4096 striped node locks       │
 │  search ── greedy descent ── search_layer (Alg.2) ── ScratchPool           │
 │                                     (epoch visited lists, heaps, query buf)│
 │  storage: [vectors: 64B-aligned, padded rows][layer0: count|2M ids][upper] │
 │           pointers into owned arrays *or* a read-only mmap of the file     │
 │  persist: header + 6 aligned sections, header & payload checksums          │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ simd::active() → l2_sq / dot : scalar | AVX2+FMA | AVX-512 | NEON          │
 │                  (picked once via __builtin_cpu_supports; own TU per ISA)  │
 │ parallel_for   → atomic-counter work sharing for add_batch / search_batch  │
 └────────────────────────────────────────────────────────────────────────────┘
```

Details: [docs/DESIGN.md](docs/DESIGN.md) (layout, locking, dispatch),
[docs/FORMAT.md](docs/FORMAT.md) (on-disk format),
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) (methodology, full results).

## Quickstart

Requirements: CMake ≥ 3.20, a C++20 compiler (GCC ≥ 11, Clang ≥ 14, AppleClang
≥ 14), Ninja, Python ≥ 3.9 with NumPy. Linux or macOS.

### Python

```bash
pip install .
```

```python
import numpy as np
import hnsw_engine

data = np.random.rand(100_000, 128).astype(np.float32)
idx = hnsw_engine.Index(dim=128, metric="l2", M=16, ef_construction=200)
idx.add(data)                                   # ids default to 0..n-1; num_threads=0 -> all cores
ids, dists = idx.search(data[:5], k=10, ef=64)  # (5, 10) int64, (5, 10) float32

idx.search(data[:5], k=10, filter=lambda label: label % 2 == 0)   # predicate
idx.search(data[:5], k=10, filter=np.arange(0, 100_000, 3))       # allow-list
idx.mark_deleted(42)

idx.save("index.bin")
ro = hnsw_engine.Index.load("index.bin", mmap=True)   # zero-copy, read-only
```

Metrics: `"l2"` (squared), `"ip"` (1 − dot), `"cosine"` (1 − cosine).

### C++

```bash
cmake --preset release && cmake --build --preset release
ctest --preset release
```

```cpp
#include "hnsw/index.hpp"

hnsw::Index index(hnsw::Params{.dim = 128, .metric = hnsw::Metric::L2, .M = 16, .ef_construction = 200});
index.add_batch(vectors, labels, n);                       // parallel
auto top10 = index.search(query, 10, /*ef=*/64);           // std::vector<Neighbor>{label, distance}

hnsw::BitsetFilter allow(n);  allow.set(7);  allow.set(99);
auto filtered = index.search(query, 10, 64, &allow);

index.save("index.bin");
auto mapped = hnsw::Index::load("index.bin", /*mmap=*/true);
```

Link against the `hnsw::engine` CMake target (`add_subdirectory` this repo).

### Reproduce a benchmark (≈ 15 minutes)

```bash
pip install numpy h5py hnswlib faiss-cpu matplotlib
python scripts/fetch_data.py sift --subset 200000      # real SIFT (500 MB download) -> data/sift-200k_*
cmake --preset bench && cmake --build --preset bench --target bench_main
./build/bench/bench/bench_main --data data --name sift-200k --nq 1000 --ef 10,20,40,80,160 --build-threads 0
HNSW_NATIVE=ON pip install .
mkdir -p bench/results/my-machine
python bench/python/compare.py --data data --name sift-200k --metric l2 --nq 1000 --reps 1 \
    --out bench/results/my-machine/sift-200k.csv
python bench/python/plot.py --results bench/results/my-machine   # sift-200k.png + summary.md
```

The full pipeline (all datasets, ablation, scaling, filters, plots, docs) is
`scripts/run_all_benchmarks.sh [--quick]`, which writes to
`bench/results/<cpu-slug>/` and re-renders the docs from every machine folder;
`--synthetic` generates offline
stand-ins when ann-benchmarks.com is unreachable.

## Results

<!-- results:begin -->
Measured with `scripts/run_all_benchmarks.sh` on the real ann-benchmarks datasets (SIFT-1M, GloVe-100, GIST-1M at 200k vectors), M = 16, ef_construction = 200, k = 10, on 2 machines:

* **Apple M5**: 4 threads (performance cores), engine SIMD NEON; Apple clang version 21.0.0 (clang-2100.1.1.101); hnswlib 0.8.0, faiss-cpu 1.13.0. On this machine hnswlib ships hand-written SIMD kernels only for x86, so on this ARM machine its distances run the compiler's generic code; FAISS (NEON kernels) is the like-for-like comparison here.
* **Intel i7-13800H (WSL2)**: 6 threads (performance cores), engine SIMD AVX2; c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; hnswlib 0.8.0, faiss-cpu 1.15.1. On this machine all three libraries use hand-written x86 SIMD kernels here (engine: AVX2), so this is the like-for-like three-way comparison.

Full tables, methodology and raw CSVs: [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and `bench/results/<machine>/`.

![recall vs QPS on SIFT-1M, Apple M5](bench/results/apple-m5/sift.png)
![recall vs QPS on SIFT-1M, Intel i7-13800H (WSL2)](bench/results/intel-i7-13800h-wsl2/sift.png)

**Single-thread QPS at a recall@10 target** (one query per Python call; QPS interpolated on each library's recall-QPS curve; bold = fastest on that machine):

| dataset | recall@10 | machine | hnsw-engine | hnswlib | FAISS HNSWFlat |
|---|---|---|---:|---:|---:|
| SIFT-1M | 0.95 | Apple M5 | **16,570** | 8,559 | 14,943 |
| SIFT-1M | 0.95 | Intel i7-13800H (WSL2) | **12,122** | 10,915 | 9,884 |
| SIFT-1M | 0.99 | Apple M5 | **7,623** | 3,989 | 6,596 |
| SIFT-1M | 0.99 | Intel i7-13800H (WSL2) | **5,663** | 5,081 | 4,332 |
| GloVe-100 | 0.90 | Apple M5 | **3,106** | 2,082 | 2,719 |
| GloVe-100 | 0.90 | Intel i7-13800H (WSL2) | **2,363** | 2,211 | 1,672 |
| GIST | 0.95 | Apple M5 | 1,696 | 773 | **2,348** |
| GIST | 0.95 | Intel i7-13800H (WSL2) | 1,341 | 1,176 | **1,674** |
| GIST | 0.99 | Apple M5 | 581 | 268 | **727** |
| GIST | 0.99 | Intel i7-13800H (WSL2) | **496** | 408 | — |

**Build time, SIFT-1M** (multi-threaded / single-threaded):

| machine | threads | hnsw-engine | hnswlib | FAISS HNSWFlat |
|---|---:|---:|---:|---:|
| Apple M5 | 4 | 42 s / 154 s | 82 s / 298 s | 56 s / 198 s |
| Intel i7-13800H (WSL2) | 6 | 45 s / 212 s | 57 s / 264 s | 67 s / 320 s |

**Where the speed comes from** (engine only, identical recall in every row):

| configuration (C++ harness, SIFT-1M, ef = 64, same graph) | Apple M5 (NEON) | Intel i7-13800H (WSL2) (AVX2) |
|---|---:|---:|
| scalar kernels, no prefetch | 7,830 (1.0×) | 4,288 (1.0×) |
| + SIMD kernels | 13,878 (1.8×) | 10,007 (2.3×) |
| + prefetch | 15,234 (1.9×) | 12,078 (2.8×) |
| + 4/6 search threads (batched) | 55,324 (7.1×) | 55,905 (13.0×) |

*Apple M5:* L2 distance kernel (d = 128) scalar 26.3 ns, NEON 5.2 ns (5.0×); parallel build 3.8× on 4 threads (200,000 vectors: 21.0 s → 5.5 s; recall@10 at ef = 64 0.9811 → 0.9810).

*Intel i7-13800H (WSL2):* L2 distance kernel (d = 128) scalar 33.8 ns, AVX2 5.8 ns (5.8×); parallel build 3.6× on 4 threads (200,000 vectors: 28.8 s → 8.1 s; recall@10 at ef = 64 0.9811 → 0.9806).

**Where it is slower on Apple M5** (every target where another library beats the engine):

* GIST, single-thread, recall 0.90: 2,715 vs FAISS HNSWFlat 3,959 QPS (-31%)
* GIST, single-thread, recall 0.95: 1,696 vs FAISS HNSWFlat 2,348 QPS (-28%)
* GIST, single-thread, recall 0.99: 581 vs FAISS HNSWFlat 727 QPS (-20%)
* GIST, batched, recall 0.99: 1,251 vs FAISS HNSWFlat 1,259 QPS (-1%)

**Where it is slower on Intel i7-13800H (WSL2)** (every target where another library beats the engine):

* GloVe-100, batched, recall 0.90: 10,303 vs hnswlib 10,447 QPS (-1%)
* GIST, single-thread, recall 0.90: 2,410 vs FAISS HNSWFlat 2,533 QPS (-5%)
* GIST, single-thread, recall 0.95: 1,341 vs FAISS HNSWFlat 1,674 QPS (-20%)
* GIST, batched, recall 0.90: 8,716 vs FAISS HNSWFlat 8,974 QPS (-3%)
<!-- results:end -->

## Testing and quality

* **C++ (GoogleTest, 54 tests):** every kernel vs the scalar reference for all
  dims 1–1024 (aligned and misaligned); recall vs a brute-force oracle for
  several dims and all metrics; edge cases (empty index, k > size, ef < k,
  duplicate labels, dim = 1, identical vectors); seed determinism; growth and
  capacity limits; save/load round trips in both modes; truncation and
  bit-flip fuzzing of the file format; filter correctness on a selectivity
  sweep; soft delete; concurrent `search_batch` and readers racing writers.
* **Python (pytest, 28 tests):** dtype / shape / contiguity handling, id
  handling, filters (callable, mask, id list), exceptions from callbacks,
  save/load, exact result parity with the C++ harness, determinism, GIL
  release and concurrent searches from Python threads.
* **Sanitizers:** the full C++ suite runs clean under ASan + UBSan and under
  ThreadSanitizer (presets `asan`, `tsan`).
* **CI:** GCC and Clang on Ubuntu, AppleClang on macOS (arm64 → NEON),
  ASan/UBSan, TSan, wheel build + pytest in a clean venv, clang-format check,
  and a smoke benchmark.

## Limitations

* Two laptops (Apple M5; Intel i7-13800H under WSL2) and one parameter
  setting (M = 16, ef_construction = 200); no server-class or AVX-512
  hardware in the published runs. On ARM, hnswlib runs without hand-written
  SIMD, which flatters the engine's lead over it. FAISS is faster than the
  engine on GIST (960-d) on both machines (see Results).
* GloVe-100 at M = 16 tops out around recall@10 = 0.94 for all three
  libraries even at ef = 640; higher recall needs a larger M.
* Index updates are append + soft delete only; deleted nodes stay in the graph
  and their space is never reclaimed.
* Inserts and searches do not overlap: mutations take an exclusive lock and
  wait for in-flight searches.
* `uint32` internal ids: at most ~4.29 billion vectors per index.
* Very selective filters (≈1% of labels allowed) cost noticeably more work
  and recall drops at a fixed ef (see BENCHMARKS.md).
* Max inner product on un-normalized data is harder for HNSW than L2/cosine
  (this affects all three libraries).
* POSIX only (mmap); no Windows build.
* Python labels are int64; labels ≥ 2⁶³ are not representable from Python.

## License

MIT — see [LICENSE](LICENSE).
