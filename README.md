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

> **About the benchmark data.** The machine these results were produced on had
> no network route to ann-benchmarks.com, so the numbers below use *synthetic
> stand-ins* with the exact shape and metric of SIFT-1M, GloVe-100 and GIST-1M
> (named `synth-sift`, `synth-glove`, `synth-gist`). They are not SIFT/GloVe/GIST
> results. `scripts/run_all_benchmarks.sh` (without `--synthetic`) runs the real
> datasets on a machine with network access.

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

### Reproduce a benchmark (≈ 10 minutes)

```bash
pip install numpy hnswlib faiss-cpu matplotlib
python scripts/fetch_data.py --synthetic sift --n 200000 --nq 1000     # or: python scripts/fetch_data.py sift
cmake --preset bench && cmake --build --preset bench --target bench_main
./build/bench/bench/bench_main --data data --name synth-sift --ef 10,20,40,80,160 --build-threads 0
HNSW_NATIVE=ON pip install .
python bench/python/compare.py --data data --name synth-sift --metric l2 --reps 1
python bench/python/plot.py      # bench/results/synth-sift.png + summary.md
```

The full pipeline (all datasets, ablation, scaling, filters, plots) is
`scripts/run_all_benchmarks.sh [--synthetic] [--quick]`.

## Results

RESULTS_PLACEHOLDER

## Testing and quality

* **C++ (GoogleTest, 53 tests):** every kernel vs the scalar reference for all
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

* Benchmarks in this repository were measured on synthetic data shaped like
  the standard datasets (see the note at the top), on one 4-core cloud VM.
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
