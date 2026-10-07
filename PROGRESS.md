# Progress

## Status

| # | Milestone | State | Evidence |
|---|---|---|---|
| M0 | Scaffolding | done | CMake presets dev/release/asan/tsan/bench; GoogleTest + Google Benchmark via FetchContent; `fetch_data.py`; fvecs/ivecs loaders + round-trip tests; CI workflow |
| M1 | Brute force + harness | done | `bench_main --mode brute` reaches recall 1.0 against independently computed (NumPy) ground truth; CSV with recall/QPS/p50/p95/p99 |
| M2 | HNSW (paper-faithful) | done | 10k uniform random vectors: recall@10 ≥ 0.95 at ef=128 (test); recall matches hnswlib on identical data |
| M3 | SIMD + layout + prefetch + visited pool | done | kernel tests for all dims 1–1024 (aligned + misaligned); microbench + ablation in `bench/results/` |
| M4 | Parallel build + batch search | done | 4.0× build speedup on 4 cores; TSan clean; all nodes reachable after parallel build (regression test); parallel-build recall within 1% of single-thread (test) |
| M5 | Python package | done | `pip install .` in a clean venv; 28 pytest tests incl. exact parity with the C++ harness |
| M6 | Persistence + mmap | done | round-trip identical results (both modes); truncation + bit-flip fuzz always throws; ASan/UBSan clean |
| M7 | Filtered search + soft delete | done | selectivity sweep 1/10/50/90% (tests + `filter.csv`); deleted labels never returned |
| M8 | Cross-library benchmarks | done (synthetic data) | `scripts/run_all_benchmarks.sh --synthetic` regenerates every CSV/PNG/table |
| M9 | Docs / polish | done | README, DESIGN.md, FORMAT.md, BENCHMARKS.md |

## Decisions

* **Datasets.** The development environment's network policy blocks
  `ann-benchmarks.com` (and other mirrors), so the real SIFT/GloVe/GIST files
  could not be downloaded. `fetch_data.py --synthetic` generates stand-ins with
  the same dimension, size and metric (Gaussian mixture on low-dimensional
  subspaces) with exact ground truth from NumPy. All published numbers are
  labelled `synth-*`; none are presented as SIFT/GloVe/GIST results. On a
  machine with network access, `scripts/run_all_benchmarks.sh` (no flag)
  downloads and benchmarks the real datasets.
* **GIST** comparison runs on a 200k subset by default (memory/time on a
  4-core, 15 GB machine); `GIST_SUBSET=0` runs the full set.
* Heuristic neighbor selection without `extendCandidates` /
  `keepPrunedConnections`; applied even with fewer than M candidates.
* Levels from splitmix64 for cross-platform determinism.
* Vector rows padded to 16 floats (64 B).
* Default ISA preference: AVX-512 > AVX2 > scalar (NEON on arm64);
  `HNSW_SIMD` env var can force a lower one.
* Payload checksum in the file format so every flipped byte is detected, not
  only structurally invalid ones.
* Inner-product recall thresholds in tests are lower (0.93 at ef=400) because
  MIPS on un-normalized clustered data is intrinsically harder for graph
  search; hnswlib scores lower on the same data.

## Known issues / limitations

* **Open:** on synth-sift the engine's recall saturates lower than hnswlib's
  at very high ef (0.9913 vs 0.9962 at ef = 640), costing it the ≥ 0.99
  target there; it matches or beats hnswlib's recall on synth-glove and
  synth-gist. Two hnswlib behaviours (single entry point between layers;
  skipping the heuristic below M candidates) were A/B tested and ruled out.
* **Fixed during M8:** parallel builds could drop edges (a node overwrote
  edges other threads had already added to it), leaving nodes unreachable on
  low-degree data. Found via the README example; fixed and regression-tested.
  Benchmarks were re-run after the fix.
* Throughput on the shared benchmark VM varies between runs (~16%
  single-thread, ~30% batched); see BENCHMARKS.md.
* Benchmarks are on synthetic data (see above).
* Python-level concurrency is tested for correctness, but the TSan run covers
  the C++ suite only (TSan-instrumenting CPython is out of scope).
* CI (GitHub Actions) is green on `main`: GCC + Clang on Ubuntu, AppleClang on
  macOS arm64 (NEON), ASan/UBSan, TSan, wheel + pytest on Linux and macOS,
  clang-format, smoke benchmark. The first macOS run exposed test data that
  differed between libstdc++ and libc++; tests now use a portable RNG.
* No Windows support (POSIX mmap).
