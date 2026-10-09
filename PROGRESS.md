# Progress

## Status

| # | Milestone | State | Evidence |
|---|---|---|---|
| M0 | Scaffolding | done | CMake presets dev/release/asan/tsan/bench; GoogleTest + Google Benchmark via FetchContent; `fetch_data.py`; fvecs/ivecs loaders + round-trip tests; CI workflow |
| M1 | Brute force + harness | done | `bench_main --mode brute` reaches recall 1.0 against NumPy ground truth (synthetic) and 0.9991 against the provided SIFT-1M ground truth (difference = exact distance ties in integer-valued SIFT); CSV with recall/QPS/p50/p95/p99 |
| M2 | HNSW (paper-faithful) | done | 10k uniform random vectors: recall@10 ≥ 0.95 at ef=128 (test); recall matches hnswlib on identical data |
| M3 | SIMD + layout + prefetch + visited pool | done | kernel tests for all dims 1–1024 (aligned + misaligned); microbench + ablation in `bench/results/` |
| M4 | Parallel build + batch search | done | 4.0× build speedup on 4 cores; TSan clean; all nodes reachable after parallel build (regression test); parallel-build recall within 1% of single-thread (test) |
| M5 | Python package | done | `pip install .` in a clean venv; 28 pytest tests incl. exact parity with the C++ harness |
| M6 | Persistence + mmap | done | round-trip identical results (both modes); truncation + bit-flip fuzz always throws; ASan/UBSan clean |
| M7 | Filtered search + soft delete | done | selectivity sweep 1/10/50/90% (tests + `filter.csv`); deleted labels never returned |
| M8 | Cross-library benchmarks | done | real SIFT-1M / GloVe-100 / GIST (200k) on an Apple M5 (NEON, 4 threads) and an Intel i7-13800H under WSL2 (AVX2, 6 threads); per-machine folders in `bench/results/<machine>/`; `scripts/run_all_benchmarks.sh` regenerates every CSV/PNG/table and the README/BENCHMARKS results sections from all machines |
| M9 | Docs / polish | done | README, DESIGN.md, FORMAT.md, BENCHMARKS.md |

## Decisions

* **Datasets.** The development cloud environment could not reach
  ann-benchmarks.com, so the pipeline was first built and run on synthetic
  stand-ins (`fetch_data.py --synthetic`; results kept as
  `bench/results/synthetic-xeon-vm/`). The published results are from the
  real datasets, run on the owner's Apple M5 MacBook Pro and an HP ZBook
  (i7-13800H, WSL2).
* **QPS at a recall target** is interpolated on the recall-vs-QPS curve
  (log-QPS linear in recall between the bracketing ef points) rather than
  taken from the best measured point above the target, which produced
  threshold cliffs (e.g. recall 0.9497 vs target 0.95).
* **GIST** comparison runs on a 200k subset by default (fits 16 GB RAM);
  `GIST_SUBSET=0` runs the full set.
* Heuristic neighbor selection without `extendCandidates` /
  `keepPrunedConnections`; applied even with fewer than M candidates.
* Levels from splitmix64 for cross-platform determinism.
* Vector rows padded to 16 floats (64 B).
* Default ISA preference: AVX-512 > AVX2 > scalar (NEON on arm64);
  `HNSW_SIMD` env var can force a lower one.
* Payload checksum in the file format so every flipped byte is detected, not
  only structurally invalid ones.
* Inner-product recall thresholds in tests are lower (0.93 at ef=1000) because
  MIPS on un-normalized clustered data is intrinsically harder for graph
  search; hnswlib scores lower on the same data.

## Known issues / limitations

* **Resolved / not reproduced:** on synthetic SIFT the engine's recall
  saturated slightly below hnswlib's at ef = 640; on real SIFT-1M it does not
  (0.9993 vs 0.9992).
* **Measured gaps (real data, interpolated):** FAISS is faster than the
  engine single-threaded on GIST-960 on both machines (Apple M5: ≈28% at
  recall 0.95; i7-13800H: ≈20%); a few batched points trail by small
  margins (see README). The earlier GloVe gap was a threshold artefact and
  disappears with interpolation. hnswlib lacks NEON kernels, so the engine's
  lead over it on ARM is partly a platform effect; on x86 (like-for-like) the
  engine leads hnswlib by ≈11% on SIFT at 0.95.
* **Run on macOS:** the "RSS growth" metric was invalid there (peak instead
  of current RSS); fixed in `compare.py`, shown as "—" for that run.
* **Fixed during M8:** parallel builds could drop edges (a node overwrote
  edges other threads had already added to it), leaving nodes unreachable on
  low-degree data. Found via the README example; fixed and regression-tested.
  Benchmarks were re-run after the fix.
* On the shared cloud VM (synthetic run), throughput varied between runs by
  ~16% single-thread and ~30% batched; each laptop run was a single pass.
* Python-level concurrency is tested for correctness, but the TSan run covers
  the C++ suite only (TSan-instrumenting CPython is out of scope).
* CI (GitHub Actions) is green on `main`: GCC + Clang on Ubuntu, AppleClang on
  macOS arm64 (NEON), ASan/UBSan, TSan, wheel + pytest on Linux and macOS,
  clang-format, smoke benchmark. The first macOS run exposed test data that
  differed between libstdc++ and libc++; tests now use a portable RNG.
* No Windows support (POSIX mmap).
