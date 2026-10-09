# Benchmarks

All numbers in this document were produced by `scripts/run_all_benchmarks.sh`
and are rendered from the CSVs in `bench/results/` by `bench/python/report.py`
(raw rows, plots and the generated `summary.md` live there). Nothing here was
typed in by hand.

## Data

The real [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) HDF5
files, downloaded and converted by `scripts/fetch_data.py` (SHA-256 printed on
download):

| dataset | dim | base | queries | metric | ground truth |
|---|---:|---:|---:|---|---|
| SIFT-1M | 128 | 1,000,000 | 10,000 | L2 | provided top-100 |
| GloVe-100 | 100 | 1,183,514 | 10,000 | cosine (angular) | provided top-100 |
| GIST-1M | 960 | 200,000 of 1,000,000 | 1,000 | L2 | recomputed exactly for the subset (NumPy) |

GIST runs on its first 200k vectors (`GIST_SUBSET`) so three 960-d indexes fit
comfortably in 16 GB of RAM; set `GIST_SUBSET=0` on a larger machine.

### Earlier synthetic run

Before the real files were reachable, the pipeline was developed and run on
synthetic stand-ins of the same shapes (`--synthetic`, Gaussian mixtures on
low-dimensional subspaces) on a 4-core Intel Xeon cloud VM with AVX-512. Those
raw CSVs and plots are kept as `bench/results/synth-*` for reference but are not
used in the tables below. One observation from that run, a slightly lower
high-recall ceiling than hnswlib on synthetic SIFT, does **not** reproduce on
real SIFT-1M (max recall 0.9993 vs hnswlib 0.9992).

## Setup

<!-- env:begin -->
```
commit: 0daf0bc3ff2198cb690d3df9e23b2a754307e929
date: 2026-10-09T00:33Z
cpu: Apple M5 (4 performance + 6 efficiency cores)
cores: 10 (threads used: 4)
memory: 16 GB
os: Darwin 25.6.0 arm64
compiler: Apple clang version 21.0.0 (clang-2100.1.1.101)
python: Python 3.9.6
hnswlib: 0.8.0
faiss-cpu: 1.13.0
numpy: 2.0.2
engine simd: neon
```
<!-- env:end -->

* Hardware and software versions are in the block above
  (`bench/results/environment.txt`). The CPU has 4 performance and 6
  efficiency cores; every multi-threaded step used **4 threads** (the
  performance cores) for all three libraries.
* Engine built with `-O3` and `-mcpu=native` (`bench` preset for the C++
  harness, `HNSW_NATIVE=ON pip install .` for the Python package), using its
  NEON kernels. hnswlib 0.8.0 is an sdist compiled locally with its default
  flags; its hand-written SIMD kernels are x86-only (SSE/AVX), so on ARM it
  runs its generic distance code. faiss-cpu 1.13.0 is the PyPI wheel (newest
  available for the Python 3.9 used here) and has NEON kernels, so **FAISS is
  the like-for-like comparison on this machine**.
* All libraries: `M = 16`, `ef_construction = 200`, `k = 10`, same metric
  (FAISS cosine = inner product on normalized vectors, which is how the other
  two implement cosine internally), same thread count (FAISS via
  `omp_set_num_threads`).
* Every library runs in its own subprocess (`compare.py`), so peak memory is
  isolated per library.
* Search comparisons use a pre-built index (standard ann-benchmarks method):
  build once, then sweep `ef_search ∈ {10, 20, 40, 80, 160, 320, 640}`.
* Two query modes: **single**: one thread, one query per Python call
  (per-query latency p50/p95/p99, includes the Python call overhead for every
  library alike); **batch**: all queries in one call on 4 threads
  (throughput). Each point: warm-up pass, then 3 repetitions, median by QPS.
* Recall@10 = |returned ∩ true top-10| / 10, averaged over queries (id
  overlap). SIFT vectors are integer-valued, so exact distance ties occur and
  a correct neighbour can occasionally count as a miss; this is why brute
  force scores 0.9991 rather than 1.0 against the provided ground truth.
* Build metrics: wall time with 4 threads and with 1 thread (SIFT), peak RSS
  of the process and serialized index size. The "RSS growth" column is shown
  as "—" for this run: on macOS the harness at this commit recorded peak
  instead of current RSS, which makes growth meaningless (fixed for future
  runs; `compare.py` now asks `ps`).
* Throughput on a laptop depends on thermals and background load; the run
  was made plugged in, with other applications closed.

## Results

<!-- results:begin -->
![sift](../bench/results/sift.png)
![glove](../bench/results/glove.png)
![gist](../bench/results/gist.png)

### gist

n=200000, dim=960, metric=l2, M=16, ef_construction=200, k=10, threads=4

**Best QPS at recall@10 ≥ threshold (single-thread, one query per call)**

| library | ≥0.90 | ≥0.95 | ≥0.99 |
|---|---:|---:|---:|
| hnsw-engine (this) | 1,685 | 1,685 | 559 |
| hnswlib | 770 | 770 | 263 |
| FAISS HNSWFlat | 2,300 | 2,300 | 680 |

**Best QPS at recall@10 ≥ threshold (batched, multi-thread)**

| library | ≥0.90 | ≥0.95 | ≥0.99 |
|---|---:|---:|---:|
| hnsw-engine (this) | 3,626 | 3,626 | 1,204 |
| hnswlib | 2,910 | 2,910 | 984 |
| FAISS HNSWFlat | 3,468 | 3,468 | 1,189 |

**Recall / QPS / latency per ef (single-thread)**

| library | ef | recall@10 | QPS | p50 µs | p99 µs |
|---|---:|---:|---:|---:|---:|
| hnsw-engine (this) | 10 | 0.5048 | 12,699 | 77 | 128 |
| hnsw-engine (this) | 20 | 0.6494 | 8,351 | 119 | 181 |
| hnsw-engine (this) | 40 | 0.7869 | 5,120 | 200 | 268 |
| hnsw-engine (this) | 80 | 0.8910 | 2,955 | 352 | 437 |
| hnsw-engine (this) | 160 | 0.9507 | 1,685 | 618 | 777 |
| hnsw-engine (this) | 320 | 0.9794 | 957 | 1,091 | 1,355 |
| hnsw-engine (this) | 640 | 0.9908 | 559 | 1,864 | 2,341 |
| hnswlib | 10 | 0.4940 | 5,373 | 184 | 290 |
| hnswlib | 20 | 0.6515 | 3,554 | 280 | 430 |
| hnswlib | 40 | 0.7871 | 2,259 | 455 | 617 |
| hnswlib | 80 | 0.8909 | 1,338 | 778 | 971 |
| hnswlib | 160 | 0.9504 | 770 | 1,351 | 1,680 |
| hnswlib | 320 | 0.9791 | 444 | 2,344 | 2,946 |
| hnswlib | 640 | 0.9904 | 263 | 3,978 | 4,986 |
| FAISS HNSWFlat | 10 | 0.5175 | 17,831 | 55 | 88 |
| FAISS HNSWFlat | 20 | 0.6687 | 11,578 | 86 | 130 |
| FAISS HNSWFlat | 40 | 0.7966 | 7,153 | 142 | 187 |
| FAISS HNSWFlat | 80 | 0.8965 | 4,106 | 252 | 316 |
| FAISS HNSWFlat | 160 | 0.9520 | 2,300 | 452 | 541 |
| FAISS HNSWFlat | 320 | 0.9776 | 1,263 | 821 | 985 |
| FAISS HNSWFlat | 640 | 0.9915 | 680 | 1,522 | 1,814 |

**Build**

| library | version | build s (N threads) | build s (1 thread) | index file MB | RSS growth MB | peak RSS MB |
|---|---|---:|---:|---:|---:|---:|
| hnsw-engine (this) | 0.1.0 | 54.7 | — | 760 | — | 1528 |
| hnswlib | 0.8.0 | 82.8 | — | 761 | — | 1557 |
| FAISS HNSWFlat | 1.13.0 | 59.6 | — | 760 | — | 1572 |

### glove

n=1183514, dim=100, metric=cosine, M=16, ef_construction=200, k=10, threads=4

**Best QPS at recall@10 ≥ threshold (single-thread, one query per call)**

| library | ≥0.90 | ≥0.95 | ≥0.99 |
|---|---:|---:|---:|
| hnsw-engine (this) | 1,705 | — | — |
| hnswlib | 1,144 | — | — |
| FAISS HNSWFlat | 2,437 | — | — |

**Best QPS at recall@10 ≥ threshold (batched, multi-thread)**

| library | ≥0.90 | ≥0.95 | ≥0.99 |
|---|---:|---:|---:|
| hnsw-engine (this) | 6,266 | — | — |
| hnswlib | 4,040 | — | — |
| FAISS HNSWFlat | 8,931 | — | — |

**Recall / QPS / latency per ef (single-thread)**

| library | ef | recall@10 | QPS | p50 µs | p99 µs |
|---|---:|---:|---:|---:|---:|
| hnsw-engine (this) | 10 | 0.4699 | 46,890 | 20 | 38 |
| hnsw-engine (this) | 20 | 0.5979 | 29,153 | 32 | 62 |
| hnsw-engine (this) | 40 | 0.7025 | 17,702 | 54 | 96 |
| hnsw-engine (this) | 80 | 0.7871 | 10,295 | 96 | 159 |
| hnsw-engine (this) | 160 | 0.8505 | 5,810 | 174 | 255 |
| hnsw-engine (this) | 320 | 0.8988 | 3,170 | 322 | 431 |
| hnsw-engine (this) | 640 | 0.9360 | 1,705 | 600 | 790 |
| hnswlib | 10 | 0.4709 | 29,669 | 32 | 61 |
| hnswlib | 20 | 0.5955 | 18,750 | 51 | 96 |
| hnswlib | 40 | 0.7036 | 11,466 | 84 | 152 |
| hnswlib | 80 | 0.7875 | 6,715 | 147 | 239 |
| hnswlib | 160 | 0.8504 | 3,809 | 265 | 392 |
| hnswlib | 320 | 0.8990 | 2,117 | 484 | 646 |
| hnswlib | 640 | 0.9358 | 1,144 | 895 | 1,168 |
| FAISS HNSWFlat | 10 | 0.4966 | 32,959 | 29 | 49 |
| FAISS HNSWFlat | 20 | 0.6210 | 22,696 | 42 | 72 |
| FAISS HNSWFlat | 40 | 0.7247 | 14,564 | 66 | 110 |
| FAISS HNSWFlat | 80 | 0.8035 | 8,263 | 119 | 188 |
| FAISS HNSWFlat | 160 | 0.8625 | 4,655 | 215 | 301 |
| FAISS HNSWFlat | 320 | 0.9076 | 2,437 | 415 | 534 |
| FAISS HNSWFlat | 640 | 0.9421 | 1,175 | 858 | 1,056 |

**Build**

| library | version | build s (N threads) | build s (1 thread) | index file MB | RSS growth MB | peak RSS MB |
|---|---|---:|---:|---:|---:|---:|
| hnsw-engine (this) | 0.1.0 | 65.6 | — | 671 | — | 1270 |
| hnswlib | 0.8.0 | 108.7 | — | 619 | — | 1282 |
| FAISS HNSWFlat | 1.13.0 | 96.3 | — | 614 | — | 1798 |

### sift

n=1000000, dim=128, metric=l2, M=16, ef_construction=200, k=10, threads=4

**Best QPS at recall@10 ≥ threshold (single-thread, one query per call)**

| library | ≥0.90 | ≥0.95 | ≥0.99 |
|---|---:|---:|---:|
| hnsw-engine (this) | 21,666 | 12,414 | 6,910 |
| hnswlib | 11,127 | 6,424 | 3,608 |
| FAISS HNSWFlat | 17,884 | 10,062 | 5,419 |

**Best QPS at recall@10 ≥ threshold (batched, multi-thread)**

| library | ≥0.90 | ≥0.95 | ≥0.99 |
|---|---:|---:|---:|
| hnsw-engine (this) | 82,278 | 46,978 | 26,101 |
| hnswlib | 41,403 | 23,841 | 13,377 |
| FAISS HNSWFlat | 70,989 | 37,933 | 19,518 |

**Recall / QPS / latency per ef (single-thread)**

| library | ef | recall@10 | QPS | p50 µs | p99 µs |
|---|---:|---:|---:|---:|---:|
| hnsw-engine (this) | 10 | 0.7077 | 54,808 | 18 | 29 |
| hnsw-engine (this) | 20 | 0.8390 | 35,743 | 28 | 42 |
| hnsw-engine (this) | 40 | 0.9272 | 21,666 | 47 | 65 |
| hnsw-engine (this) | 80 | 0.9746 | 12,414 | 83 | 106 |
| hnsw-engine (this) | 160 | 0.9931 | 6,910 | 150 | 190 |
| hnsw-engine (this) | 320 | 0.9983 | 3,829 | 272 | 348 |
| hnsw-engine (this) | 640 | 0.9993 | 2,151 | 483 | 633 |
| hnswlib | 10 | 0.7104 | 27,433 | 36 | 60 |
| hnswlib | 20 | 0.8390 | 18,216 | 55 | 84 |
| hnswlib | 40 | 0.9273 | 11,127 | 92 | 126 |
| hnswlib | 80 | 0.9749 | 6,424 | 162 | 206 |
| hnswlib | 160 | 0.9932 | 3,608 | 289 | 368 |
| hnswlib | 320 | 0.9983 | 2,012 | 518 | 669 |
| hnswlib | 640 | 0.9992 | 1,129 | 923 | 1,222 |
| FAISS HNSWFlat | 10 | 0.7232 | 41,544 | 24 | 36 |
| FAISS HNSWFlat | 20 | 0.8501 | 28,472 | 35 | 51 |
| FAISS HNSWFlat | 40 | 0.9365 | 17,884 | 57 | 75 |
| FAISS HNSWFlat | 80 | 0.9797 | 10,062 | 102 | 127 |
| FAISS HNSWFlat | 160 | 0.9948 | 5,419 | 190 | 234 |
| FAISS HNSWFlat | 320 | 0.9986 | 2,795 | 368 | 451 |
| FAISS HNSWFlat | 640 | 0.9992 | 1,342 | 762 | 920 |

**Build**

| library | version | build s (N threads) | build s (1 thread) | index file MB | RSS growth MB | peak RSS MB |
|---|---|---:|---:|---:|---:|---:|
| hnsw-engine (this) | 0.1.0 | 42.5 | 153.8 | 628 | — | 1248 |
| hnswlib | 0.8.0 | 82.2 | 298.2 | 630 | — | 1310 |
| FAISS HNSWFlat | 1.13.0 | 56.3 | 198.0 | 626 | — | 1347 |

### Ablation (bench_main, engine only)

| label | isa | prefetch | search_threads | ef | recall | qps | p50_us | p99_us |
|---|---|---|---|---|---|---|---|---|
| 1 scalar kernels (no prefetch) | scalar | 0 | 1 | 64 | 0.96344 | 7829.9 | 132.0 | 171.1 |
| 2 +NEON kernels | neon | 0 | 1 | 64 | 0.96344 | 13877.9 | 74.2 | 95.8 |
| 3 +prefetch | neon | 1 | 1 | 64 | 0.96344 | 15234.0 | 67.7 | 87.6 |
| 5 +4 search threads | neon | 1 | 4 | 64 | 0.96344 | 55323.7 | nan | nan |

### Build thread scaling (bench_main, engine only)

| build_threads | n | build_s | ef | recall |
|---|---|---|---|---|
| 1 | 200000 | 20.985 | 64 | 0.98110 |
| 2 | 200000 | 10.652 | 64 | 0.98080 |
| 4 | 200000 | 5.525 | 64 | 0.98100 |

Notes on the tables:

* Ablation and scaling rows come from the C++ harness (`bench_main`). The ablation loads one
  pre-built index, so every row searches the identical graph (identical recall); the last row is
  batched, so it has no per-query latency.
* gist peak RSS is dominated by loading the 960-d dataset in each worker, so it is the same
  for all three libraries; compare the RSS-growth column instead.
* Brute force on sift (1,000 queries) reaches recall 0.99910 against the provided ground truth (`bruteforce.csv`), validating the harness. Anything below 1.0 comes from exact distance ties (SIFT vectors are integer-valued, so neighbours at rank 10 and 11 can be equidistant and either is correct); recall here counts id overlap, as the spec defines it.

### Filtered search (engine, sift[:200000], random allow-lists)

| allowed | ef | recall@10 | QPS | filter violations |
|---:|---:|---:|---:|---:|
| 1% | 40 | 1.0000 | 2,893 | 0 |
| 1% | 80 | 1.0000 | 1,632 | 0 |
| 1% | 160 | 1.0000 | 936 | 0 |
| 1% | 320 | 1.0000 | 538 | 0 |
| 10% | 40 | 0.9984 | 17,597 | 0 |
| 10% | 80 | 0.9999 | 10,227 | 0 |
| 10% | 160 | 1.0000 | 5,860 | 0 |
| 10% | 320 | 1.0000 | 3,392 | 0 |
| 50% | 40 | 0.9818 | 59,612 | 0 |
| 50% | 80 | 0.9967 | 33,557 | 0 |
| 50% | 160 | 0.9997 | 19,487 | 0 |
| 50% | 320 | 0.9999 | 11,272 | 0 |
| 90% | 40 | 0.9596 | 93,512 | 0 |
| 90% | 80 | 0.9902 | 56,005 | 0 |
| 90% | 160 | 0.9977 | 30,450 | 0 |
| 90% | 320 | 0.9996 | 17,659 | 0 |

Every returned id satisfied the filter. Recall stays high at low selectivity because the search
keeps exploring until it has `ef` eligible results; the cost is throughput.

### Kernel microbenchmarks (Google Benchmark, ns per call)

| dim | scalar L2 | NEON L2 | scalar dot | NEON dot |
|---:|---:|---:|---:|---:|
| 16 | 2.2 | 1.4 | 1.8 | 1.2 |
| 100 | 17.6 | 4.3 | 16.4 | 4.1 |
| 128 | 26.3 | 5.2 | 22.9 | 5.2 |
| 384 | 112.5 | 15.7 | 108.2 | 15.1 |
| 768 | 288.9 | 33.4 | 277.9 | 31.0 |
| 960 | 381.2 | 42.3 | 374.1 | 39.4 |
| 1024 | 410.4 | 46.0 | 417.1 | 43.8 |

## Resume-ready summary (measured; SIFT-1M / GIST-1M)

* Built an HNSW vector search engine from scratch in C++20 (Malkov & Yashunin, Algorithms 1–5) with AVX2/AVX-512/NEON kernels and runtime CPU dispatch: **5.0× faster L2 kernel** (NEON vs scalar, d = 128) and **1.9× single-thread QPS from SIMD + prefetching** at identical recall (1M × 128).
* 1,000,000 × 128, l2, recall@10 ≥ 0.95, single thread: **12,414 QPS vs hnswlib 6,424 (+93%) and FAISS HNSWFlat 10,062 (+23%)**. Slower than FAISS HNSWFlat on 200,000 × 960, l2 (-27% at recall ≥ 0.95). 
* Parallel build scales 3.8× on 4 threads (200,000 vectors: 21.0 s → 5.5 s; recall@10 at ef=64 0.9811 → 0.9810). 1M-vector build in 42 s on 4 threads (hnswlib 82 s, FAISS 56 s) at the same index size.
* Memory-mapped, checksummed on-disk format whose loader rejects every truncated or bit-flipped file in fuzz tests; ASan/UBSan- and TSan-clean; GoogleTest + pytest suites; pybind11 package that releases the GIL and matches the C++ results exactly.
<!-- results:end -->
