# Benchmarks

All numbers in this document were produced by `scripts/run_all_benchmarks.sh
--synthetic` and are copied from the CSVs in `bench/results/` (raw rows,
plots and the generated `summary.md` live there). Nothing here was typed in
by hand from memory.

## Read this first: the data is synthetic

The benchmark machine could not reach ann-benchmarks.com (blocked by its
network policy), so the real SIFT-1M / GloVe-100 / GIST-1M HDF5 files were not
available. `scripts/fetch_data.py --synthetic` generates stand-ins with the
**same dimension, base size, query count and metric**:

| name | dim | base | queries | metric | real counterpart |
|---|---:|---:|---:|---|---|
| synth-sift | 128 | 1,000,000 | 10,000 | L2 | SIFT-1M |
| synth-glove | 100 | 1,183,514 | 10,000 | cosine | GloVe-100 (angular) |
| synth-gist | 960 | 1,000,000 (200,000 used) | 1,000 | L2 | GIST-1M |

Vectors are drawn from a 256-component Gaussian mixture in which every
component lives on its own random 24-dimensional subspace plus small isotropic
noise (low intrinsic dimension, like real descriptors/embeddings). Exact
top-100 ground truth is computed with NumPy (float32 shortlist, float64
re-rank) — independently of the engine. Absolute recall/QPS on real data will
differ; the *relative* comparison between libraries under identical
conditions is the point of these numbers. Run the script without
`--synthetic` on a networked machine to benchmark the real files.

## Setup

ENVIRONMENT_PLACEHOLDER

* Engine built with `-O3 -march=native` (`bench` preset for the C++ harness,
  `HNSW_NATIVE=ON pip install .` for the Python package). hnswlib 0.8.0 is
  distributed as an sdist and was compiled locally with its default
  `-O3 -march=native`. faiss-cpu 1.15.1 is the PyPI wheel, which dispatches at
  runtime to its AVX-512 build on this CPU (`get_compile_options()` →
  `OPTIMIZE DD AVX2 AVX512`).
* All libraries: `M = 16`, `ef_construction = 200`, `k = 10`, same metric
  (FAISS cosine = inner product on normalized vectors, which is how the other
  two implement cosine internally), same thread count (4; FAISS via
  `omp_set_num_threads`).
* Every library runs in its own subprocess (`compare.py`), so peak RSS is
  isolated per library.
* Search comparisons use a pre-built index (standard ann-benchmarks method):
  build once, then sweep `ef_search ∈ {10, 20, 40, 80, 160, 320, 640}`.
* Two query modes: **single** — one thread, one query per Python call
  (per-query latency p50/p95/p99, includes the Python call overhead for every
  library alike); **batch** — all queries in one call on 4 threads
  (throughput). Each point: warm-up pass, then 3 repetitions, median by QPS.
* Recall@10 = |returned ∩ true top-10| / 10, averaged over queries (id
  overlap; with exact distance ties a correct answer could count as a miss —
  ties do not occur in this continuous synthetic data).
* Build metrics: wall time with 4 threads and with 1 thread (synth-sift),
  RSS growth during the build, peak RSS of the process, serialized file size.
* GIST is run on a 200k subset (`GIST_SUBSET`, ground truth recomputed
  exactly for the subset) to keep three 960-d builds within the 15 GB / 4-core
  budget of the benchmark machine.

## Results

RESULTS_PLACEHOLDER
