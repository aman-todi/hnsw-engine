# SPEC: HNSW Vector Search Engine (C++20 + Python bindings)

Working name: `hnsw-engine` (rename freely). Build an approximate nearest-neighbor (ANN) search library from scratch: an HNSW index in C++20 with SIMD distance kernels, multithreading, persistence, filtered search, pybind11 Python bindings, and a reproducible benchmark against **hnswlib** and **FAISS (IndexHNSWFlat)** on standard public datasets.

This is a portfolio/resume project. Priorities, in order: **correctness → measurable performance → engineering rigor (tests, sanitizers, CI, docs)**. Every performance claim must be backed by a reproducible benchmark.

## 0. How to work (instructions for Claude Code)

- Work **one milestone at a time** (section 6). Do not start a milestone until the previous one's acceptance criteria pass.
- Commit at the end of each milestone with a clear message. Keep `PROGRESS.md` updated (done / in progress / decisions / known issues).
- Write tests alongside code, not after. Never weaken a test to make it pass.
- No placeholder or stubbed implementations in finished milestones. If something is ambiguous, pick the simplest reasonable option, note it in `docs/DESIGN.md`, and continue.
- Do not copy code from hnswlib or FAISS. Implement from the HNSW paper (Malkov & Yashunin, 2016/2018), Algorithms 1–5.
- Never report benchmark numbers you did not measure. Results are produced by scripts and written to `bench/results/`.

## 1. Goals and non-goals

**Goals**
- Dense float32 vectors; metrics: L2 (squared), inner product, cosine.
- HNSW build + k-NN search with tunable `M`, `ef_construction`, `ef_search`.
- AVX2/FMA kernels (plus NEON on arm64, scalar fallback), runtime CPU dispatch on x86.
- Parallel build and parallel batch search; searches use no per-node locking and run concurrently with each other.
- Save/load, with memory-mapped load.
- Filtered search (predicate / allow-list).
- Python package (`pip install .`) with a NumPy API that releases the GIL.
- Benchmarks on SIFT-1M, GloVe-100, GIST-1M vs hnswlib and FAISS.

**Non-goals**: GPU, distributed serving, product quantization/IVF, a network server, updates beyond append and soft-delete. (Optional stretch items are listed in section 8.)

## 2. Datasets

Use the ann-benchmarks HDF5 files as the single source (each contains `train`, `test`, `neighbors`, `distances`; ground truth is top-100 per query).

| Name | Dim | Train | Queries | Metric | Source file |
|---|---|---|---|---|---|
| SIFT-1M | 128 | 1,000,000 | 10,000 | L2 | `http://ann-benchmarks.com/sift-128-euclidean.hdf5` |
| GloVe-100 | 100 | ~1,183,514 | 10,000 | cosine | `http://ann-benchmarks.com/glove-100-angular.hdf5` |
| GIST-1M | 960 | 1,000,000 | 1,000 | L2 | `http://ann-benchmarks.com/gist-960-euclidean.hdf5` |

- `scripts/fetch_data.py`: download (with resume + checksum print) into `data/` (gitignored), then convert each HDF5 into raw files the C++ harness can read without an HDF5 dependency: `<name>_base.fvecs`, `<name>_query.fvecs`, `<name>_gt.ivecs` (fvecs/ivecs = per vector: int32 dim, then dim values).
- Memory note: GIST-1M needs ~3.8 GB for raw vectors alone. Support `--subset N` everywhere so development can run on 100k–200k vectors.
- Recall@k = |returned ∩ true top-k| / k, averaged over queries (id overlap; mention the tie caveat in docs).

## 3. Repository layout

```
CMakeLists.txt, CMakePresets.json, pyproject.toml, README.md, spec.md, PROGRESS.md, LICENSE
include/hnsw/        # public headers: index.hpp, metric.hpp, filter.hpp, io.hpp
src/                 # index.cpp, search_layer.cpp, kernels_{scalar,avx2,neon}.cpp, dispatch.cpp, io.cpp, threadpool.cpp
python/              # bindings.cpp, hnsw_engine/__init__.py, type stubs (.pyi)
tests/               # cpp/ (GoogleTest), py/ (pytest)
bench/               # cpp harness (bench_main.cpp), python/ (compare.py, plot.py), results/ (CSV + PNG)
scripts/             # fetch_data.py, run_all_benchmarks.sh
docs/                # DESIGN.md, BENCHMARKS.md, FORMAT.md (on-disk format)
.github/workflows/   # ci.yml
```

Tooling: CMake ≥ 3.20, C++20, GoogleTest + Google Benchmark via `FetchContent`, pybind11 ≥ 2.11, `scikit-build-core` for the Python build, `.clang-format`, `.clang-tidy`. Warnings: `-Wall -Wextra -Wpedantic -Werror`. Release flags: `-O3` (and `-march=native` only for the bench preset; the shipped library uses runtime dispatch).

## 4. Public API

### C++ (`namespace hnsw`)

```cpp
enum class Metric { L2, InnerProduct, Cosine };

struct Params { size_t dim; Metric metric = Metric::L2;
                size_t M = 16; size_t ef_construction = 200;
                size_t max_elements = 0;   // 0 = grow geometrically
                uint64_t seed = 100; };

struct Neighbor { uint64_t label; float distance; };

class Index {
 public:
  explicit Index(const Params&);
  void add(const float* vec, uint64_t label);                         // single insert
  void add_batch(const float* vecs, const uint64_t* labels, size_t n,
                 size_t num_threads = 0);                             // 0 = hardware_concurrency
  std::vector<Neighbor> search(const float* q, size_t k, size_t ef = 0,
                               const Filter* f = nullptr) const;      // ef=0 -> default
  void search_batch(const float* qs, size_t nq, size_t k, size_t ef,
                    Neighbor* out /* nq*k */, size_t num_threads = 0,
                    const Filter* f = nullptr) const;
  void mark_deleted(uint64_t label);                                  // soft delete (tombstone)
  size_t size() const; size_t dim() const;
  void set_ef(size_t ef);
  void save(const std::string& path) const;
  static Index load(const std::string& path, bool mmap = false);
};
```

- `Filter`: `std::function<bool(uint64_t label)>` wrapper and a bitset-backed variant. Filtered-out nodes are still **traversed** (needed for graph connectivity) but excluded from results.
- Cosine: normalize at insert and at query, then use inner-product distance `1 - dot`. Do not mutate caller buffers.
- Errors: throw `std::invalid_argument` / `std::runtime_error` at the API boundary (bad dim, duplicate label, k = 0, corrupt file). `search` with `k > size()` returns `size()` results.

### Python (`import hnsw_engine`)

```python
idx = hnsw_engine.Index(dim=128, metric="l2", M=16, ef_construction=200)
idx.add(vectors, ids=None, num_threads=0)           # float32 (n, dim); ids default to arange
ids, dists = idx.search(queries, k=10, ef=64, num_threads=0, filter=None)  # (nq,k) int64, float32
idx.set_ef(64); idx.mark_deleted(i); len(idx)
idx.save(path); hnsw_engine.Index.load(path, mmap=True)
```

Accept C-contiguous float32 arrays (convert or raise clearly otherwise). Release the GIL in `add`/`search`. Ship `.pyi` stubs.

## 5. Design requirements

**Algorithm (paper-faithful)**
- Level: `floor(-ln(U(0,1)) * mL)`, `mL = 1/ln(M)`. Max neighbors: `M` on layers ≥ 1, `M0 = 2M` on layer 0.
- Insert: greedy descent (ef = 1) from the top layer to `level+1`; at each layer ≤ `level` run `search_layer` with `ef_construction`, pick neighbors with the **heuristic selection** (Algorithm 4: keep a candidate only if it is closer to the query than to every already-selected neighbor), link bidirectionally, and re-prune any neighbor list that overflows.
- `search_layer`: min-heap of candidates + max-heap of results, terminate when the nearest candidate is farther than the worst result and the result set is full.
- Entry point and max level update atomically on the first insert at a new top level.

**Memory layout (this is where performance comes from)**
- Vectors in one contiguous 64-byte-aligned arena (custom aligned allocator, RAII, no raw owning pointers). Optional zero-padding of each vector to a multiple of the SIMD width.
- Layer-0 adjacency as a flat fixed-stride array: per node `[count:uint32][neighbors: M0 × uint32]`. Upper layers stored sparsely (only nodes with level ≥ 1).
- Internal ids are `uint32`; external labels `uint64` via a label↔id map.
- **Visited set**: per-thread versioned array (epoch counter) from a pool; never clear per query.
- Prefetch (`__builtin_prefetch`) the next candidate's vector and neighbor list while processing the current one.

**SIMD kernels**
- Functions: `l2_sq(a,b,dim)`, `dot(a,b,dim)`. Implementations: scalar (reference), AVX2+FMA, NEON; optional AVX-512.
- x86: select at startup with `__builtin_cpu_supports`; function pointer set once, not per call. Compile each kernel TU with only its own ISA flags.
- Handle any `dim` (e.g., 100 is not a multiple of 8): vector main loop + scalar tail. Use multiple accumulators to hide FMA latency.
- Every kernel must match the scalar reference within a relative tolerance (≈1e-5) in tests.

**Concurrency**
- Parallel `add_batch`: per-node striped locks for neighbor-list updates, a global lock only for entry-point/max-level changes; work distribution via an atomic counter over a simple thread pool.
- Search only reads the graph (no per-node locks, no writes to shared state besides the pooled visited lists); `search_batch` parallelizes across queries.
- **Contract:** searches run concurrently with each other; mutating calls (`add`, `add_batch`, `mark_deleted`, `load`) are exclusive. Enforce with one `std::shared_mutex` at the API boundary so misuse blocks instead of corrupting memory. Document in `DESIGN.md`; test with concurrent C++ and Python threads (TSan-clean).
- Truly concurrent insert + search is a stretch goal (section 8).
- Must be clean under ThreadSanitizer for the supported usage.

**Persistence**
- Define a versioned binary format in `docs/FORMAT.md`: magic, version, endianness marker, params, counts, then the vector arena, label map, level array, and adjacency arrays at aligned offsets.
- `load(mmap=true)` maps the file and uses the arrays in place (read-only index). `load` validates magic/version/sizes and throws on any inconsistency; a corrupted or truncated file must never crash.

## 6. Milestones and acceptance criteria

| # | Milestone | Acceptance |
|---|---|---|
| M0 | Scaffolding: CMake + presets (`dev`, `release`, `asan`, `tsan`, `bench`), GoogleTest wired, `fetch_data.py`, fvecs/ivecs loaders, CI skeleton | Clean build on Linux + macOS; a trivial test runs in CI; datasets download and convert; loaders round-trip |
| M1 | Brute-force index + eval harness (recall@k, QPS, p50/p95/p99 latency, CSV output) | Brute force gets recall = 1.0 vs provided ground truth on all 3 datasets (or subsets); harness output is deterministic |
| M2 | HNSW, single-threaded, scalar kernels, heuristic neighbor selection | 10k random vectors: recall@10 ≥ 0.95 vs brute force at modest ef. SIFT-1M (M=16, efC=200): recall@10 ≥ 0.95 reachable for some ef |
| M3 | SIMD kernels + runtime dispatch + memory layout + prefetch + visited-list pool | Kernel tests pass (all dims 1–1024 sampled, incl. non-multiples of 8); microbench shows AVX2 ≥ 3× scalar on dim=128; end-to-end QPS improves ≥ 2× over M2 at equal recall |
| M4 | Multithreaded build + batch search | Build speedup scales near-linearly to physical cores (report table); TSan clean; results identical in recall quality (±0.5%) to single-thread build |
| M5 | pybind11 package, NumPy API, GIL release, pytest suite | `pip install .` works in a clean venv; pytest passes; Python results match C++ results for same seed/params |
| M6 | Persistence + mmap load | Round-trip test: identical search results after save/load (both modes); corrupt-file fuzz test (truncate/flip bytes) throws, never crashes; ASan clean |
| M7 | Filtered search + soft delete | Correct on selectivity sweep (1%, 10%, 50%, 90% allowed): returned ids always satisfy the filter; report recall vs selectivity; deleted labels never returned |
| M8 | Cross-library benchmarks (see 7) | `scripts/run_all_benchmarks.sh` reproduces CSVs, plots, and the results table end to end |
| M9 | Polish: README, DESIGN.md, BENCHMARKS.md, CI badges, release tag | A new reader can build, test, and reproduce one benchmark from the README in < 15 minutes |

**Performance targets** (aim, not promise; report whatever is measured honestly): within ~20% of hnswlib QPS at equal recall@10 on SIFT-1M single-threaded; stretch goal: match or beat it. Always report where you are slower.

## 7. Benchmark methodology

- **Harnesses**: a C++ harness (`bench/bench_main.cpp`, engine only, fast dev loop for M1–M4) and a Python harness (`bench/python/compare.py`, engine vs `hnswlib` vs `faiss-cpu` IndexHNSWFlat, all through Python for a like-for-like comparison).
- **Fairness**: identical `M`, `ef_construction`, metric, dataset, k=10; identical thread counts per run; comparable compiler flags (build all native code with `-O3 -march=native` for the benchmark; note the exact versions). FAISS: set its OpenMP threads explicitly.
- **Sweep**: `ef_search ∈ {10, 20, 40, 80, 160, 320, 640}`. For each point record recall@10, QPS, p50/p99 per-query latency.
- **Two query modes**: single-thread sequential (latency) and batched multi-thread (throughput). Warm up first; run 3 repetitions and report the median.
- **Build metrics**: build time (1 thread and N threads), peak RSS, serialized index size.
- **Dataset runs**: SIFT-1M (primary, L2), GloVe-100 (cosine), GIST-1M (high-dim; SIMD/memory-layout showcase). Optionally a modern-embedding set (≥768-d) as an extra.
- **Outputs** in `bench/results/`: raw CSV per run, recall-vs-QPS plots (one per dataset, log-y QPS), and `BENCHMARKS.md` with the hardware (CPU model, cores, RAM, OS, compiler + version), commit SHA, and exact commands.
- Search comparisons use a pre-built index (standard ann-benchmarks method); note this in `BENCHMARKS.md`.
- **Ablation table** (SIFT-1M, fixed ef): scalar → +AVX2 → +layout/prefetch → +threads, showing the contribution of each optimization.

## 8. Testing, CI, quality

- **C++ tests (GoogleTest)**: kernel equivalence; brute-force oracle recall on random data (several dims, all metrics); edge cases (empty index, k > size, ef < k clamps, duplicate labels, dim = 1, dims not multiple of 8, identical vectors); determinism for fixed seed single-thread; serialization round-trip + corrupt-file tests; filter correctness; multithreaded stress (concurrent `search_batch`).
- **Python tests (pytest)**: dtype/shape/contiguity errors, id/label handling, parity with C++ results, save/load, filter callback, GIL release smoke test (run searches from several Python threads).
- **Microbenchmarks (Google Benchmark)**: `l2_sq`/`dot` across dims; `search_layer` hot loop.
- **CI (GitHub Actions)**: matrix ubuntu (gcc + clang) and macOS; jobs for build+test, ASan+UBSan, TSan, Python wheel build + pytest, clang-format check, and a tiny smoke benchmark on random data. Full dataset benchmarks are run locally, not in CI.
- **Stretch (only after M9)**: int8/fp16 scalar quantization; AVX-512 kernels; truly concurrent insert+search (prefer an immutable-snapshot swap over fine-grained locking inside the graph); `cibuildwheel` release wheels; a `VectorStore` adapter so the engine can serve as a backend for a code-search/RAG project.

## 9. Deliverables / definition of done

- All milestones M0–M9 accepted; CI green; sanitizers clean.
- README with: one-paragraph pitch, architecture diagram, quickstart (C++ and Python), the results table + recall-vs-QPS plots, and an honest limitations section.
- `docs/DESIGN.md` explaining layout, locking, and dispatch decisions; `docs/FORMAT.md`; `docs/BENCHMARKS.md`.
- Resume-ready summary block at the bottom of `BENCHMARKS.md`, filled **only with measured numbers**: recall@10, QPS vs hnswlib/FAISS, SIMD speedup, thread scaling, memory.
