# Design

This document explains how `hnsw-engine` is put together and why. The
algorithm follows Malkov & Yashunin, *Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs* (2016/2018),
Algorithms 1–5. No code was taken from hnswlib or FAISS.

```
            Python (hnsw_engine)                       C++ (namespace hnsw)
 ┌──────────────────────────────────┐   ┌─────────────────────────────────────────────┐
 │ Index.add / search / save / load │──▶│ Index (pimpl)  ── shared_mutex at API edge  │
 │ NumPy in/out, GIL released       │   │   ├─ build:  link() Alg.1 + Alg.4 heuristic  │
 └──────────────────────────────────┘   │   ├─ search: greedy descent + search_layer   │
                                        │   ├─ storage: aligned arena, flat layer 0,   │
                                        │   │           sparse upper layers            │
                                        │   ├─ ScratchPool: visited epochs + heaps     │
                                        │   └─ persist: checksummed format, mmap       │
                                        │ simd::active(): scalar | AVX2 | AVX-512 |    │
                                        │                 NEON  (chosen once)          │
                                        │ parallel_for: atomic-counter work sharing    │
                                        └─────────────────────────────────────────────┘
```

## Algorithm

* **Levels.** `level = floor(-ln(U) · mL)`, `mL = 1/ln(M)`, with `U ∈ (0, 1]`
  drawn from splitmix64 (fully specified, so levels are identical on every
  platform for a given seed; `std::uniform_real_distribution` is not).
  Levels are capped at 64.
* **Degrees.** `M` on layers ≥ 1, `M0 = 2M` on layer 0. A new node selects
  `M` neighbors on every layer (as in the paper); `M0` only caps how many
  reverse edges a layer-0 list can accumulate.
* **Insert (Alg. 1).** Greedy ef=1 descent from the top layer to `level+1`,
  then for each layer `min(level, maxLevel) … 0`: `search_layer` with
  `ef_construction`, heuristic selection, bidirectional linking. A neighbor
  whose list overflows is re-pruned with the same heuristic over its old
  neighbors plus the new node. The result set of one layer is the entry set
  of the next.
* **Heuristic (Alg. 4).** Walk candidates in increasing distance; keep `e`
  only if it is closer to the base element than to every already-kept
  neighbor. `extendCandidates` and `keepPrunedConnections` are off (the paper
  recommends this for most data; it matches common practice). The heuristic
  is applied even when there are fewer than `M` candidates (paper-faithful).
* **search_layer (Alg. 2).** Min-heap of candidates, max-heap of results;
  stop when the nearest candidate is farther than the worst result *and* the
  result set holds `ef` items.
* **Query (Alg. 5).** Greedy descent to layer 1, `search_layer` on layer 0
  with `max(ef, k)`, return the best `k`.
* **Distances.** L2 is squared Euclidean. Inner product and cosine use
  `1 − dot`; for cosine, vectors are normalized into internal storage at
  insert and queries into a scratch buffer, so caller buffers are never
  mutated. A zero vector is stored unnormalized.

## Memory layout

All per-node data is in flat arrays indexed by a dense `uint32` internal id;
external `uint64` labels map to ids through a hash map (ids → labels is a
plain array).

* **Vectors**: one contiguous arena, 64-byte aligned (custom aligned
  allocator under `std::vector`, so RAII and no owning raw pointers). Each row
  is zero-padded to a multiple of 16 floats, so every row starts on a cache
  line and kernels see a length that is a multiple of every SIMD width (the
  kernels still handle any length; padding only removes the scalar tail from
  the hot loop).
* **Layer 0**: a fixed-stride array, per node `[count:u32][M0 × u32]`. One
  multiplication finds a node's list; the list is one or two cache lines for
  M = 16.
* **Upper layers**: only nodes with level ≥ 1 own storage, in one flat `u32`
  array with a per-node offset; each layer is `[count][M × u32]`. About
  1/M of nodes have upper layers, so this is small.
* **Visited set**: a per-thread `uint16` array with an epoch counter.
  Starting a search bumps the epoch instead of clearing; the array is
  zero-filled only when the epoch wraps (every 65,535 searches). Visited lists
  live in a `ScratchPool` (mutex-protected free list) together with the heap
  buffers and the padded query buffer, so steady-state searches do not
  allocate.
* **Prefetch**: while scoring a candidate's neighbors, the next neighbor's
  vector (first two cache lines) and visited mark are prefetched, and the
  next candidate's adjacency list is prefetched before the inner loop starts.
  It can be switched off (`hnsw::tuning::set_prefetch`) for the ablation.

The raw pointers the algorithm uses point either into these owned containers
or into a read-only file mapping (see FORMAT.md), so the same search code
serves both in-memory and mmap indexes.

## SIMD kernels and dispatch

`l2_sq` and `dot` exist as scalar (reference), AVX2+FMA, AVX-512F and NEON.
Each x86 kernel is in its own translation unit compiled with only its own ISA
flags (`-mavx2 -mfma` / `-mavx512f -mfma`), so no AVX instruction can leak
into code that runs before the CPU check. At first use, `simd::active()`
picks the best ISA the CPU supports via `__builtin_cpu_supports` and stores
the function-pointer table in an atomic pointer; `HNSW_SIMD=scalar|avx2|…`
can force a lower ISA. Each `Index` copies the chosen function pointer into
its `Space` at construction, so the hot loop does one indirect call per
distance and no dispatch. On arm64 NEON is baseline and always used.

The kernels use four independent accumulators to cover FMA latency, then a
single-vector loop, then a scalar tail (AVX-512 uses a masked load for the
tail). Tests compare every ISA with the scalar reference for every dim from 1
to 1024, aligned and misaligned, within 1e-5 relative tolerance, and check
exact results on small-integer inputs.

## Concurrency

**Contract.** Searches (`search`, `search_batch`, `size`, …) may run
concurrently with each other. Mutations (`add`, `add_batch`, `mark_deleted`,
`set_ef`) are exclusive. One `std::shared_mutex` at the API boundary enforces
this: searches take it shared, mutations exclusive, so misuse blocks instead
of corrupting memory. `load` constructs a fresh object and needs no lock.
Inside `search_batch` the shared lock is taken once and workers call an
unlocked internal search, which avoids recursive shared locking (that can
deadlock when a writer is queued).

**Parallel build.** `add_batch` first validates all labels, grows storage,
and sequentially stores vectors, labels and levels (levels are drawn in input
order, so they do not depend on the thread count). Only then do workers link
nodes, pulling indices from an atomic counter (`parallel_for`). During
linking:

* each node's adjacency is guarded by one of 4,096 striped mutexes
  (`id & 4095`); readers copy a neighbor list under its lock, then compute
  distances unlocked; a thread never holds two node locks at once, so there
  is no lock-ordering problem;
* a global mutex protects the entry point and max level. A thread reads them
  under the lock and releases it immediately unless its node will become the
  new top level, in which case it holds the lock for its whole insert (rare:
  ~1/M^L of nodes) so two threads cannot both install new tops;
* vectors, levels and upper offsets are written before the workers start
  (thread creation is the happens-before edge) and never change during
  linking, so they are read without locks.

**Search** only reads the graph; the only shared writes are checkouts from
the scratch pool. `search_batch` parallelizes over queries. A storage
reallocation (growth) can only happen under the exclusive lock.

The ThreadSanitizer build runs the whole C++ suite, including parallel
builds, concurrent `search_batch` from four threads, and readers racing with
`add_batch` / `mark_deleted` / `set_ef` writers. The Python suite does the
same from Python threads.

Truly concurrent insert + search (beyond the blocking contract) is out of
scope; the stretch plan is an immutable-snapshot swap.

## Filtered search and soft delete

`Filter` is an interface with a `std::function` wrapper (`FunctionFilter`)
and a bitset allow-list (`BitsetFilter`). On layer 0, filtered-out and
tombstoned nodes are still pushed to the candidate heap and traversed (needed
for connectivity) but never enter the result heap; the search therefore
continues until it has `ef` eligible results or runs out of candidates.
Greedy descent on upper layers ignores filters. Tombstones are a per-node
byte; deleted nodes remain in the graph and keep serving as routing points.
Very selective filters (≈1%) cost more distance computations because many
traversed nodes are ineligible — see BENCHMARKS.md for recall vs selectivity.

## Errors

API-boundary checks throw `std::invalid_argument` (dim = 0, M < 2, k = 0,
duplicate or unknown label, ef = 0) or `std::runtime_error` (index full,
read-only index, I/O, corrupt file). `add_batch` validates every label before
inserting anything, so a failed batch leaves the index unchanged. `search`
with `k > size()` returns `size()` results; `search_batch` pads with
`kInvalidLabel` / `+inf` (−1 / inf in Python).

## Decisions and simplifications

* `max_elements = 0` grows capacity geometrically (×2, min 1,024); a positive
  value preallocates and makes `add` throw when full.
* Internal ids are `uint32`, so an index holds at most 2³²−2 vectors.
* `ef_construction` is clamped to ≥ M.
* The label → id map is rebuilt on load (it is not part of the file).
* Ground-truth recall uses id overlap. Datasets with exact distance ties can
  make a correct result look like a miss; brute force is tie-broken by label.
* `mmap` load validates the whole file, which costs one sequential read; it
  trades load latency for the guarantee that a corrupt file cannot crash.
* Windows is not supported (POSIX `mmap`).
