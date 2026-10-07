// HNSW construction (Malkov & Yashunin, Algorithms 1 and 4) and the public
// Index API. Search lives in search_layer.cpp, persistence in persist.cpp.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "hnsw/index.hpp"
#include "index_impl.hpp"
#include "threadpool.hpp"

namespace hnsw {

namespace detail {
std::atomic<bool> g_prefetch_enabled{true};

namespace {
/// splitmix64: tiny, portable and fully specified (unlike std distributions),
/// so level assignment is identical on every platform for a given seed.
inline uint64_t splitmix64(uint64_t& state) noexcept {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
}  // namespace
}  // namespace detail

namespace tuning {
void set_prefetch(bool enabled) noexcept {
  detail::g_prefetch_enabled.store(enabled);
}
bool prefetch_enabled() noexcept {
  return detail::g_prefetch_enabled.load();
}
}  // namespace tuning

using detail::Cand;
using detail::CandLess;

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

Index::Impl::Impl(const Params& p) : params(p) {
  if (p.dim == 0) throw std::invalid_argument("Index: dim must be > 0");
  if (p.M < 2) throw std::invalid_argument("Index: M must be >= 2");
  if (p.M > 2048) throw std::invalid_argument("Index: M must be <= 2048");
  if (p.ef_construction == 0) throw std::invalid_argument("Index: ef_construction must be > 0");
  if (p.max_elements > std::numeric_limits<uint32_t>::max() - 1) {
    throw std::invalid_argument("Index: max_elements exceeds the 32-bit internal id space");
  }
  dim = p.dim;
  padded_dim = detail::round_up(dim, detail::kPadFloats);
  M = p.M;
  M0 = 2 * p.M;
  ef_construction = std::max(p.ef_construction, p.M);
  mL = 1.0 / std::log(static_cast<double>(M));
  l0_stride = 1 + M0;
  upper_stride = 1 + M;
  space = detail::Space(p.metric, simd::active());
  rng_state = p.seed;
  if (p.max_elements > 0) reserve(p.max_elements);
}

void Index::Impl::refresh_pointers() {
  vecs = vec_store.data();
  l0 = l0_store.data();
  upper = upper_store.data();
  labels = label_store.data();
  levels = level_store.data();
}

void Index::Impl::reserve(std::size_t n) {
  if (n <= capacity) return;
  if (read_only) throw std::runtime_error("index is read-only (memory-mapped)");
  if (n > std::numeric_limits<uint32_t>::max() - 1) {
    throw std::runtime_error("Index: capacity exceeds the 32-bit internal id space");
  }
  vec_store.resize(n * padded_dim, 0.0f);
  l0_store.resize(n * l0_stride, 0u);
  label_store.resize(n, 0);
  level_store.resize(n, 0);
  upper_offset.resize(n, 0);
  deleted.resize(n, 0);
  capacity = n;
  refresh_pointers();
  // Pooled visited lists are resized lazily on their next checkout.
}

int Index::Impl::draw_level() {
  // U in (0, 1]: 53 random mantissa bits.
  const double u = 1.0 - static_cast<double>(detail::splitmix64(rng_state) >> 11) * 0x1.0p-53;
  const double lvl = std::floor(-std::log(u) * mL);
  return static_cast<int>(std::min(lvl, static_cast<double>(detail::kMaxLevel)));
}

/// Store vector/label/level for a new node (sequential part of insertion).
uint32_t Index::Impl::append_node(const float* v, uint64_t label) {
  const auto id = static_cast<uint32_t>(count);
  float* dst = vec_store.data() + std::size_t{id} * padded_dim;
  detail::prepare_vector(params.metric, v, dst, dim);
  label_store[id] = label;
  const int level = draw_level();
  level_store[id] = static_cast<uint8_t>(level);
  if (level > 0) {
    upper_offset[id] = upper_used;
    upper_used += static_cast<std::size_t>(level) * upper_stride;
    if (upper_store.size() < upper_used) {
      upper_store.resize(std::max(upper_used, upper_store.size() * 2), 0u);
      upper = upper_store.data();
    }
  }
  list0(id)[0] = 0;
  label_to_id.emplace(label, id);
  ++count;
  return id;
}

/// Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC), without extendCandidates and
/// keepPrunedConnections: walk candidates in increasing distance and keep e
/// only if it is closer to the base element than to every kept neighbor.
/// `cands` must be sorted by increasing distance.
void Index::Impl::select_neighbors(std::vector<Cand>& cands, std::size_t m, std::vector<Cand>& out) const {
  out.clear();
  for (const Cand& e : cands) {
    if (out.size() >= m) break;
    const float* ev = vec(e.id);
    bool good = true;
    for (const Cand& r : out) {
      if (dist(ev, vec(r.id)) < e.dist) {
        good = false;
        break;
      }
    }
    if (good) out.push_back(e);
  }
}

/// Set id's adjacency at `level` and add the reverse edges, re-pruning any
/// neighbor list that overflows (Algorithm 1, lines 11-16).
void Index::Impl::connect(uint32_t id, int level, const std::vector<Cand>& selected) {
  const std::size_t mmax = max_degree(level);
  std::vector<Cand> cands;
  std::vector<Cand> pruned;
  {
    std::lock_guard<std::mutex> lock(node_lock(id));
    uint32_t* l = list(id, level);
    const uint32_t existing = l[0];
    if (existing == 0) {
      l[0] = static_cast<uint32_t>(selected.size());
      for (std::size_t i = 0; i < selected.size(); ++i) l[1 + i] = selected[i].id;
    } else {
      // Parallel build only: the entry set of this layer is the previous
      // layer's result set, so a concurrent inserter can reach this node via
      // an upper layer and link to it here before we do. Keep those edges
      // (overwriting them silently disconnects nodes) and re-prune on overflow.
      const float* q = vec(id);
      cands.assign(selected.begin(), selected.end());
      for (uint32_t j = 0; j < existing; ++j) {
        const uint32_t e = l[1 + j];
        bool dup = false;
        for (const Cand& c : cands) dup = dup || c.id == e;
        if (!dup) cands.push_back(Cand{dist(q, vec(e)), e});
      }
      std::sort(cands.begin(), cands.end(), CandLess{});
      if (cands.size() > mmax) {
        select_neighbors(cands, mmax, pruned);
        cands.swap(pruned);
      }
      l[0] = static_cast<uint32_t>(cands.size());
      for (std::size_t i = 0; i < cands.size(); ++i) l[1 + i] = cands[i].id;
    }
  }

  for (const Cand& nb : selected) {
    std::lock_guard<std::mutex> lock(node_lock(nb.id));
    uint32_t* l = list(nb.id, level);
    const uint32_t n = l[0];
    bool present = false;
    for (uint32_t j = 0; j < n; ++j) present = present || l[1 + j] == id;
    if (present) continue;
    if (n < mmax) {
      l[1 + n] = id;
      l[0] = n + 1;
      continue;
    }
    // Overflow: shrink eConn with the heuristic over old neighbors + id.
    const float* nv = vec(nb.id);
    cands.clear();
    cands.push_back(Cand{nb.dist, id});
    for (uint32_t j = 0; j < n; ++j) cands.push_back(Cand{dist(nv, vec(l[1 + j])), l[1 + j]});
    std::sort(cands.begin(), cands.end(), CandLess{});
    select_neighbors(cands, mmax, pruned);
    l[0] = static_cast<uint32_t>(pruned.size());
    for (std::size_t j = 0; j < pruned.size(); ++j) l[1 + j] = pruned[j].id;
  }
}

/// Algorithm 1 (INSERT) for a node whose vector/level were already stored.
/// Safe to run concurrently for different nodes.
void Index::Impl::link(uint32_t id) {
  const int level = levels[id];
  const float* q = vec(id);

  std::unique_lock<std::mutex> top_lock(global_mutex);
  const uint32_t ep0 = entry_point;
  const int top = max_level;
  if (ep0 == detail::kNoNode) {
    entry_point = id;
    max_level = level;
    return;
  }
  if (level <= top) top_lock.unlock();  // keep holding only if we become the new top

  auto s = scratch.acquire(capacity, padded_dim);
  Cand cur{dist(q, vec(ep0)), ep0};
  for (int lc = top; lc > level; --lc) cur = greedy_closest<true>(q, cur, lc, s->neighbor_buf);

  std::vector<Cand> eps{cur};
  std::vector<Cand> w;
  std::vector<Cand> selected;
  for (int lc = std::min(level, top); lc >= 0; --lc) {
    search_layer<true, false>(q, eps.data(), eps.size(), ef_construction, lc, *s, nullptr);
    w.assign(s->results.begin(), s->results.end());
    std::sort(w.begin(), w.end(), CandLess{});
    // Never link to self (cannot normally happen; defensive).
    w.erase(std::remove_if(w.begin(), w.end(), [id](const Cand& c) { return c.id == id; }), w.end());
    select_neighbors(w, M, selected);
    connect(id, lc, selected);
    eps.swap(w);
  }

  if (level > top) {
    entry_point = id;
    max_level = level;
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

Index::Index(const Params& params) : impl_(std::make_unique<Impl>(params)) {}
Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Index::~Index() = default;
Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;

namespace {
void check_writable(const Index::Impl& im) {
  if (im.read_only)
    throw std::runtime_error("index is read-only (memory-mapped); load without mmap to modify");
}

void grow_for(Index::Impl& im, std::size_t extra) {
  const std::size_t need = im.count + extra;
  if (need <= im.capacity) return;
  if (im.params.max_elements > 0) {
    throw std::runtime_error("Index is full (max_elements = " + std::to_string(im.params.max_elements) + ")");
  }
  im.reserve(std::max({need, im.capacity * 2, std::size_t{1024}}));
}
}  // namespace

void Index::add(const float* vec, uint64_t label) {
  add_batch(vec, &label, 1, 1);
}

void Index::add_batch(const float* vecs, const uint64_t* labels, std::size_t n, std::size_t num_threads) {
  if (n == 0) return;
  if (vecs == nullptr || labels == nullptr) throw std::invalid_argument("add_batch: null input");
  std::unique_lock<std::shared_mutex> api(impl_->api_mutex);
  Impl& im = *impl_;
  check_writable(im);

  // Validate everything before mutating anything.
  if (n > 1) {
    std::unordered_set<uint64_t> seen;
    seen.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      if (!seen.insert(labels[i]).second) {
        throw std::invalid_argument("add_batch: duplicate label " + std::to_string(labels[i]) + " in batch");
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (im.label_to_id.count(labels[i]) != 0) {
      throw std::invalid_argument("add: label " + std::to_string(labels[i]) + " already exists");
    }
  }
  grow_for(im, n);
  im.label_to_id.reserve(im.count + n);

  // Sequential: store vectors, labels and levels (levels drawn in input
  // order, so they do not depend on the thread count).
  const auto first = static_cast<uint32_t>(im.count);
  for (std::size_t i = 0; i < n; ++i) im.append_node(vecs + i * im.dim, labels[i]);

  // Parallel: link nodes into the graph.
  detail::parallel_for(n, num_threads,
                       [&](std::size_t i, std::size_t) { im.link(first + static_cast<uint32_t>(i)); });
}

void Index::mark_deleted(uint64_t label) {
  std::unique_lock<std::shared_mutex> api(impl_->api_mutex);
  auto it = impl_->label_to_id.find(label);
  if (it == impl_->label_to_id.end()) {
    throw std::invalid_argument("mark_deleted: unknown label " + std::to_string(label));
  }
  if (impl_->deleted[it->second] == 0) {
    impl_->deleted[it->second] = 1;
    ++impl_->num_deleted;
  }
}

bool Index::is_deleted(uint64_t label) const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  auto it = impl_->label_to_id.find(label);
  return it != impl_->label_to_id.end() && impl_->deleted[it->second] != 0;
}

bool Index::contains(uint64_t label) const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  return impl_->label_to_id.count(label) != 0;
}

std::size_t Index::size() const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  return impl_->count;
}
std::size_t Index::dim() const noexcept {
  return impl_->dim;
}
Metric Index::metric() const noexcept {
  return impl_->params.metric;
}
const Params& Index::params() const noexcept {
  return impl_->params;
}
std::size_t Index::capacity() const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  return impl_->capacity;
}
std::size_t Index::ef() const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  return impl_->ef_search;
}
void Index::set_ef(std::size_t ef) {
  if (ef == 0) throw std::invalid_argument("set_ef: ef must be > 0");
  std::unique_lock<std::shared_mutex> api(impl_->api_mutex);
  impl_->ef_search = ef;
}
int Index::max_level() const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  return impl_->max_level;
}
bool Index::read_only() const noexcept {
  return impl_->read_only;
}

std::size_t Index::memory_usage() const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  const Impl& im = *impl_;
  return im.count *
             (im.padded_dim * sizeof(float) + im.l0_stride * sizeof(uint32_t) + sizeof(uint64_t) * 2 + 2) +
         im.upper_used * sizeof(uint32_t);
}

std::vector<float> Index::get_vector(uint64_t label) const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  auto it = impl_->label_to_id.find(label);
  if (it == impl_->label_to_id.end()) throw std::invalid_argument("get_vector: unknown label");
  const float* v = impl_->vec(it->second);
  return std::vector<float>(v, v + impl_->dim);
}

}  // namespace hnsw
