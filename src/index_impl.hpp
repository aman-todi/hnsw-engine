#pragma once

// Internal state of hnsw::Index. See docs/DESIGN.md for the rationale behind
// the memory layout and locking scheme.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "hnsw/index.hpp"
#include "mmap_file.hpp"
#include "space.hpp"

namespace hnsw::detail {

inline constexpr std::size_t kAlignment = 64;      // cache line; also AVX-512 width
inline constexpr std::size_t kPadFloats = 16;      // pad vectors to a multiple of 64 bytes
inline constexpr std::size_t kLockStripes = 4096;  // power of two
inline constexpr uint32_t kNoNode = ~uint32_t{0};
inline constexpr int kMaxLevel = 64;

/// Minimal 64-byte aligned allocator for std::vector (RAII via the container).
template <typename T>
struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() noexcept = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U>&) noexcept {}  // NOLINT(google-explicit-constructor)

  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    const std::size_t bytes = (n * sizeof(T) + kAlignment - 1) / kAlignment * kAlignment;
    return static_cast<T*>(::operator new(bytes, std::align_val_t{kAlignment}));
  }
  void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t{kAlignment}); }
  template <typename U>
  bool operator==(const AlignedAllocator<U>&) const noexcept {
    return true;
  }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

inline std::size_t round_up(std::size_t v, std::size_t m) {
  return (v + m - 1) / m * m;
}

/// Candidate = (distance, internal id).
struct Cand {
  float dist;
  uint32_t id;
};
struct CandLess {  // max-heap on distance
  bool operator()(const Cand& a, const Cand& b) const noexcept {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
  }
};
struct CandGreater {  // min-heap on distance
  bool operator()(const Cand& a, const Cand& b) const noexcept {
    return a.dist > b.dist || (a.dist == b.dist && a.id > b.id);
  }
};

/// Versioned visited set: marking is a store, "clearing" is an epoch bump.
/// The array is only zero-filled when the 16-bit epoch wraps around.
class VisitedList {
 public:
  void prepare(std::size_t n) {
    if (marks_.size() < n) {
      marks_.assign(n, 0);
      epoch_ = 0;
    }
    if (++epoch_ == 0) {
      std::fill(marks_.begin(), marks_.end(), uint16_t{0});
      epoch_ = 1;
    }
  }
  bool visited(uint32_t id) const noexcept { return marks_[id] == epoch_; }
  void mark(uint32_t id) noexcept { marks_[id] = epoch_; }
  /// Mark and return whether it was already visited.
  bool test_and_mark(uint32_t id) noexcept {
    if (marks_[id] == epoch_) return true;
    marks_[id] = epoch_;
    return false;
  }
  const uint16_t* data() const noexcept { return marks_.data(); }

 private:
  std::vector<uint16_t> marks_;
  uint16_t epoch_ = 0;
};

/// Per-thread scratch space checked out from a pool for one search/insert.
struct Scratch {
  VisitedList visited;
  std::vector<Cand> candidates;  // min-heap
  std::vector<Cand> results;     // max-heap
  std::vector<Cand> tmp;
  std::vector<uint32_t> neighbor_buf;
  AlignedVector<float> query;  // padded, zero tail
};

class ScratchPool {
 public:
  class Handle {
   public:
    Handle(ScratchPool* pool, std::unique_ptr<Scratch> s) : pool_(pool), s_(std::move(s)) {}
    Handle(Handle&&) noexcept = default;
    Handle& operator=(Handle&&) = delete;
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() {
      if (s_) pool_->release(std::move(s_));
    }
    Scratch* operator->() const noexcept { return s_.get(); }
    Scratch& operator*() const noexcept { return *s_; }

   private:
    ScratchPool* pool_;
    std::unique_ptr<Scratch> s_;
  };

  Handle acquire(std::size_t num_nodes, std::size_t padded_dim) {
    std::unique_ptr<Scratch> s;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!free_.empty()) {
        s = std::move(free_.back());
        free_.pop_back();
      }
    }
    if (!s) s = std::make_unique<Scratch>();
    s->visited.prepare(num_nodes);
    if (s->query.size() != padded_dim) s->query.assign(padded_dim, 0.0f);
    return Handle(this, std::move(s));
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_.clear();
  }

 private:
  void release(std::unique_ptr<Scratch> s) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_.push_back(std::move(s));
  }
  std::mutex mutex_;
  std::vector<std::unique_ptr<Scratch>> free_;
};

/// Global prefetch switch (ablation benchmarks only).
extern std::atomic<bool> g_prefetch_enabled;

inline void prefetch(const void* p) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p, 0, 3);
#else
  (void)p;
#endif
}

}  // namespace hnsw::detail

namespace hnsw {

struct Index::Impl {
  // ---- parameters -------------------------------------------------------
  Params params;
  std::size_t dim = 0;
  std::size_t padded_dim = 0;
  std::size_t M = 0;   // max degree on layers >= 1
  std::size_t M0 = 0;  // max degree on layer 0
  std::size_t ef_construction = 0;
  std::size_t ef_search = 10;
  double mL = 0.0;
  std::size_t l0_stride = 0;     // uint32 per node in layer 0: 1 + M0
  std::size_t upper_stride = 0;  // uint32 per node per upper layer: 1 + M
  detail::Space space;

  // ---- storage ----------------------------------------------------------
  // The algorithm only uses the raw pointers below. They point either into
  // the owned containers or into a read-only file mapping (mmap load).
  detail::AlignedVector<float> vec_store;
  detail::AlignedVector<uint32_t> l0_store;
  std::vector<uint32_t> upper_store;
  std::vector<uint64_t> label_store;
  std::vector<uint8_t> level_store;

  const float* vecs = nullptr;
  uint32_t* l0 = nullptr;
  uint32_t* upper = nullptr;
  const uint64_t* labels = nullptr;
  const uint8_t* levels = nullptr;

  std::vector<uint64_t> upper_offset;  // per node, in uint32 units (owned)
  std::vector<uint8_t> deleted;        // per node tombstones (owned)
  std::size_t num_deleted = 0;
  std::unordered_map<uint64_t, uint32_t> label_to_id;

  std::size_t count = 0;
  std::size_t capacity = 0;
  std::size_t upper_used = 0;  // uint32 slots used in upper storage
  uint32_t entry_point = detail::kNoNode;
  int max_level = -1;

  detail::MappedFile mapping;
  bool read_only = false;

  // ---- synchronization --------------------------------------------------
  mutable std::shared_mutex api_mutex;  // shared: search; exclusive: mutation
  std::mutex global_mutex;              // entry point / max level during build
  std::unique_ptr<std::mutex[]> stripes{new std::mutex[detail::kLockStripes]};
  mutable detail::ScratchPool scratch;

  uint64_t rng_state = 0;

  explicit Impl(const Params& p);

  // ---- accessors --------------------------------------------------------
  const float* vec(uint32_t id) const noexcept { return vecs + std::size_t{id} * padded_dim; }
  uint32_t* list0(uint32_t id) const noexcept { return l0 + std::size_t{id} * l0_stride; }
  uint32_t* list(uint32_t id, int level) const noexcept {
    return level == 0 ? list0(id) : upper + upper_offset[id] + std::size_t(level - 1) * upper_stride;
  }
  std::size_t max_degree(int level) const noexcept { return level == 0 ? M0 : M; }
  std::mutex& node_lock(uint32_t id) const noexcept { return stripes[id & (detail::kLockStripes - 1)]; }
  float dist(const float* a, const float* b) const noexcept { return space(a, b, padded_dim); }

  // ---- build (index.cpp) ------------------------------------------------
  void reserve(std::size_t n);
  int draw_level();
  uint32_t append_node(const float* v, uint64_t label);
  void link(uint32_t id);
  void select_neighbors(std::vector<detail::Cand>& cands, std::size_t m,
                        std::vector<detail::Cand>& out) const;
  void connect(uint32_t id, int level, const std::vector<detail::Cand>& selected);
  void refresh_pointers();

  // ---- search (search_layer.cpp) ----------------------------------------
  /// Greedy ef=1 descent on one layer, starting from `cur`.
  template <bool Locked>
  detail::Cand greedy_closest(const float* q, detail::Cand cur, int level, std::vector<uint32_t>& buf) const;
  /// Algorithm 2. Results end up as a max-heap in s.results.
  template <bool Locked, bool Filtered>
  void search_layer(const float* q, const detail::Cand* eps, std::size_t n_eps, std::size_t ef, int level,
                    detail::Scratch& s, const Filter* f) const;
  /// Full query (caller holds api_mutex in shared mode).
  std::vector<Neighbor> search_unlocked(const float* q, std::size_t k, std::size_t ef, const Filter* f) const;
  bool eligible(uint32_t id, const Filter* f) const {
    return deleted[id] == 0 && (f == nullptr || f->allowed(labels[id]));
  }
};

}  // namespace hnsw
