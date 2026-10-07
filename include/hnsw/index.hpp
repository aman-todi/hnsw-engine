#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hnsw/filter.hpp"
#include "hnsw/metric.hpp"

namespace hnsw {

struct Params {
  std::size_t dim = 0;
  Metric metric = Metric::L2;
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  std::size_t max_elements = 0;  ///< 0 = grow geometrically
  uint64_t seed = 100;
};

struct Neighbor {
  uint64_t label;
  float distance;
};

/// Label used to pad `search_batch` output when fewer than k results exist.
inline constexpr uint64_t kInvalidLabel = ~uint64_t{0};

/// Hierarchical Navigable Small World index (Malkov & Yashunin).
///
/// Thread-safety contract: `search` / `search_batch` / accessors may run
/// concurrently with each other. Mutating calls (`add`, `add_batch`,
/// `mark_deleted`, `set_ef`) are exclusive; they block (via a shared_mutex)
/// until in-flight searches finish rather than corrupting memory.
class Index {
 public:
  explicit Index(const Params& params);
  ~Index();
  Index(Index&&) noexcept;
  Index& operator=(Index&&) noexcept;
  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;

  /// Insert one vector. Throws std::invalid_argument on a duplicate label.
  void add(const float* vec, uint64_t label);
  /// Insert n vectors in parallel (num_threads = 0 -> hardware_concurrency).
  /// All labels are validated before anything is inserted.
  void add_batch(const float* vecs, const uint64_t* labels, std::size_t n, std::size_t num_threads = 0);

  /// k nearest neighbors sorted by increasing distance. ef = 0 uses the
  /// index default (set_ef); ef is clamped to >= k. Returns min(k, #eligible)
  /// results.
  std::vector<Neighbor> search(const float* q, std::size_t k, std::size_t ef = 0,
                               const Filter* f = nullptr) const;
  /// Parallel search over nq queries; writes nq*k results row-major into `out`,
  /// padding missing entries with {kInvalidLabel, +inf}.
  void search_batch(const float* qs, std::size_t nq, std::size_t k, std::size_t ef, Neighbor* out,
                    std::size_t num_threads = 0, const Filter* f = nullptr) const;

  /// Soft delete: the node stays in the graph for traversal but is never
  /// returned. Throws std::invalid_argument for an unknown label.
  void mark_deleted(uint64_t label);
  bool is_deleted(uint64_t label) const;
  bool contains(uint64_t label) const;

  std::size_t size() const;
  std::size_t dim() const noexcept;
  Metric metric() const noexcept;
  const Params& params() const noexcept;
  std::size_t capacity() const;
  std::size_t ef() const;
  void set_ef(std::size_t ef);
  int max_level() const;
  bool read_only() const noexcept;
  /// Approximate bytes used by vectors + graph.
  std::size_t memory_usage() const;

  /// Copy of the stored (possibly normalized) vector for a label.
  std::vector<float> get_vector(uint64_t label) const;

  void save(const std::string& path) const;
  /// Load an index. With mmap = true the file is mapped read-only and the
  /// vector/graph arrays are used in place; such an index rejects `add`.
  static Index load(const std::string& path, bool mmap = false);

  struct Impl;

 private:
  explicit Index(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

namespace tuning {
/// Toggle software prefetching in the search hot loop (ablation benchmarks).
void set_prefetch(bool enabled) noexcept;
bool prefetch_enabled() noexcept;
}  // namespace tuning

/// Exact search over all vectors; used as the correctness oracle and for
/// ground-truth computation.
class BruteForceIndex {
 public:
  BruteForceIndex(std::size_t dim, Metric metric);
  void add(const float* vecs, const uint64_t* labels, std::size_t n);
  std::vector<Neighbor> search(const float* q, std::size_t k, const Filter* f = nullptr) const;
  void search_batch(const float* qs, std::size_t nq, std::size_t k, Neighbor* out,
                    std::size_t num_threads = 0, const Filter* f = nullptr) const;
  std::size_t size() const noexcept { return labels_.size(); }
  std::size_t dim() const noexcept { return dim_; }

 private:
  std::size_t dim_;
  Metric metric_;
  std::vector<float> data_;
  std::vector<uint64_t> labels_;
};

}  // namespace hnsw
