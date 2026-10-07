#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace hnsw {

/// Predicate over external labels used by filtered search. Filtered-out nodes
/// are still traversed (to keep the graph connected) but never returned.
/// Implementations must be safe to call concurrently from several threads.
class Filter {
 public:
  virtual ~Filter() = default;
  virtual bool allowed(uint64_t label) const = 0;
  bool operator()(uint64_t label) const { return allowed(label); }
};

/// Wraps an arbitrary callable.
class FunctionFilter final : public Filter {
 public:
  explicit FunctionFilter(std::function<bool(uint64_t)> fn);
  bool allowed(uint64_t label) const override { return fn_(label); }

 private:
  std::function<bool(uint64_t)> fn_;
};

/// Allow-list stored as a bitset indexed by label. Labels >= size() are
/// rejected.
class BitsetFilter final : public Filter {
 public:
  BitsetFilter() = default;
  explicit BitsetFilter(uint64_t num_labels);
  /// Build from an allow-list of labels; the bitset is sized to max label + 1.
  static BitsetFilter from_labels(const uint64_t* labels, std::size_t n);

  void set(uint64_t label, bool value = true);
  bool allowed(uint64_t label) const override {
    return label < size_ && ((words_[label >> 6] >> (label & 63)) & 1ULL) != 0;
  }
  uint64_t size() const noexcept { return size_; }
  std::size_t count() const noexcept;

 private:
  std::vector<uint64_t> words_;
  uint64_t size_ = 0;
};

}  // namespace hnsw
