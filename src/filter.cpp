#include <algorithm>
#include <bit>
#include <stdexcept>
#include <utility>

#include "hnsw/filter.hpp"

namespace hnsw {

FunctionFilter::FunctionFilter(std::function<bool(uint64_t)> fn) : fn_(std::move(fn)) {
  if (!fn_) throw std::invalid_argument("FunctionFilter: empty function");
}

BitsetFilter::BitsetFilter(uint64_t num_labels)
    : words_(static_cast<std::size_t>((num_labels + 63) / 64), 0), size_(num_labels) {}

BitsetFilter BitsetFilter::from_labels(const uint64_t* labels, std::size_t n) {
  uint64_t max_label = 0;
  for (std::size_t i = 0; i < n; ++i) max_label = std::max(max_label, labels[i]);
  BitsetFilter f(n == 0 ? 0 : max_label + 1);
  for (std::size_t i = 0; i < n; ++i) f.set(labels[i]);
  return f;
}

void BitsetFilter::set(uint64_t label, bool value) {
  if (label >= size_) throw std::out_of_range("BitsetFilter::set: label out of range");
  const uint64_t bit = 1ULL << (label & 63);
  if (value) {
    words_[label >> 6] |= bit;
  } else {
    words_[label >> 6] &= ~bit;
  }
}

std::size_t BitsetFilter::count() const noexcept {
  std::size_t c = 0;
  for (uint64_t w : words_) c += static_cast<std::size_t>(std::popcount(w));
  return c;
}

}  // namespace hnsw
