// Exact k-NN by scanning every vector. Used as the correctness oracle.

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "hnsw/index.hpp"
#include "space.hpp"
#include "threadpool.hpp"

namespace hnsw {

namespace {
inline bool neighbor_less(const Neighbor& a, const Neighbor& b) {
  return a.distance < b.distance || (a.distance == b.distance && a.label < b.label);
}
}  // namespace

BruteForceIndex::BruteForceIndex(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {
  if (dim == 0) throw std::invalid_argument("BruteForceIndex: dim must be > 0");
}

void BruteForceIndex::add(const float* vecs, const uint64_t* labels, std::size_t n) {
  const std::size_t old = labels_.size();
  data_.resize((old + n) * dim_);
  labels_.resize(old + n);
  for (std::size_t i = 0; i < n; ++i) {
    detail::prepare_vector(metric_, vecs + i * dim_, data_.data() + (old + i) * dim_, dim_);
    labels_[old + i] = labels != nullptr ? labels[i] : static_cast<uint64_t>(old + i);
  }
}

std::vector<Neighbor> BruteForceIndex::search(const float* q, std::size_t k, const Filter* f) const {
  if (k == 0) throw std::invalid_argument("search: k must be > 0");
  const detail::Space space(metric_, simd::active());
  std::vector<float> query(dim_);
  detail::prepare_vector(metric_, q, query.data(), dim_);

  // Max-heap (by neighbor_less) holding the best k seen so far.
  std::vector<Neighbor> heap;
  heap.reserve(k + 1);
  for (std::size_t i = 0; i < labels_.size(); ++i) {
    if (f != nullptr && !f->allowed(labels_[i])) continue;
    const Neighbor cand{labels_[i], space(query.data(), data_.data() + i * dim_, dim_)};
    if (heap.size() < k) {
      heap.push_back(cand);
      std::push_heap(heap.begin(), heap.end(), neighbor_less);
    } else if (neighbor_less(cand, heap.front())) {
      std::pop_heap(heap.begin(), heap.end(), neighbor_less);
      heap.back() = cand;
      std::push_heap(heap.begin(), heap.end(), neighbor_less);
    }
  }
  std::sort_heap(heap.begin(), heap.end(), neighbor_less);
  return heap;
}

void BruteForceIndex::search_batch(const float* qs, std::size_t nq, std::size_t k, Neighbor* out,
                                   std::size_t num_threads, const Filter* f) const {
  if (k == 0) throw std::invalid_argument("search: k must be > 0");
  detail::parallel_for(nq, num_threads, [&](std::size_t i, std::size_t) {
    const auto res = search(qs + i * dim_, k, f);
    Neighbor* row = out + i * k;
    for (std::size_t j = 0; j < k; ++j) {
      row[j] = j < res.size() ? res[j] : Neighbor{kInvalidLabel, std::numeric_limits<float>::infinity()};
    }
  });
}

}  // namespace hnsw
