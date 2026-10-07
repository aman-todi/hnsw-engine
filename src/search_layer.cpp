// Layer search (Malkov & Yashunin, Algorithm 2), greedy descent, and the
// query entry points (Algorithm 5).

#include <algorithm>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

#include "hnsw/index.hpp"
#include "index_impl.hpp"
#include "threadpool.hpp"

namespace hnsw {

using detail::Cand;
using detail::CandGreater;
using detail::CandLess;

namespace {
/// Prefetch the first two cache lines of a vector; the hardware stream
/// prefetcher picks up the rest of a long vector.
inline void prefetch_vector(const float* v) noexcept {
  detail::prefetch(v);
  detail::prefetch(v + 16);
}
}  // namespace

template <bool Locked>
Cand Index::Impl::greedy_closest(const float* q, Cand cur, int level, std::vector<uint32_t>& buf) const {
  bool changed = true;
  while (changed) {
    changed = false;
    const uint32_t* nbrs = nullptr;
    uint32_t n = 0;
    if constexpr (Locked) {
      std::lock_guard<std::mutex> lock(node_lock(cur.id));
      const uint32_t* l = list(cur.id, level);
      n = l[0];
      buf.assign(l + 1, l + 1 + n);
      nbrs = buf.data();
    } else {
      const uint32_t* l = list(cur.id, level);
      n = l[0];
      nbrs = l + 1;
    }
    for (uint32_t j = 0; j < n; ++j) {
      const float d = dist(q, vec(nbrs[j]));
      if (d < cur.dist) {
        cur = Cand{d, nbrs[j]};
        changed = true;
      }
    }
  }
  return cur;
}

template <bool Locked, bool Filtered>
void Index::Impl::search_layer(const float* q, const Cand* eps, std::size_t n_eps, std::size_t ef, int level,
                               detail::Scratch& s, const Filter* f) const {
  detail::VisitedList& visited = s.visited;
  visited.prepare(capacity);
  std::vector<Cand>& cand = s.candidates;  // min-heap: closest first
  std::vector<Cand>& res = s.results;      // max-heap: furthest first
  cand.clear();
  res.clear();

  for (std::size_t i = 0; i < n_eps; ++i) {
    const Cand& e = eps[i];
    if (visited.test_and_mark(e.id)) continue;
    cand.push_back(e);
    if (!Filtered || eligible(e.id, f)) res.push_back(e);
  }
  std::make_heap(cand.begin(), cand.end(), CandGreater{});
  std::make_heap(res.begin(), res.end(), CandLess{});
  while (res.size() > ef) {
    std::pop_heap(res.begin(), res.end(), CandLess{});
    res.pop_back();
  }

  const bool pf = detail::g_prefetch_enabled.load(std::memory_order_relaxed);
  while (!cand.empty()) {
    const Cand c = cand.front();
    if (res.size() >= ef && c.dist > res.front().dist) break;
    std::pop_heap(cand.begin(), cand.end(), CandGreater{});
    cand.pop_back();

    const uint32_t* nbrs = nullptr;
    uint32_t n = 0;
    if constexpr (Locked) {
      std::lock_guard<std::mutex> lock(node_lock(c.id));
      const uint32_t* l = list(c.id, level);
      n = l[0];
      s.neighbor_buf.assign(l + 1, l + 1 + n);
      nbrs = s.neighbor_buf.data();
    } else {
      const uint32_t* l = list(c.id, level);
      n = l[0];
      nbrs = l + 1;
    }

    if (pf) {
      // While this candidate's neighbors are scored, pull in the adjacency of
      // the next candidate and the first neighbor's vector.
      if (!cand.empty()) detail::prefetch(list(cand.front().id, level));
      if (n > 0) {
        prefetch_vector(vec(nbrs[0]));
        detail::prefetch(visited.data() + nbrs[0]);
      }
    }

    for (uint32_t j = 0; j < n; ++j) {
      const uint32_t id = nbrs[j];
      if (pf && j + 1 < n) {
        prefetch_vector(vec(nbrs[j + 1]));
        detail::prefetch(visited.data() + nbrs[j + 1]);
      }
      if (visited.test_and_mark(id)) continue;
      const float d = dist(q, vec(id));
      if (res.size() < ef || d < res.front().dist) {
        cand.push_back(Cand{d, id});
        std::push_heap(cand.begin(), cand.end(), CandGreater{});
        if (!Filtered || eligible(id, f)) {
          res.push_back(Cand{d, id});
          std::push_heap(res.begin(), res.end(), CandLess{});
          if (res.size() > ef) {
            std::pop_heap(res.begin(), res.end(), CandLess{});
            res.pop_back();
          }
        }
      }
    }
  }
}

// Explicit instantiations used by index.cpp (build) and below (query).
template Cand Index::Impl::greedy_closest<true>(const float*, Cand, int, std::vector<uint32_t>&) const;
template Cand Index::Impl::greedy_closest<false>(const float*, Cand, int, std::vector<uint32_t>&) const;
template void Index::Impl::search_layer<true, false>(const float*, const Cand*, std::size_t, std::size_t, int,
                                                     detail::Scratch&, const Filter*) const;
template void Index::Impl::search_layer<false, false>(const float*, const Cand*, std::size_t, std::size_t,
                                                      int, detail::Scratch&, const Filter*) const;
template void Index::Impl::search_layer<false, true>(const float*, const Cand*, std::size_t, std::size_t, int,
                                                     detail::Scratch&, const Filter*) const;

std::vector<Neighbor> Index::Impl::search_unlocked(const float* q, std::size_t k, std::size_t ef,
                                                   const Filter* f) const {
  std::vector<Neighbor> out;
  if (count == 0) return out;
  ef = std::max(ef == 0 ? ef_search : ef, k);

  auto s = scratch.acquire(capacity, padded_dim);
  detail::prepare_vector(params.metric, q, s->query.data(), dim);
  const float* qq = s->query.data();

  Cand cur{dist(qq, vec(entry_point)), entry_point};
  for (int lc = max_level; lc > 0; --lc) cur = greedy_closest<false>(qq, cur, lc, s->neighbor_buf);

  if (f != nullptr || num_deleted > 0) {
    search_layer<false, true>(qq, &cur, 1, ef, 0, *s, f);
  } else {
    search_layer<false, false>(qq, &cur, 1, ef, 0, *s, nullptr);
  }

  std::vector<Cand>& res = s->results;
  std::sort(res.begin(), res.end(), CandLess{});
  const std::size_t n = std::min(k, res.size());
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) out.push_back(Neighbor{labels[res[i].id], res[i].dist});
  return out;
}

std::vector<Neighbor> Index::search(const float* q, std::size_t k, std::size_t ef, const Filter* f) const {
  if (k == 0) throw std::invalid_argument("search: k must be > 0");
  if (q == nullptr) throw std::invalid_argument("search: null query");
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  return impl_->search_unlocked(q, k, ef, f);
}

void Index::search_batch(const float* qs, std::size_t nq, std::size_t k, std::size_t ef, Neighbor* out,
                         std::size_t num_threads, const Filter* f) const {
  if (k == 0) throw std::invalid_argument("search_batch: k must be > 0");
  if (nq == 0) return;
  if (qs == nullptr || out == nullptr) throw std::invalid_argument("search_batch: null pointer");
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  const Impl& im = *impl_;
  detail::parallel_for(nq, num_threads, [&](std::size_t i, std::size_t) {
    const auto res = im.search_unlocked(qs + i * im.dim, k, ef, f);
    Neighbor* row = out + i * k;
    for (std::size_t j = 0; j < k; ++j) {
      row[j] = j < res.size() ? res[j] : Neighbor{kInvalidLabel, std::numeric_limits<float>::infinity()};
    }
  });
}

}  // namespace hnsw
