#pragma once

#include <cstddef>
#include <functional>

namespace hnsw::detail {

/// 0 -> std::thread::hardware_concurrency() (at least 1).
std::size_t resolve_threads(std::size_t requested) noexcept;

/// Run fn(i, worker_id) for every i in [0, n) on `num_threads` threads
/// (the caller participates as worker 0). Work is handed out through a shared
/// atomic counter so uneven per-item cost balances automatically. The first
/// exception thrown by any worker stops the distribution of new items and is
/// rethrown in the caller after all workers have joined.
void parallel_for(std::size_t n, std::size_t num_threads,
                  const std::function<void(std::size_t, std::size_t)>& fn);

}  // namespace hnsw::detail
