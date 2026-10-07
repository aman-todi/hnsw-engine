#include "threadpool.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace hnsw::detail {

std::size_t resolve_threads(std::size_t requested) noexcept {
  if (requested != 0) return requested;
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : hw;
}

void parallel_for(std::size_t n, std::size_t num_threads,
                  const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n == 0) return;
  const std::size_t workers = std::min(resolve_threads(num_threads), n);
  if (workers == 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i, 0);
    return;
  }

  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr error;
  std::mutex error_mutex;

  auto run = [&](std::size_t worker_id) {
    try {
      while (!failed.load(std::memory_order_relaxed)) {
        const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) break;
        fn(i, worker_id);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error) error = std::current_exception();
      failed.store(true, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  for (std::size_t t = 1; t < workers; ++t) threads.emplace_back(run, t);
  run(0);
  for (auto& th : threads) th.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace hnsw::detail
