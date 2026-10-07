#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "hnsw/filter.hpp"
#include "hnsw/index.hpp"
#include "test_util.hpp"

using namespace hnsw;

namespace {

constexpr std::size_t kDim = 32;

TEST(Concurrency, ParallelBuildMatchesSingleThreadRecall) {
  const std::size_t n = 8000, nq = 200, k = 10;
  const auto data = test::clustered_vectors(n, kDim, 101);
  const auto queries = test::clustered_vectors(nq, kDim, 102);
  const auto labels = test::iota_labels(n);
  BruteForceIndex bf(kDim, Metric::L2);
  bf.add(data.data(), labels.data(), n);
  std::vector<Neighbor> truth(nq * k);
  bf.search_batch(queries.data(), nq, k, truth.data());

  auto recall_with = [&](std::size_t threads) {
    Index index(Params{kDim});
    index.add_batch(data.data(), labels.data(), n, threads);
    EXPECT_EQ(index.size(), n);
    std::vector<Neighbor> got(nq * k);
    index.search_batch(queries.data(), nq, k, 128, got.data(), 4);
    return test::recall(got, truth, nq, k);
  };
  const double r1 = recall_with(1);
  const double r4 = recall_with(4);
  EXPECT_GE(r1, 0.95);
  EXPECT_NEAR(r4, r1, 0.01);
}

TEST(Concurrency, ParallelBuildKeepsEveryNodeReachable) {
  // Regression: a concurrent inserter can link to a node on a layer before
  // that node writes its own list there; the node must merge, not overwrite.
  // Collinear points make the heuristic keep ~2 edges per node, so a single
  // lost edge disconnects part of the graph.
  const std::size_t n = 100, dim = 128;
  std::vector<float> data(n * dim, 0.5f);
  for (std::size_t i = 0; i < n; ++i) data[i * dim] = static_cast<float>(i);
  const auto labels = test::iota_labels(n);
  for (uint64_t seed = 0; seed < 20; ++seed) {
    Params p{dim};
    p.seed = seed;
    Index index(p);
    index.add_batch(data.data(), labels.data(), n, 4);
    EXPECT_EQ(index.search(data.data(), n, 2 * n).size(), n) << "seed=" << seed;
    BitsetFilter allow(n);
    allow.set(7);
    allow.set(99);
    EXPECT_EQ(index.search(data.data(), 10, 64, &allow).size(), 2u) << "seed=" << seed;
  }
}

TEST(Concurrency, BatchSearchEqualsSequentialSearch) {
  const std::size_t n = 3000, nq = 300, k = 5;
  const auto data = test::clustered_vectors(n, kDim, 7);
  Index index(Params{kDim});
  index.add_batch(data.data(), test::iota_labels(n).data(), n, 4);
  std::vector<Neighbor> batch(nq * k);
  index.search_batch(data.data(), nq, k, 50, batch.data(), 4);
  for (std::size_t i = 0; i < nq; ++i) {
    const auto r = index.search(data.data() + i * kDim, k, 50);
    for (std::size_t j = 0; j < k; ++j) EXPECT_EQ(batch[i * k + j].label, r[j].label);
  }
}

TEST(Concurrency, ConcurrentSearchBatchFromManyThreads) {
  const std::size_t n = 4000, nq = 100, k = 10;
  const auto data = test::clustered_vectors(n, kDim, 8);
  Index index(Params{kDim});
  index.add_batch(data.data(), test::iota_labels(n).data(), n);
  std::vector<Neighbor> expected(nq * k);
  index.search_batch(data.data(), nq, k, 40, expected.data(), 1);

  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&] {
      for (int rep = 0; rep < 5; ++rep) {
        std::vector<Neighbor> got(nq * k);
        index.search_batch(data.data(), nq, k, 40, got.data(), 2);
        for (std::size_t i = 0; i < nq * k; ++i) {
          if (got[i].label != expected[i].label) ++mismatches;
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(mismatches.load(), 0);
}

TEST(Concurrency, MutationsBlockInsteadOfCorrupting) {
  // Interleave writers (add / mark_deleted / set_ef) with readers. The
  // shared_mutex serializes them; under TSan this must be race-free.
  const std::size_t n = 2000;
  const auto data = test::clustered_vectors(n * 2, kDim, 9);
  Index index(Params{kDim});
  index.add_batch(data.data(), test::iota_labels(n).data(), n, 2);

  std::atomic<bool> stop{false};
  std::atomic<std::size_t> searches{0};
  std::vector<std::thread> readers;
  for (int t = 0; t < 3; ++t) {
    readers.emplace_back([&, t] {
      std::size_t i = static_cast<std::size_t>(t);
      while (!stop.load()) {
        const auto r = index.search(data.data() + (i % n) * kDim, 5, 32);
        EXPECT_FALSE(r.empty());
        ++searches;
        ++i;
      }
    });
  }
  for (std::size_t i = n; i < 2 * n; i += 50) {
    const auto labels = test::iota_labels(50, i);
    index.add_batch(data.data() + i * kDim, labels.data(), 50, 2);
    index.mark_deleted(i - n);
    index.set_ef(16 + (i % 32));
  }
  stop.store(true);
  for (auto& th : readers) th.join();
  EXPECT_EQ(index.size(), 2 * n);
  EXPECT_GT(searches.load(), 0u);
}

}  // namespace
