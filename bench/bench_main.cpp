// Engine-only benchmark harness: build an index (HNSW or brute force) on an
// fvecs dataset, sweep ef_search, and write recall@k / QPS / latency rows as
// CSV. Used for the fast dev loop (M1-M4) and the ablation table.
//
// Example:
//   bench_main --data data --name sift --subset 200000 --ef 10,20,40,80
//              --build-threads 4 --search-threads 1 --out bench/results/sift.csv

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "hnsw/index.hpp"
#include "hnsw/io.hpp"

using namespace hnsw;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
  std::string data_dir = "data";
  std::string name;
  std::string base, query, gt;
  std::size_t subset = 0;
  std::size_t nq = 0;
  std::string metric = "l2";
  std::string mode = "hnsw";
  std::size_t M = 16;
  std::size_t efc = 200;
  std::vector<std::size_t> efs = {10, 20, 40, 80, 160, 320, 640};
  std::size_t k = 10;
  std::size_t build_threads = 0;
  std::size_t search_threads = 1;
  std::size_t reps = 3;
  std::string isa;
  bool prefetch = true;
  std::string out;
  std::string save;
  std::string load;
  bool mmap = false;
  std::string label;  // free-form tag written to the CSV (e.g. ablation step)
  std::string dump;   // write result ids of the first ef to this .ivecs file
};

[[noreturn]] void usage() {
  std::cerr << "usage: bench_main (--name NAME [--data DIR] | --base F --query F [--gt F])\n"
               "  [--subset N] [--nq N] [--metric l2|ip|cosine] [--mode hnsw|brute]\n"
               "  [--M 16] [--efc 200] [--ef 10,20,...] [--k 10] [--build-threads 0] [--search-threads 1]\n"
               "  [--reps 3] [--isa scalar|avx2|avx512|neon] [--no-prefetch] [--save F] [--load F [--mmap]]\n"
               "  [--label TAG] [--dump ids.ivecs] [--out results.csv]\n";
  std::exit(2);
}

std::vector<std::size_t> parse_list(const std::string& s) {
  std::vector<std::size_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoull(item));
  return out;
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (k == "--data")
      a.data_dir = val();
    else if (k == "--name")
      a.name = val();
    else if (k == "--base")
      a.base = val();
    else if (k == "--query")
      a.query = val();
    else if (k == "--gt")
      a.gt = val();
    else if (k == "--subset")
      a.subset = std::stoull(val());
    else if (k == "--nq")
      a.nq = std::stoull(val());
    else if (k == "--metric")
      a.metric = val();
    else if (k == "--mode")
      a.mode = val();
    else if (k == "--M")
      a.M = std::stoull(val());
    else if (k == "--efc")
      a.efc = std::stoull(val());
    else if (k == "--ef")
      a.efs = parse_list(val());
    else if (k == "--k")
      a.k = std::stoull(val());
    else if (k == "--build-threads")
      a.build_threads = std::stoull(val());
    else if (k == "--search-threads")
      a.search_threads = std::stoull(val());
    else if (k == "--reps")
      a.reps = std::max<std::size_t>(1, std::stoull(val()));
    else if (k == "--isa")
      a.isa = val();
    else if (k == "--no-prefetch")
      a.prefetch = false;
    else if (k == "--out")
      a.out = val();
    else if (k == "--save")
      a.save = val();
    else if (k == "--load")
      a.load = val();
    else if (k == "--mmap")
      a.mmap = true;
    else if (k == "--label")
      a.label = val();
    else if (k == "--dump")
      a.dump = val();
    else
      usage();
  }
  if (!a.name.empty()) {
    if (a.base.empty()) a.base = a.data_dir + "/" + a.name + "_base.fvecs";
    if (a.query.empty()) a.query = a.data_dir + "/" + a.name + "_query.fvecs";
    if (a.gt.empty()) a.gt = a.data_dir + "/" + a.name + "_gt.ivecs";
  }
  if (a.base.empty() || a.query.empty() || a.k == 0) usage();
  return a;
}

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

long peak_rss_kb() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return ru.ru_maxrss / 1024;
#else
  return ru.ru_maxrss;
#endif
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return std::nan("");
  std::sort(v.begin(), v.end());
  const std::size_t idx = std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size())));
  return v[idx];
}

double recall_at_k(const std::vector<Neighbor>& got, const io::Matrix<int32_t>& gt, std::size_t nq,
                   std::size_t k) {
  std::size_t hits = 0;
  for (std::size_t i = 0; i < nq; ++i) {
    std::set<uint64_t> truth;
    for (std::size_t j = 0; j < k; ++j) truth.insert(static_cast<uint64_t>(gt.row(i)[j]));
    for (std::size_t j = 0; j < k; ++j) hits += truth.count(got[i * k + j].label);
  }
  return static_cast<double>(hits) / static_cast<double>(nq * k);
}

struct RunResult {
  double recall = 0, qps = 0, p50 = 0, p95 = 0, p99 = 0;
};

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  if (!a.isa.empty()) simd::set_active(simd::parse_isa(a.isa));
  tuning::set_prefetch(a.prefetch);
  const Metric metric = parse_metric(a.metric);

  auto base = io::read_fvecs(a.base, a.subset);
  auto queries = io::read_fvecs(a.query, a.nq);
  const std::size_t n = base.rows, dim = base.cols, nq = queries.rows, k = a.k;
  if (queries.cols != dim) {
    std::cerr << "query dim " << queries.cols << " != base dim " << dim << "\n";
    return 1;
  }
  std::cerr << "dataset: n=" << n << " dim=" << dim << " nq=" << nq << " metric=" << a.metric
            << " isa=" << simd::isa_name(simd::active().isa) << "\n";

  std::vector<uint64_t> labels(n);
  for (std::size_t i = 0; i < n; ++i) labels[i] = i;

  // Ground truth: the provided file is only valid for the full base set; for
  // subsets (or when missing) compute it exactly with brute force.
  io::Matrix<int32_t> gt;
  bool have_gt = false;
  if (!a.gt.empty() && a.subset == 0 && std::filesystem::exists(a.gt)) {
    gt = io::read_ivecs(a.gt, nq);
    have_gt = gt.rows == nq && gt.cols >= k;
  }
  if (!have_gt) {
    std::cerr << "computing exact ground truth by brute force...\n";
    const auto t0 = Clock::now();
    BruteForceIndex bf(dim, metric);
    bf.add(base.data.data(), labels.data(), n);
    std::vector<Neighbor> res(nq * k);
    bf.search_batch(queries.data.data(), nq, k, res.data(), 0);
    gt.rows = nq;
    gt.cols = k;
    gt.data.resize(nq * k);
    for (std::size_t i = 0; i < nq * k; ++i) gt.data[i] = static_cast<int32_t>(res[i].label);
    std::cerr << "  done in " << seconds_since(t0) << " s\n";
  }

  std::optional<Index> index;
  std::optional<BruteForceIndex> brute;
  double build_s = 0;
  std::size_t mem_bytes = 0;
  if (a.mode == "brute") {
    const auto t0 = Clock::now();
    brute.emplace(dim, metric);
    brute->add(base.data.data(), labels.data(), n);
    build_s = seconds_since(t0);
    mem_bytes = n * dim * sizeof(float);
  } else if (!a.load.empty()) {
    const auto t0 = Clock::now();
    index.emplace(Index::load(a.load, a.mmap));
    build_s = seconds_since(t0);
    mem_bytes = index->memory_usage();
    std::cerr << "loaded index in " << build_s << " s\n";
  } else {
    Params p{dim, metric, a.M, a.efc};
    index.emplace(p);
    const auto t0 = Clock::now();
    index->add_batch(base.data.data(), labels.data(), n, a.build_threads);
    build_s = seconds_since(t0);
    mem_bytes = index->memory_usage();
    std::cerr << "built index in " << build_s << " s (max level " << index->max_level() << ")\n";
    if (!a.save.empty()) index->save(a.save);
  }
  // Release the raw base vectors; the index keeps its own copy.
  if (index) {
    base.data.clear();
    base.data.shrink_to_fit();
  }

  std::vector<Neighbor> got(nq * k);
  auto run_once = [&](std::size_t ef) {
    RunResult r;
    std::vector<double> lat;
    const auto t0 = Clock::now();
    if (a.search_threads == 1) {
      lat.reserve(nq);
      for (std::size_t i = 0; i < nq; ++i) {
        const auto q0 = Clock::now();
        const auto res = brute ? brute->search(queries.row(i), k) : index->search(queries.row(i), k, ef);
        lat.push_back(seconds_since(q0) * 1e6);
        for (std::size_t j = 0; j < k; ++j) {
          got[i * k + j] = j < res.size() ? res[j] : Neighbor{kInvalidLabel, 0.0f};
        }
      }
    } else if (brute) {
      brute->search_batch(queries.data.data(), nq, k, got.data(), a.search_threads);
    } else {
      index->search_batch(queries.data.data(), nq, k, ef, got.data(), a.search_threads);
    }
    const double wall = seconds_since(t0);
    r.recall = recall_at_k(got, gt, nq, k);
    r.qps = static_cast<double>(nq) / wall;
    r.p50 = percentile(lat, 0.50);
    r.p95 = percentile(lat, 0.95);
    r.p99 = percentile(lat, 0.99);
    return r;
  };

  std::ofstream csv_file;
  std::ostream* csv = &std::cout;
  if (!a.out.empty()) {
    std::filesystem::path out_path(a.out);
    if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
    const bool exists = std::filesystem::exists(out_path) && std::filesystem::file_size(out_path) > 0;
    csv_file.open(a.out, std::ios::app);
    csv = &csv_file;
    if (!exists) {
      *csv << "label,dataset,n,dim,metric,mode,isa,prefetch,M,efc,build_threads,build_s,mem_bytes,"
              "peak_rss_kb,ef,k,search_threads,recall,qps,p50_us,p95_us,p99_us\n";
    }
  } else {
    *csv << "label,dataset,n,dim,metric,mode,isa,prefetch,M,efc,build_threads,build_s,mem_bytes,"
            "peak_rss_kb,ef,k,search_threads,recall,qps,p50_us,p95_us,p99_us\n";
  }

  const std::vector<std::size_t> efs = brute ? std::vector<std::size_t>{0} : a.efs;
  run_once(efs.front());  // warm-up
  if (!a.dump.empty()) {
    std::vector<int32_t> ids(nq * k);
    for (std::size_t i = 0; i < nq * k; ++i) ids[i] = static_cast<int32_t>(got[i].label);
    io::write_ivecs(a.dump, ids.data(), nq, k);
  }
  for (std::size_t ef : efs) {
    std::vector<RunResult> runs;
    for (std::size_t r = 0; r < a.reps; ++r) runs.push_back(run_once(ef));
    std::sort(runs.begin(), runs.end(), [](const RunResult& x, const RunResult& y) { return x.qps < y.qps; });
    const RunResult& med = runs[runs.size() / 2];
    char line[1024];
    std::snprintf(line, sizeof(line),
                  "%s,%s,%zu,%zu,%s,%s,%s,%d,%zu,%zu,%zu,%.3f,%zu,%ld,%zu,%zu,%zu,%.5f,%.1f,%.1f,%.1f,%.1f\n",
                  a.label.c_str(), a.name.empty() ? a.base.c_str() : a.name.c_str(), n, dim, a.metric.c_str(),
                  a.mode.c_str(), std::string(simd::isa_name(simd::active().isa)).c_str(), a.prefetch ? 1 : 0,
                  a.M, a.efc, a.build_threads, build_s, mem_bytes, peak_rss_kb(), ef, k, a.search_threads,
                  med.recall, med.qps, med.p50, med.p95, med.p99);
    *csv << line;
    csv->flush();
    std::cerr << line;
  }
  return 0;
}
