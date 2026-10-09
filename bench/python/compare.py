#!/usr/bin/env python3
"""Like-for-like benchmark: hnsw_engine vs hnswlib vs FAISS IndexHNSWFlat.

All three libraries are driven through Python with identical M,
ef_construction, metric, dataset, k and thread counts. Each library runs in
its own subprocess so peak RSS is measured in isolation.

For every library:
  * build the index with --threads threads (and optionally with 1 thread),
    recording build time, RSS growth, peak RSS and serialized index size;
  * sweep ef_search over --ef in two query modes
      - "single": 1 thread, one query per call (latency: p50/p95/p99),
      - "batch":  all queries in one call on --threads threads (throughput);
    after a warm-up pass, each point is repeated --reps times and the run
    with the median QPS is reported.

Example:
    python bench/python/compare.py --data data --name synth-sift --metric l2 \\
        --threads 4 --out bench/results/synth-sift.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from fetch_data import exact_knn, read_fvecs, read_ivecs  # noqa: E402

LIBS = ("engine", "hnswlib", "faiss")
FIELDS = ["library", "version", "dataset", "n", "dim", "metric", "M", "efc", "threads",
          "build_s", "build_1t_s", "index_bytes", "rss_delta_mb", "peak_rss_mb", "mode", "ef", "k",
          "recall", "qps", "p50_us", "p95_us", "p99_us"]


def current_rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:  # no /proc (macOS): ask ps for the current resident set size (KiB)
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                 capture_output=True, text=True, check=True).stdout
            return int(out.strip()) / 1024
        except (OSError, ValueError, subprocess.CalledProcessError):
            return float("nan")


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def version_of(lib: str) -> str:
    dist = {"engine": "hnsw-engine", "hnswlib": "hnswlib", "faiss": "faiss-cpu"}[lib]
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Library adapters
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, dim, metric, M, efc, seed=100):
        import hnsw_engine
        self.idx = hnsw_engine.Index(dim=dim, metric=metric, M=M, ef_construction=efc, seed=seed)

    def build(self, x, threads):
        self.idx.add(x, num_threads=threads)

    def set_ef(self, ef):
        self.idx.set_ef(ef)

    def search(self, q, k, threads):
        return self.idx.search(q, k=k, num_threads=threads)[0]

    def save(self, path):
        self.idx.save(path)


class Hnswlib:
    def __init__(self, dim, metric, M, efc, seed=100):
        import hnswlib
        space = {"l2": "l2", "ip": "ip", "cosine": "cosine"}[metric]
        self.idx = hnswlib.Index(space=space, dim=dim)
        self.M, self.efc, self.seed = M, efc, seed

    def build(self, x, threads):
        self.idx.init_index(max_elements=len(x), M=self.M, ef_construction=self.efc, random_seed=self.seed)
        self.idx.add_items(x, np.arange(len(x)), num_threads=threads)

    def set_ef(self, ef):
        self.idx.set_ef(ef)

    def search(self, q, k, threads):
        return self.idx.knn_query(q, k=k, num_threads=threads)[0]

    def save(self, path):
        self.idx.save_index(path)


class Faiss:
    def __init__(self, dim, metric, M, efc, seed=100):
        import faiss
        self.faiss = faiss
        self.normalize = metric == "cosine"
        fmetric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        self.idx = faiss.IndexHNSWFlat(dim, M, fmetric)
        self.idx.hnsw.efConstruction = efc

    def _prep(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.normalize:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
        return x

    def build(self, x, threads):
        self.faiss.omp_set_num_threads(threads)
        self.idx.add(self._prep(x))

    def set_ef(self, ef):
        self.idx.hnsw.efSearch = ef

    def search(self, q, k, threads):
        self.faiss.omp_set_num_threads(threads)
        return self.idx.search(self._prep(q), k)[1]

    def save(self, path):
        self.faiss.write_index(self.idx, path)


ADAPTERS = {"engine": Engine, "hnswlib": Hnswlib, "faiss": Faiss}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def recall_at_k(got: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = 0
    for g, t in zip(got, gt):
        hits += len(set(g[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(gt) * k)


def run_mode(lib, queries, gt, k, ef, mode, threads):
    lib.set_ef(ef)
    nq = len(queries)
    if mode == "single":
        out = np.empty((nq, k), dtype=np.int64)
        lat = np.empty(nq)
        t0 = time.perf_counter()
        for i in range(nq):
            s = time.perf_counter()
            out[i] = lib.search(queries[i:i + 1], k, 1)[0]
            lat[i] = time.perf_counter() - s
        wall = time.perf_counter() - t0
        p50, p95, p99 = (np.percentile(lat, p) * 1e6 for p in (50, 95, 99))
    else:
        t0 = time.perf_counter()
        out = lib.search(queries, k, threads)
        wall = time.perf_counter() - t0
        p50 = p95 = p99 = float("nan")
    return {"recall": recall_at_k(out, gt, k), "qps": nq / wall, "p50_us": p50, "p95_us": p95, "p99_us": p99}


def worker(args) -> None:
    """Runs one library in this process and prints JSON rows to stdout."""
    base = read_fvecs(args.base, args.subset or None)
    queries = read_fvecs(args.query, args.nq or None)
    gt = np.load(args.gt_npy)
    n, dim = base.shape
    k = args.k
    Adapter = ADAPTERS[args.worker]

    build_1t = float("nan")
    if args.build_1t:
        lib1 = Adapter(dim, args.metric, args.M, args.efc)
        t0 = time.perf_counter()
        lib1.build(base, 1)
        build_1t = time.perf_counter() - t0
        del lib1

    rss0 = current_rss_mb()
    lib = Adapter(dim, args.metric, args.M, args.efc)
    t0 = time.perf_counter()
    lib.build(base, args.threads)
    build_s = time.perf_counter() - t0
    rss_delta = current_rss_mb() - rss0
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as td:
        path = os.path.join(td, "index.bin")
        lib.save(path)
        index_bytes = os.path.getsize(path)
    del base

    common = {"library": args.worker, "version": version_of(args.worker), "dataset": args.dataset,
              "n": n, "dim": dim, "metric": args.metric, "M": args.M, "efc": args.efc,
              "threads": args.threads, "build_s": round(build_s, 3), "build_1t_s": round(build_1t, 3),
              "index_bytes": index_bytes, "rss_delta_mb": round(rss_delta, 1)}
    print(f"[{args.worker}] built n={n} in {build_s:.1f}s (1t: {build_1t:.1f}s), "
          f"index {index_bytes / 2**20:.0f} MB", file=sys.stderr, flush=True)

    for mode in args.modes:
        run_mode(lib, queries, gt, k, args.efs[0], mode, args.threads)  # warm-up
        for ef in args.efs:
            runs = sorted((run_mode(lib, queries, gt, k, ef, mode, args.threads) for _ in range(args.reps)),
                          key=lambda r: r["qps"])
            med = runs[len(runs) // 2]
            row = dict(common, mode=mode, ef=ef, k=k, peak_rss_mb=round(peak_rss_mb(), 1),
                       **{key: (round(v, 5) if key == "recall" else round(v, 1)) for key, v in med.items()})
            print(json.dumps(row), flush=True)
            print(f"[{args.worker}] {mode:6s} ef={ef:4d} recall={med['recall']:.4f} "
                  f"qps={med['qps']:.0f} p50={med['p50_us']:.0f}us", file=sys.stderr, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--name", required=True, help="dataset prefix: <data>/<name>_{base,query}.fvecs")
    p.add_argument("--metric", default="l2", choices=["l2", "ip", "cosine"])
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--nq", type=int, default=0)
    p.add_argument("--libs", default=",".join(LIBS))
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--efc", type=int, default=200)
    p.add_argument("--ef", default="10,20,40,80,160,320,640")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--modes", default="single,batch")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--build-1t", action="store_true", help="also time a single-threaded build")
    p.add_argument("--out", default=None)
    # internal
    p.add_argument("--worker", choices=LIBS, help=argparse.SUPPRESS)
    p.add_argument("--base", help=argparse.SUPPRESS)
    p.add_argument("--query", help=argparse.SUPPRESS)
    p.add_argument("--gt-npy", help=argparse.SUPPRESS)
    p.add_argument("--dataset", help=argparse.SUPPRESS)
    p.add_argument("--tmp-dir", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.efs = [int(e) for e in args.ef.split(",")]
    args.modes = args.modes.split(",")

    if args.worker:
        worker(args)
        return 0

    data = Path(args.data)
    base_path = data / f"{args.name}_base.fvecs"
    query_path = data / f"{args.name}_query.fvecs"
    gt_path = data / f"{args.name}_gt.ivecs"
    queries = read_fvecs(query_path, args.nq or None)
    if args.subset == 0 and gt_path.exists():
        gt = read_ivecs(gt_path)[: len(queries), : args.k]
    else:
        print(f"computing exact ground truth for subset={args.subset}", file=sys.stderr)
        gt = exact_knn(read_fvecs(base_path, args.subset or None), queries, args.k, args.metric)
    dataset = args.name + (f"[:{args.subset}]" if args.subset else "")
    out = Path(args.out or REPO / "bench" / "results" / f"{args.name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"host: {platform.processor() or platform.machine()} cpus={os.cpu_count()} threads={args.threads}",
          file=sys.stderr)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        gt_npy = os.path.join(td, "gt.npy")
        np.save(gt_npy, gt)
        for lib in args.libs.split(","):
            cmd = [sys.executable, __file__, "--worker", lib, "--base", str(base_path), "--query", str(query_path),
                   "--gt-npy", gt_npy, "--dataset", dataset, "--metric", args.metric,
                   "--subset", str(args.subset), "--nq", str(args.nq), "--M", str(args.M),
                   "--efc", str(args.efc), "--ef", args.ef, "--k", str(args.k), "--threads", str(args.threads),
                   "--modes", ",".join(args.modes), "--reps", str(args.reps), "--name", args.name,
                   "--tmp-dir", td]
            if args.build_1t:
                cmd.append("--build-1t")
            res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            if res.returncode != 0:
                print(f"{lib} failed with exit code {res.returncode}", file=sys.stderr)
                return 1
            rows += [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
