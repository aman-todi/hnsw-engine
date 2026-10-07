#!/usr/bin/env python3
"""Filtered-search sweep: recall@10 and QPS vs filter selectivity.

For each selectivity s in {1%, 10%, 50%, 90%} a random allow-list of labels is
drawn; exact filtered ground truth is computed by brute force over the
allowed subset, and the engine is queried with the same allow-list (bitset
filter). Every returned id is checked against the filter.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from fetch_data import exact_knn, read_fvecs  # noqa: E402

import hnsw_engine  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--name", required=True)
    p.add_argument("--metric", default="l2")
    p.add_argument("--subset", type=int, default=200_000)
    p.add_argument("--nq", type=int, default=1000)
    p.add_argument("--ef", default="40,80,160,320")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=str(REPO / "bench" / "results" / "filter.csv"))
    a = p.parse_args()

    base = read_fvecs(Path(a.data) / f"{a.name}_base.fvecs", a.subset or None)
    queries = read_fvecs(Path(a.data) / f"{a.name}_query.fvecs", a.nq or None)
    idx = hnsw_engine.Index(dim=base.shape[1], metric=a.metric, M=16, ef_construction=200)
    idx.add(base, num_threads=a.threads)
    rng = np.random.default_rng(0)
    rows = []
    for sel in (0.01, 0.10, 0.50, 0.90):
        mask = rng.random(len(base)) < sel
        allowed = np.flatnonzero(mask)
        gt = allowed[exact_knn(base[allowed], queries, 10, a.metric)]
        for ef in (int(e) for e in a.ef.split(",")):
            t0 = time.perf_counter()
            ids, _ = idx.search(queries, k=10, ef=ef, num_threads=a.threads, filter=mask)
            wall = time.perf_counter() - t0
            valid = ids[ids >= 0]
            assert mask[valid].all(), "filter violated"
            rec = np.mean([len(set(g) & set(t)) / 10 for g, t in zip(ids, gt)])
            row = {"dataset": f"{a.name}[:{len(base)}]", "selectivity": sel, "ef": ef,
                   "recall": round(float(rec), 4), "qps": round(len(queries) / wall, 1),
                   "violations": 0}
            rows.append(row)
            print(row, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
