"""Shared helpers for the benchmark report scripts (plot.py, report.py)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO / "bench" / "results"


def fnum(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def read_env(results_dir: Path) -> dict:
    out = {}
    p = results_dir / "environment.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            k, _, v = line.partition(": ")
            out[k] = v
    return out


def qps_at_recall(rows: list[dict], lib: str, mode: str, target: float) -> float:
    """QPS at exactly `target` recall@10, read off the library's recall-vs-QPS
    curve (points ordered by ef). Between the two ef points that bracket the
    target, log(QPS) is interpolated linearly in recall; if even the smallest
    ef already exceeds the target, that point's QPS is returned (no
    extrapolation). NaN if the library never reaches the target.

    Taking the best *measured* point at recall >= target instead makes the
    result jump whenever a curve lands just below the threshold (e.g. 0.9497
    vs 0.95), so a library can look 40% slower while being faster at the same
    ef; interpolation removes that artefact."""
    pts = sorted(((int(fnum(r["ef"])), fnum(r["recall"]), fnum(r["qps"]))
                  for r in rows if r["library"] == lib and r["mode"] == mode), key=lambda p: p[0])
    for i, (_, rec, qps) in enumerate(pts):
        if rec >= target:
            if i == 0:
                return qps
            _, r0, q0 = pts[i - 1]
            if rec <= r0:
                return qps
            w = (target - r0) / (rec - r0)
            return math.exp(math.log(q0) + w * (math.log(qps) - math.log(q0)))
    return math.nan
