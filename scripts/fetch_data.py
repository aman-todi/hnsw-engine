#!/usr/bin/env python3
"""Download ann-benchmarks datasets and convert them to fvecs/ivecs.

Real datasets (default):
    python scripts/fetch_data.py sift glove gist [--data-dir data] [--subset N]

Each HDF5 file (train/test/neighbors/distances) is downloaded with resume
support, its SHA-256 is printed, and it is converted to
    <name>_base.fvecs, <name>_query.fvecs, <name>_gt.ivecs
which the C++ harness reads without an HDF5 dependency.

Synthetic stand-ins (for offline environments):
    python scripts/fetch_data.py --synthetic sift glove gist [--n N] [--nq N]

generates datasets named synth-<name> with the same dimensionality, metric
and (by default) size as the real ones, drawn from a Gaussian mixture with a
low intrinsic dimension, and computes exact top-100 ground truth with NumPy
(an implementation independent of the engine). Results on synthetic data are
always labelled as such; they are not SIFT/GloVe/GIST numbers.

With --subset N the base set is truncated to N vectors and the ground truth
is recomputed exactly for that subset (the provided neighbors are only valid
for the full base set).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

DATASETS = {
    "sift": {"url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5", "dim": 128,
             "n": 1_000_000, "nq": 10_000, "metric": "l2"},
    "glove": {"url": "http://ann-benchmarks.com/glove-100-angular.hdf5", "dim": 100,
              "n": 1_183_514, "nq": 10_000, "metric": "cosine"},
    "gist": {"url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5", "dim": 960,
             "n": 1_000_000, "nq": 1_000, "metric": "l2"},
}
GT_K = 100


# ---------------------------------------------------------------------------
# fvecs / ivecs
# ---------------------------------------------------------------------------

def write_vecs(path: Path, arr: np.ndarray, dtype) -> None:
    path = Path(path)
    arr = np.ascontiguousarray(arr, dtype=dtype)
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = arr.view(np.int32)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)


def write_fvecs(path: Path, arr: np.ndarray) -> None:
    write_vecs(path, arr, np.float32)


def write_ivecs(path: Path, arr: np.ndarray) -> None:
    write_vecs(path, arr, np.int32)


def read_fvecs(path: Path, max_rows: int | None = None) -> np.ndarray:
    if os.path.getsize(path) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    raw = np.memmap(path, dtype=np.int32, mode="r")  # only the rows we keep are read
    d = int(raw[0])
    rows = raw.reshape(-1, d + 1)
    if max_rows:
        rows = rows[:max_rows]
    if not (rows[:, 0] == d).all():
        raise ValueError(f"{path}: inconsistent dimensions")
    return np.ascontiguousarray(rows[:, 1:]).view(np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy()


# ---------------------------------------------------------------------------
# Exact ground truth (NumPy, blockwise, exact float64 re-ranking)
# ---------------------------------------------------------------------------

def exact_knn(base: np.ndarray, queries: np.ndarray, k: int, metric: str,
              block: int | None = None) -> np.ndarray:
    """Exact top-k ids. A float32 matmul shortlists 2k+16 candidates per
    query, which are then re-ranked with float64 distances."""
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if metric == "cosine":
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
        queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    k = min(k, base.shape[0])
    if block is None:  # keep the (block x n) distance matrix around 400 MB
        block = max(1, min(1024, int(1e8 // max(1, base.shape[0]))))
    shortlist = min(base.shape[0], 2 * k + 16)
    base_sq = np.einsum("ij,ij->i", base, base)  # float32, no full-size temporary
    out = np.empty((queries.shape[0], k), dtype=np.int32)
    t0 = time.time()
    for s in range(0, queries.shape[0], block):
        q = queries[s:s + block]
        dots = q @ base.T  # (b, n)
        if metric == "l2":
            score = base_sq[None, :] - 2.0 * dots  # + |q|^2 is constant per row
        else:
            score = -dots
        cand = np.argpartition(score, shortlist - 1, axis=1)[:, :shortlist]
        for i in range(q.shape[0]):
            c = cand[i]
            v = base[c].astype(np.float64)
            qq = q[i].astype(np.float64)
            if metric == "l2":
                d = ((v - qq) ** 2).sum(axis=1)
            else:
                d = 1.0 - v @ qq
            order = np.lexsort((c, d))[:k]
            out[s + i] = c[order]
        if (s // block) % max(1, queries.shape[0] // block // 10) == 0:
            print(f"    ground truth {s + q.shape[0]}/{queries.shape[0]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Download + convert
# ---------------------------------------------------------------------------

USER_AGENT = "Mozilla/5.0 (compatible; hnsw-engine-fetch/0.1; +https://github.com/aman-todi/hnsw-engine)"


def _download_curl(url: str, part: Path) -> None:
    """curl with resume (-C -), redirects, retries and a progress bar."""
    subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-C", "-", "-A", USER_AGENT,
                    "-o", str(part), url], check=True)


def _download_urllib(url: str, part: Path) -> None:
    have = part.stat().st_size if part.exists() else 0
    # Some servers reject Python's default "Python-urllib" User-Agent with 403.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming at {have / 1e6:.1f} MB")
    with urllib.request.urlopen(req) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored Range: start over
        total = resp.headers.get("Content-Length")
        total = int(total) + have if total else None
        with open(part, "ab" if have else "wb") as f:
            done = have
            last = time.time()
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    pct = f" ({100 * done / total:.1f}%)" if total else ""
                    print(f"    {done / 1e6:.0f} MB{pct}", flush=True)
                    last = time.time()


def download(url: str, dest: Path) -> None:
    """Download with resume (curl if available, else urllib); prints the SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"  {dest} already present")
    else:
        print(f"  downloading {url}")
        if shutil.which("curl"):
            try:
                _download_curl(url, part)
            except subprocess.CalledProcessError as e:
                raise OSError(f"curl failed (exit {e.returncode}) for {url}") from e
        else:
            _download_urllib(url, part)
        os.replace(part, dest)
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    print(f"  sha256({dest.name}) = {h.hexdigest()}")


def convert_real(name: str, data_dir: Path, subset: int | None) -> None:
    import h5py  # only needed for real datasets

    spec = DATASETS[name]
    h5 = data_dir / Path(spec["url"]).name
    download(spec["url"], h5)
    with h5py.File(h5, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)
    out_name = name if not subset else f"{name}-{subset // 1000}k"
    if subset:
        train = train[:subset]
        print(f"  subset {subset}: recomputing exact ground truth")
        neighbors = exact_knn(train, test, GT_K, spec["metric"])
    write_outputs(out_name, data_dir, train, test, neighbors)


def write_outputs(name: str, data_dir: Path, base, queries, gt) -> None:
    write_fvecs(data_dir / f"{name}_base.fvecs", base)
    write_fvecs(data_dir / f"{name}_query.fvecs", queries)
    write_ivecs(data_dir / f"{name}_gt.ivecs", gt)
    print(f"  wrote {name}: base {base.shape}, queries {queries.shape}, gt {gt.shape}")


# ---------------------------------------------------------------------------
# Synthetic stand-ins
# ---------------------------------------------------------------------------

def synthetic(dim: int, n: int, nq: int, seed: int, intrinsic: int = 24,
              clusters: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian mixture on a low-dimensional manifold: every cluster has its
    own random `intrinsic`-dimensional subspace, plus small isotropic noise.
    Real embedding/descriptor sets have low intrinsic dimension too, which is
    what makes graph ANN search effective on them."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(clusters, dim)).astype(np.float32) * 2.0
    bases = rng.normal(0.0, 1.0 / np.sqrt(intrinsic),
                       size=(clusters, intrinsic, dim)).astype(np.float32)
    weights = rng.dirichlet(np.full(clusters, 2.0))

    def draw(count: int) -> np.ndarray:
        out = np.empty((count, dim), dtype=np.float32)
        step = 50_000
        for s in range(0, count, step):
            m = min(step, count - s)
            c = rng.choice(clusters, size=m, p=weights)
            z = rng.normal(0.0, 1.0, size=(m, intrinsic)).astype(np.float32)
            x = centers[c] + rng.normal(0.0, 0.05, size=(m, dim)).astype(np.float32)
            for j in np.unique(c):  # per-cluster matmul: no (m, intrinsic, dim) gather
                rows = np.flatnonzero(c == j)
                x[rows] += z[rows] @ bases[j]
            out[s:s + m] = x
        return out

    return draw(n), draw(nq)


def make_synthetic(name: str, data_dir: Path, n: int | None, nq: int | None) -> None:
    spec = DATASETS[name]
    n = n or spec["n"]
    nq = nq or spec["nq"]
    out_name = f"synth-{name}"
    print(f"generating {out_name}: n={n} nq={nq} dim={spec['dim']} metric={spec['metric']}")
    seed = {"sift": 1, "glove": 2, "gist": 3}[name]
    base, queries = synthetic(spec["dim"], n, nq, seed)
    print("  computing exact ground truth (NumPy)")
    gt = exact_knn(base, queries, GT_K, spec["metric"])
    write_outputs(out_name, data_dir, base, queries, gt)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    p.add_argument("--data-dir", default="data", type=Path)
    p.add_argument("--subset", type=int, default=None, help="truncate base set (real datasets)")
    p.add_argument("--synthetic", action="store_true", help="generate offline stand-ins")
    p.add_argument("--n", type=int, default=None, help="synthetic base size")
    p.add_argument("--nq", type=int, default=None, help="synthetic query count")
    args = p.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        if args.synthetic:
            make_synthetic(name, args.data_dir, args.n, args.nq)
        else:
            print(f"{name}:")
            try:
                convert_real(name, args.data_dir, args.subset)
            except (OSError, urllib.error.URLError) as e:
                print(f"  failed: {e}\n  If the server keeps refusing, download the file manually into "
                      f"{args.data_dir}/ (e.g. with a browser) and re-run; existing files are reused.\n"
                      "  No network at all? Use --synthetic.", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
