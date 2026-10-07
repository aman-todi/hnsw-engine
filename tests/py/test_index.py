"""pytest suite for the hnsw_engine Python bindings."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import hnsw_engine

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from fetch_data import read_ivecs, write_fvecs  # noqa: E402


def clustered(n: int, dim: int, seed: int, clusters: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, size=(clusters, dim))
    c = rng.integers(0, clusters, size=n)
    return (centers[c] + rng.normal(size=(n, dim))).astype(np.float32)


def brute_force(base: np.ndarray, queries: np.ndarray, k: int, metric: str = "l2") -> np.ndarray:
    b = base.astype(np.float64)
    q = queries.astype(np.float64)
    if metric == "cosine":
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    if metric == "l2":
        d = (q ** 2).sum(1)[:, None] - 2 * q @ b.T + (b ** 2).sum(1)[None, :]
    else:
        d = 1.0 - q @ b.T
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def recall(got: np.ndarray, truth: np.ndarray) -> float:
    k = truth.shape[1]
    return float(np.mean([len(set(g) & set(t)) / k for g, t in zip(got, truth)]))


@pytest.fixture(scope="module")
def data():
    base = clustered(3000, 32, 1)
    queries = clustered(100, 32, 2)
    return base, queries


@pytest.fixture(scope="module")
def index(data):
    base, _ = data
    idx = hnsw_engine.Index(dim=32, metric="l2", M=16, ef_construction=200)
    idx.add(base)
    return idx


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

def test_basic_properties(index):
    assert len(index) == 3000
    assert index.dim == 32
    assert index.metric == "l2"
    assert index.M == 16
    assert index.ef_construction == 200
    assert not index.read_only
    assert 0 in index and 2999 in index and 3000 not in index
    assert "Index dim=32" in repr(index)
    assert hnsw_engine.simd_isa() in {"scalar", "avx2", "avx512", "neon"}


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_recall_vs_brute_force(metric):
    base = clustered(2000, 24, 3)
    queries = clustered(50, 24, 4)
    idx = hnsw_engine.Index(dim=24, metric=metric)
    idx.add(base)
    ids, dists = idx.search(queries, k=10, ef=400 if metric == "ip" else 100)
    assert ids.shape == (50, 10) and ids.dtype == np.int64
    assert dists.shape == (50, 10) and dists.dtype == np.float32
    assert np.all(np.diff(dists, axis=1) >= 0)
    assert recall(ids, brute_force(base, queries, 10, metric)) >= (0.9 if metric == "ip" else 0.95)


def test_single_query_returns_1d(index, data):
    _, queries = data
    ids, dists = index.search(queries[0], k=5)
    assert ids.shape == (5,) and dists.shape == (5,)
    ids2, _ = index.search(queries[:1], k=5)
    np.testing.assert_array_equal(ids, ids2[0])


def test_k_larger_than_size_pads_with_minus_one():
    idx = hnsw_engine.Index(dim=4)
    idx.add(np.eye(4, dtype=np.float32))
    ids, dists = idx.search(np.ones((2, 4), np.float32), k=6)
    assert (ids[:, :4] >= 0).all()
    assert (ids[:, 4:] == -1).all()
    assert np.isinf(dists[:, 4:]).all()


def test_empty_index_search():
    idx = hnsw_engine.Index(dim=3)
    ids, _ = idx.search(np.zeros(3, np.float32), k=2)
    assert (ids == -1).all()


# ---------------------------------------------------------------------------
# dtype / shape / contiguity handling
# ---------------------------------------------------------------------------

def test_float64_and_noncontiguous_are_converted(data):
    base, queries = data
    idx = hnsw_engine.Index(dim=32)
    idx.add(base[:500].astype(np.float64))
    ref, _ = idx.search(queries[:10], k=5, ef=50)
    fortran = np.asfortranarray(queries[:10])
    strided = np.repeat(queries[:10], 2, axis=0)[::2]
    assert not strided.flags["C_CONTIGUOUS"] or not fortran.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(idx.search(fortran, k=5, ef=50)[0], ref)
    np.testing.assert_array_equal(idx.search(strided, k=5, ef=50)[0], ref)
    np.testing.assert_array_equal(idx.search(queries[:10].tolist(), k=5, ef=50)[0], ref)


def test_shape_errors(index):
    with pytest.raises(ValueError, match="shape"):
        index.search(np.zeros((3, 31), np.float32))
    with pytest.raises(ValueError):
        index.search(np.zeros(31, np.float32))
    with pytest.raises(ValueError):
        index.search(np.zeros((2, 2, 32), np.float32))
    with pytest.raises(ValueError, match="k must be"):
        index.search(np.zeros(32, np.float32), k=0)
    with pytest.raises(TypeError):
        index.search(np.array(["a", "b"]))
    idx = hnsw_engine.Index(dim=8)
    with pytest.raises(ValueError):
        idx.add(np.zeros((4, 7), np.float32))


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        hnsw_engine.Index(dim=0)
    with pytest.raises(ValueError):
        hnsw_engine.Index(dim=8, metric="hamming")
    with pytest.raises(ValueError):
        hnsw_engine.Index(dim=8, M=1)


# ---------------------------------------------------------------------------
# ids / labels
# ---------------------------------------------------------------------------

def test_custom_ids_and_default_arange():
    base = clustered(300, 8, 5)
    idx = hnsw_engine.Index(dim=8)
    idx.add(base[:100], ids=np.arange(1000, 1100))
    idx.add(base[100:200])  # defaults to arange(len, len + n) = 100..199
    assert 1000 in idx and 150 in idx and 99 not in idx
    ids, dists = idx.search(base[[5, 150]], k=1)
    assert ids[0, 0] == 1005 and ids[1, 0] == 150
    assert dists[0, 0] == pytest.approx(0.0, abs=1e-5)
    idx.add(base[200:202], ids=np.array([7, 8], dtype=np.uint64))
    assert 7 in idx


def test_id_errors():
    idx = hnsw_engine.Index(dim=4)
    v = np.random.default_rng(0).random((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        idx.add(v, ids=[1, -2, 3])
    with pytest.raises(ValueError, match="shape"):
        idx.add(v, ids=[1, 2])
    with pytest.raises(TypeError):
        idx.add(v, ids=[1.5, 2.5, 3.5])
    with pytest.raises(ValueError, match="duplicate"):
        idx.add(v, ids=[1, 1, 2])
    idx.add(v, ids=[1, 2, 3])
    with pytest.raises(ValueError, match="already exists"):
        idx.add(v[:1], ids=[2])
    assert len(idx) == 3


def test_get_vector_and_cosine_normalization():
    v = np.array([[3.0, 4.0]], np.float32)
    idx = hnsw_engine.Index(dim=2, metric="cosine")
    idx.add(v)
    np.testing.assert_allclose(idx.get_vector(0), [0.6, 0.8], rtol=1e-6)
    np.testing.assert_array_equal(v, [[3.0, 4.0]])  # caller buffer untouched


# ---------------------------------------------------------------------------
# Filters and soft delete
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["callable", "mask", "ids", "list"])
def test_filter_variants(index, data, kind):
    base, queries = data
    allowed = np.zeros(len(base), dtype=bool)
    allowed[::7] = True
    flt = {
        "callable": lambda label: bool(allowed[label]),
        "mask": allowed,
        "ids": np.flatnonzero(allowed),
        "list": np.flatnonzero(allowed).tolist(),
    }[kind]
    ids, _ = index.search(queries[:30], k=10, ef=100, filter=flt, num_threads=2)
    assert allowed[ids[ids >= 0]].all()
    truth = np.flatnonzero(allowed)[brute_force(base[allowed], queries[:30], 10)]
    assert recall(ids, truth) >= 0.9


def test_filter_callback_exception_propagates(index, data):
    def bad(label):
        raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        index.search(data[1][:4], k=3, filter=bad, num_threads=2)


def test_filter_type_error(index, data):
    with pytest.raises(TypeError):
        index.search(data[1][:2], k=3, filter=np.array([0.5, 1.5]))
    with pytest.raises(TypeError):
        index.search(data[1][:2], k=3, filter="nope")


def test_mark_deleted():
    base = clustered(1000, 16, 6)
    idx = hnsw_engine.Index(dim=16)
    idx.add(base)
    for label in range(0, 1000, 2):
        idx.mark_deleted(label)
    assert idx.is_deleted(0) and not idx.is_deleted(1)
    ids, _ = idx.search(base[:50], k=10, ef=64)
    assert (ids % 2 == 1).all()
    with pytest.raises(ValueError):
        idx.mark_deleted(5000)
    assert len(idx) == 1000


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mmap", [False, True])
def test_save_load(tmp_path, index, data, mmap):
    _, queries = data
    index.set_ef(77)
    path = tmp_path / "index.bin"
    index.save(str(path))
    loaded = hnsw_engine.Index.load(str(path), mmap=mmap)
    assert len(loaded) == len(index)
    assert loaded.ef == 77
    assert loaded.read_only == mmap
    a = index.search(queries, k=10)
    b = loaded.search(queries, k=10)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    if mmap:
        with pytest.raises(RuntimeError, match="read-only"):
            loaded.add(queries[:1], ids=[999999])
    index.ef = 10


def test_load_corrupt_file(tmp_path, index):
    path = tmp_path / "index.bin"
    index.save(str(path))
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    bad = tmp_path / "bad.bin"
    bad.write_bytes(bytes(raw))
    for mmap in (False, True):
        with pytest.raises(RuntimeError, match="corrupt"):
            hnsw_engine.Index.load(str(bad), mmap=mmap)
    (tmp_path / "short.bin").write_bytes(path.read_bytes()[:1000])
    with pytest.raises(RuntimeError):
        hnsw_engine.Index.load(str(tmp_path / "short.bin"))
    with pytest.raises(RuntimeError):
        hnsw_engine.Index.load(str(tmp_path / "missing.bin"))


# ---------------------------------------------------------------------------
# Determinism and parity with the C++ harness
# ---------------------------------------------------------------------------

def test_deterministic_single_thread(data):
    base, queries = data
    results = []
    for _ in range(2):
        idx = hnsw_engine.Index(dim=32, seed=7)
        idx.add(base, num_threads=1)
        results.append(idx.search(queries, k=10, ef=40, num_threads=1))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def _find_bench_main() -> Path | None:
    env = os.environ.get("HNSW_BENCH_MAIN")
    if env:
        return Path(env)
    for preset in ("release", "bench", "dev"):
        p = REPO / "build" / preset / "bench" / "bench_main"
        if p.exists():
            return p
    return shutil.which("bench_main") and Path(shutil.which("bench_main"))


def test_parity_with_cpp_harness(tmp_path, data):
    bench_main = _find_bench_main()
    if bench_main is None or not bench_main.exists():
        pytest.skip("bench_main not built (set HNSW_BENCH_MAIN)")
    base, queries = data
    write_fvecs(tmp_path / "p_base.fvecs", base)
    write_fvecs(tmp_path / "p_query.fvecs", queries)
    dump = tmp_path / "ids.ivecs"
    subprocess.run([str(bench_main), "--data", str(tmp_path), "--name", "p", "--M", "12", "--efc", "100",
                    "--ef", "32", "--k", "10", "--build-threads", "1", "--search-threads", "1",
                    "--reps", "1", "--dump", str(dump)],
                   check=True, capture_output=True)
    cpp_ids = read_ivecs(dump)
    idx = hnsw_engine.Index(dim=32, M=12, ef_construction=100)  # same default seed (100)
    idx.add(base, num_threads=1)
    py_ids, _ = idx.search(queries, k=10, ef=32, num_threads=1)
    np.testing.assert_array_equal(py_ids, cpp_ids)


# ---------------------------------------------------------------------------
# GIL release
# ---------------------------------------------------------------------------

def test_concurrent_searches_from_python_threads(index, data):
    _, queries = data
    expected, _ = index.search(queries, k=10, ef=50, num_threads=1)
    errors = []

    def worker():
        try:
            for _ in range(20):
                got, _ = index.search(queries, k=10, ef=50, num_threads=1)
                np.testing.assert_array_equal(got, expected)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_gil_released_during_add():
    """While a long add() runs in a background thread, the main thread keeps
    executing Python bytecode (it could not if the GIL were held)."""
    base = np.random.default_rng(0).random((40000, 64), dtype=np.float32)
    idx = hnsw_engine.Index(dim=64, ef_construction=100)
    done = threading.Event()

    def build():
        idx.add(base, num_threads=1)
        done.set()

    t = threading.Thread(target=build)
    t.start()
    ticks = 0
    time.sleep(0.05)
    while not done.is_set():
        ticks += 1
        time.sleep(0.001)
    t.join()
    assert len(idx) == 40000
    assert ticks > 10


def test_mutation_while_searching_from_threads(data):
    """add/mark_deleted from one thread while others search: must not crash
    (mutations take the exclusive lock)."""
    base, queries = data
    idx = hnsw_engine.Index(dim=32)
    idx.add(base[:1000])
    stop = threading.Event()
    errors = []

    def reader():
        try:
            while not stop.is_set():
                ids, _ = idx.search(queries[:10], k=5)
                assert (ids >= 0).all()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in readers:
        t.start()
    for s in range(1000, 3000, 250):
        idx.add(base[s:s + 250], ids=np.arange(s, s + 250))
        idx.mark_deleted(s - 1000)
    stop.set()
    for t in readers:
        t.join()
    assert not errors
    assert len(idx) == 3000
