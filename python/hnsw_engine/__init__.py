"""hnsw_engine: HNSW approximate nearest-neighbor search (C++20 core).

    >>> import numpy as np, hnsw_engine
    >>> idx = hnsw_engine.Index(dim=128, metric="l2", M=16, ef_construction=200)
    >>> idx.add(np.random.rand(1000, 128).astype(np.float32))
    >>> ids, dists = idx.search(np.random.rand(5, 128).astype(np.float32), k=10)
"""

from ._core import Index, __version__, simd_isa

__all__ = ["Index", "__version__", "simd_isa"]
