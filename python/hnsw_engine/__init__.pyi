from collections.abc import Callable, Sequence
from os import PathLike
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

__version__: str

def simd_isa() -> str: ...

_Filter = Union[None, Callable[[int], bool], npt.NDArray[np.bool_], npt.NDArray[np.integer], Sequence[int]]

class Index:
    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        M: int = 16,
        ef_construction: int = 200,
        max_elements: int = 0,
        seed: int = 100,
    ) -> None: ...
    def add(
        self,
        vectors: npt.ArrayLike,
        ids: Optional[npt.ArrayLike] = None,
        num_threads: int = 0,
    ) -> None:
        """Insert float32 vectors of shape (n, dim). ids default to
        arange(len(self), len(self) + n). Releases the GIL."""
    def search(
        self,
        queries: npt.ArrayLike,
        k: int = 10,
        ef: Optional[int] = None,
        num_threads: int = 0,
        filter: _Filter = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """k-NN search. Returns (ids, distances) of shape (nq, k) (or (k,) for
        a 1-D query). Missing results are id -1 / distance inf. Releases the
        GIL. `filter` is a callable(label) -> bool, a bool mask indexed by
        label, or an array of allowed labels."""
    def set_ef(self, ef: int) -> None: ...
    def mark_deleted(self, label: int) -> None: ...
    def is_deleted(self, label: int) -> bool: ...
    def get_vector(self, label: int) -> npt.NDArray[np.float32]: ...
    def save(self, path: Union[str, PathLike[str]]) -> None: ...
    @staticmethod
    def load(path: Union[str, PathLike[str]], mmap: bool = False) -> Index: ...
    def __len__(self) -> int: ...
    def __contains__(self, label: int) -> bool: ...
    @property
    def dim(self) -> int: ...
    @property
    def metric(self) -> str: ...
    @property
    def M(self) -> int: ...
    @property
    def ef_construction(self) -> int: ...
    @property
    def max_elements(self) -> int: ...
    @property
    def ef(self) -> int: ...
    @ef.setter
    def ef(self, value: int) -> None: ...
    @property
    def read_only(self) -> bool: ...
    @property
    def max_level(self) -> int: ...
    @property
    def memory_usage(self) -> int: ...
