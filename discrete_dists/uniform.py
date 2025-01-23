from typing import overload
import numpy as np
import numpy.typing as npt
import discrete_dists.utils.npu as npu
from discrete_dists.distribution import Distribution


class Uniform(Distribution):
    def __init__(self, support: int):
        self._support = support

    @overload
    def update(self, idxs: np.ndarray) -> None: ...
    @overload
    def update(self, idxs: np.ndarray, values: np.ndarray) -> None: ...
    def update(self, idxs: np.ndarray, values: np.ndarray | None = None):
        self._support = max(self._support, idxs.max())


    @overload
    def update_single(self, idx: int) -> None: ...
    @overload
    def update_single(self, idx: int, value: float) -> None: ...
    def update_single(self, idx: int, value: float = 0):
        self._support = max(self._support, idx)


    def sample(self, rng: np.random.Generator, n: int):
        if self._support == 1:
            return np.zeros(n, dtype=np.int64)

        return rng.integers(0, self._support, size=n)


    def stratified_sample(self, rng: np.random.Generator, n: int):
        return npu.stratified_sample_integers(rng, n, self._support)


    def probs(self, idxs: npt.ArrayLike):
        return np.full_like(idxs, fill_value=(1 / self._support), dtype=np.float64)
