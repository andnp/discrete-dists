from typing import overload
import numpy as np
import numpy.typing as npt
import discrete_dists.utils.npu as npu
from discrete_dists.distribution import Distribution, Support


class Uniform(Distribution):
    def __init__(self, support: Support | int):
        if isinstance(support, int):
            support = (0, support)

        self._support: Support = support


    # ------------------------
    # -- Changes to support --
    # ------------------------
    @overload
    def update(self, idxs: np.ndarray) -> None: ...
    @overload
    def update(self, idxs: np.ndarray, values: np.ndarray) -> None: ...
    def update(self, idxs: np.ndarray, values: np.ndarray | None = None):
        self._support = (
            min(self._support[0], idxs.min()),
            max(self._support[1], idxs.max())
        )


    @overload
    def update_single(self, idx: int) -> None: ...
    @overload
    def update_single(self, idx: int, value: float) -> None: ...
    def update_single(self, idx: int, value: float = 0):
        self._support = (
            min(self._support[0], idx),
            max(self._support[1], idx),
        )


    def update_support(self, support: Support | int):
        if isinstance(support, int):
            self._support = (0, support)
        else:
            self._support = support


    # --------------
    # -- Samplers --
    # --------------
    def sample(self, rng: np.random.Generator, n: int):
        if self._support == (0, 1):
            return np.zeros(n, dtype=np.int64)

        return rng.integers(*self._support, size=n)


    def stratified_sample(self, rng: np.random.Generator, n: int):
        return npu.stratified_sample_integers(rng, n, *self._support)


    def probs(self, idxs: npt.ArrayLike):
        d = self._support[1] - self._support[0]
        return np.full_like(idxs, fill_value=(1 / d), dtype=np.float64)
