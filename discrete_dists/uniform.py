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
    def update(self, elements: np.ndarray) -> None: ...
    @overload
    def update(self, elements: np.ndarray, values: np.ndarray) -> None: ...
    def update(self, elements: np.ndarray, values: np.ndarray | None = None):
        self._support = (
            min(self._support[0], elements.min()),
            max(self._support[1], elements.max())
        )


    @overload
    def update_single(self, element: int) -> None: ...
    @overload
    def update_single(self, element: int, value: float) -> None: ...
    def update_single(self, element: int, value: float = 0):
        self._support = (
            min(self._support[0], element),
            max(self._support[1], element),
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


    def probs(self, elements: npt.ArrayLike):
        d = self._support[1] - self._support[0]
        return np.full_like(elements, fill_value=(1 / d), dtype=np.float64)
