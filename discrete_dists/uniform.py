import numpy as np
import numpy.typing as npt
import discrete_dists.utils.npu as npu
from discrete_dists.distribution import Distribution, Support


class Uniform(Distribution):
    """
    A uniform distribution defined over some support.
    If only an integer support is given, assume support
    is [0, support).
    """
    def __init__(self, support: Support | int):
        if isinstance(support, int):
            support = (0, support)

        self._support: Support = support


    # ------------------------
    # -- Changes to support --
    # ------------------------
    def update(self, elements: np.ndarray) -> None:
        """
        Update the support of the distribution to cover the
        given elements if these are outside the current support.
        """
        self._support = (
            min(self._support[0], elements.min()),
            max(self._support[1], elements.max() + 1)
        )


    def update_single(self, element: int) -> None:
        """
        Update the support of the distribution to cover the
        given element if it is outside the current support.
        """
        self._support = (
            min(self._support[0], element),
            max(self._support[1], element + 1),
        )


    def update_support(self, support: Support | int):
        """
        Update the support of the distribution to the new
        given support.
        """
        if isinstance(support, int):
            self._support = (0, support)
        else:
            self._support = support


    # --------------
    # -- Samplers --
    # --------------
    def sample(self, rng: np.random.Generator, n: int):
        """
        Sample `n` values from the support of the distribution
        with replacement.
        """
        size = self._support[1] - self._support[0]
        if size < 2:
            return self._support[0] + np.zeros(n, dtype=np.int64)

        return rng.integers(*self._support, size=n)


    def stratified_sample(self, rng: np.random.Generator, n: int):
        """
        Sample `n` evenly spaced values from the support
        of the distribution with replacement.
        """
        return npu.stratified_sample_integers(rng, n, *self._support)


    def probs(self, elements: npt.ArrayLike):
        """
        Get the probabilities of the given elements
        under the current distribution.
        """
        d = self._support[1] - self._support[0]
        return np.full_like(elements, fill_value=(1 / d), dtype=np.float64)


    @property
    def is_defunct(self) -> bool:
        return self._support[0] == self._support[1]
