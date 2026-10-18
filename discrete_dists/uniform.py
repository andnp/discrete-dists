import numpy as np
import numpy.typing as npt
from discrete_dists.distribution import Distribution, Support
from discrete_dists.rust import (
    Rng,
    uniform_sample,
    uniform_sample_without_replacement,
    uniform_stratified_sample,
)


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
    def sample(self, rng: Rng, n: int):
        """
        Sample `n` values from the support of the distribution
        with replacement.
        """
        return uniform_sample(rng, self._support[0], self._support[1], n)


    def stratified_sample(self, rng: Rng, n: int):
        """
        Sample `n` evenly spaced values from the support
        of the distribution with replacement.
        """
        return uniform_stratified_sample(rng, self._support[0], self._support[1], n)


    def sample_without_replacement(self, rng: Rng, n: int) -> np.ndarray:
        """
        Sample exactly `n` distinct values from the support.
        """
        return uniform_sample_without_replacement(rng, self._support[0], self._support[1], n)


    def probs(self, elements: npt.ArrayLike):
        """
        Get the probabilities of the given elements
        under the current distribution.
        """
        elements = np.asarray(elements)
        d = self._support[1] - self._support[0]
        if d == 0:
            return np.zeros_like(elements, dtype=np.float64)

        probs = np.zeros_like(elements, dtype=np.float64)
        in_support = (elements >= self._support[0]) & (elements < self._support[1])
        probs[in_support] = 1 / d
        return probs


    @property
    def is_defunct(self) -> bool:
        return self._support[0] == self._support[1]
