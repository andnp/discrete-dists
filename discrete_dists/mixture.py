from dataclasses import dataclass
from typing import Sequence

import numpy as np

from discrete_dists._elements import as_elements
from discrete_dists.alias import Alias
from discrete_dists.distribution import Distribution
from discrete_dists.proportional import Proportional
from discrete_dists.rust import Alias as _RustAlias
from discrete_dists.rust import (
    Rng,
    mixture_probs,
    mixture_sample,
    mixture_sample_without_replacement,
    mixture_stratified_sample,
)
from discrete_dists.rust import UniformSet as _RustUniformSet
from discrete_dists.uniform import Uniform
from discrete_dists.uniform_set import UniformSet
from discrete_dists.utils.SumTree import SumTree

_Part = tuple[str, tuple[int, int] | tuple[int, SumTree] | _RustUniformSet | _RustAlias, float]


@dataclass
class SubDistribution:
    """
    A pair joining a distribution with the ratio of
    contribution of this distribution to the mixture.
    """
    d: Distribution
    p: float


def _unsupported(d: Distribution) -> TypeError:
    return TypeError(
        "MixtureDistribution supports only Uniform, UniformSet, Alias and Proportional sub-distributions, "
        f"got {type(d).__name__}"
    )


class MixtureDistribution(Distribution):
    """
    A mixture over an arbitrary number of sub-distributions.
    Will sample from sub-distributions according to their
    respective probabilities.
    """
    def __init__(self, dists: Sequence[SubDistribution]):
        super().__init__()

        self._dims = len(dists)

        self.dists = [sub.d for sub in dists]
        self._weights = np.array([sub.p for sub in dists])

        for d in self.dists:
            if not isinstance(d, (Proportional, Uniform, UniformSet, Alias)):
                raise _unsupported(d)

        assert np.isclose(self._weights.sum(), 1)


    def probs(self, elements: np.ndarray):
        """
        Get the probabilities of the given elements
        under the current distribution.
        """
        return mixture_probs(self._parts(), as_elements(elements))

    def sample(self, rng: Rng, n: int):
        """
        Sample `n` values from the mixture distribution,
        partitioning these `n` values over the various
        sub-distributions according to their respective
        probabilities.
        """
        return mixture_sample(rng, self._parts(), n)

    def stratified_sample(self, rng: Rng, n: int):
        """
        Sample `n` values from the mixture distribution,
        partitioning these `n` values over the various
        sub-distributions according to their respective
        probabilities.

        The `m < n` values sampled from each sub-distribution
        will be evenly spaced within that distribution.
        """
        return mixture_stratified_sample(rng, self._parts(), n)

    def sample_without_replacement(self, rng: Rng, n: int) -> np.ndarray:
        """
        Sample exactly `n` distinct values by rejection; raises
        `ValueError` if a fixed draw budget runs out first.
        """
        return mixture_sample_without_replacement(rng, self._parts(), n)


    @property
    def is_defunct(self) -> bool:
        return all(d.is_defunct for d in self.dists)


    def _parts(self) -> list[_Part]:
        parts: list[_Part] = []
        for d, w in zip(self.dists, self._weights):
            if isinstance(d, Proportional):
                parts.append(("proportional", (d._support[0], d.tree), float(w)))
            elif isinstance(d, Uniform):
                parts.append(("uniform", d._support, float(w)))
            elif isinstance(d, UniformSet):
                parts.append(("uniform_set", d._inner, float(w)))
            elif isinstance(d, Alias):
                parts.append(("alias", d._inner, float(w)))
            else:
                raise _unsupported(d)

        return parts

    def update(self, elements: np.ndarray, values: np.ndarray):
        """
        Update the the proportion values for a given set
        of elements. This changes the shape of the distribution.

        UniformSet and Alias parts have no updatable weights, so they are left unchanged.
        """
        for d in self.dists:
            if isinstance(d, Proportional):
                d.update(elements, values)
            elif isinstance(d, Uniform):
                d.update(elements)
