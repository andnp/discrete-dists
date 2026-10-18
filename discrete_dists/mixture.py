from dataclasses import dataclass
from typing import Sequence

import numpy as np

from discrete_dists.distribution import Distribution
from discrete_dists.rust import Rng
from discrete_dists.proportional import Proportional
from discrete_dists.uniform import Uniform

@dataclass
class SubDistribution:
    """
    A pair joining a distribution with the ratio of
    contribution of this distribution to the mixture.
    """
    d: Distribution
    p: float


def _draw_parts(rng: Rng, idxs: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    cdf = np.cumsum(weights)
    slots = np.searchsorted(cdf, rng.random(n), side="right")
    return idxs[np.minimum(slots, len(idxs) - 1)]


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

        assert np.isclose(self._weights.sum(), 1)


    def probs(self, elements: np.ndarray):
        """
        Get the probabilities of the given elements
        under the current distribution.
        """
        elements = np.asarray(elements)
        idxs, weights = self.filter_defunct()

        if len(idxs) == 0:
            return np.zeros(len(elements), dtype=np.float64)

        sub = np.array([self.dists[int(i)].probs(elements) for i in idxs])
        p = weights.dot(sub)
        return p

    def sample(self, rng: Rng, n: int):
        """
        Sample `n` values from the mixture distribution,
        partitioning these `n` values over the various
        sub-distributions according to their respective
        probabilities.
        """
        out = np.empty(n, dtype=np.int64)
        idxs, weights = self.filter_defunct()

        if len(idxs) == 0:
            if n == 0:
                return out
            raise ValueError("cannot sample from an all-defunct mixture distribution")

        subs = _draw_parts(rng, idxs, weights, n)
        elements, counts = np.unique(subs, return_counts=True)

        total = 0
        for element, count in zip(elements, counts, strict=True):
            d = self.dists[int(element)]
            next_t = total + count
            out[total:next_t] = d.sample(rng, count)
            total = next_t

        return out[np.argsort(rng.random(n))]

    def stratified_sample(self, rng: Rng, n: int):
        """
        Sample `n` values from the mixture distribution,
        partitioning these `n` values over the various
        sub-distributions according to their respective
        probabilities.

        The `m < n` values sampled from each sub-distribution
        will be evenly spaced within that distribution.
        """
        out = np.empty(n, dtype=np.int64)
        idxs, weights = self.filter_defunct()

        if len(idxs) == 0:
            if n == 0:
                return out
            raise ValueError("cannot sample from an all-defunct mixture distribution")

        subs = _draw_parts(rng, idxs, weights, n)
        elements, counts = np.unique(subs, return_counts=True)

        total = 0
        for element, count in zip(elements, counts, strict=True):
            d = self.dists[int(element)]
            next_t = total + count
            out[total:next_t] = d.stratified_sample(rng, count)
            total = next_t

        return out[np.argsort(rng.random(n))]


    @property
    def is_defunct(self) -> bool:
        return all(d.is_defunct for d in self.dists)


    def filter_defunct(self):
        """
        Remove any defunct distributions from the mixture, where
        a defunct distribution is defined as having zero support.
        """

        # fastpath for the common case that there are no defunct distributions
        if all(not d.is_defunct for d in self.dists):
            return np.arange(len(self.dists)), self._weights


        dist_idxs = np.array([
            i for i, d in enumerate(self.dists)
            if not d.is_defunct
        ], dtype=np.int64)

        if len(dist_idxs) == 0:
            return dist_idxs, np.array([], dtype=np.float64)

        reweighted = self._weights[dist_idxs]
        reweighted /= reweighted.sum()
        return dist_idxs, reweighted

    def update(self, elements: np.ndarray, values: np.ndarray):
        """
        Update the the proportion values for a given set
        of elements. This changes the shape of the distribution.
        """
        for d in self.dists:
            if isinstance(d, Proportional):
                d.update(elements, values)
            elif isinstance(d, Uniform):
                d.update(elements)
