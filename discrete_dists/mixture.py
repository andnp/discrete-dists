from dataclasses import dataclass
from typing import Sequence

import numpy as np

from discrete_dists.distribution import Distribution
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
        sub = np.array([d.probs(elements) for d in self.dists])
        p = self._weights.dot(sub)
        return p

    def sample(self, rng: np.random.Generator, n: int):
        """
        Sample `n` values from the mixture distribution,
        partitioning these `n` values over the various
        sub-distributions according to their respective
        probabilities.
        """
        out = np.empty(n, dtype=np.int64)
        idxs, weights = self.filter_defunct()
        subs = rng.choice(idxs, size=n, replace=True, p=weights)
        elements, counts = np.unique(subs, return_counts=True)

        total = 0
        for element, count in zip(elements, counts, strict=True):
            d = self.dists[int(element)]
            next_t = total + count
            out[total:next_t] = d.sample(rng, count)
            total = next_t

        rng.shuffle(out)
        return out

    def stratified_sample(self, rng: np.random.Generator, n: int):
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
        subs = rng.choice(idxs, size=n, replace=True, p=weights)
        elements, counts = np.unique(subs, return_counts=True)

        total = 0
        for element, count in zip(elements, counts, strict=True):
            d = self.dists[int(element)]
            next_t = total + count
            out[total:next_t] = d.stratified_sample(rng, count)
            total = next_t

        rng.shuffle(out)
        return out


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
        ])

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
