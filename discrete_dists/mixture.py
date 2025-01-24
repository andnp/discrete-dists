from dataclasses import dataclass
from typing import Sequence

import numpy as np

from discrete_dists.distribution import Distribution

@dataclass
class SubDistribution:
    d: Distribution
    p: float


class MixtureDistribution(Distribution):
    def __init__(self, dists: Sequence[SubDistribution]):
        super().__init__()

        self._dims = len(dists)

        self.dists = [sub.d for sub in dists]
        self._weights = np.array([sub.p for sub in dists])

        assert np.isclose(self._weights.sum(), 1)


    def probs(self, elements: np.ndarray):
        sub = np.array([d.probs(elements) for d in self.dists])
        p = self._weights.dot(sub)
        return p

    def sample(self, rng: np.random.Generator, n: int):
        out = np.empty(n, dtype=np.int64)
        subs = rng.choice(len(self.dists), size=n, replace=True, p=self._weights)
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
        out = np.empty(n, dtype=np.int64)
        subs = rng.choice(len(self.dists), size=n, replace=True, p=self._weights)
        elements, counts = np.unique(subs, return_counts=True)

        total = 0
        for element, count in zip(elements, counts, strict=True):
            d = self.dists[int(element)]
            next_t = total + count
            out[total:next_t] = d.stratified_sample(rng, count)
            total = next_t

        rng.shuffle(out)
        return out

    def update(self, elements: np.ndarray, values: np.ndarray):
        for d in self.dists:
            d.update(elements, values)

    def update_single(self, element: int, value: float):
        for d in self.dists:
            d.update_single(element, value)
