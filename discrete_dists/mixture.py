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


    def probs(self, idxs: np.ndarray):
        sub = np.array([d.probs(idxs) for d in self.dists])
        p = self._weights.dot(sub)
        return p

    def sample(self, rng: np.random.Generator, n: int):
        out = np.empty(n, dtype=np.int64)
        subs = rng.choice(len(self.dists), size=n, replace=True, p=self._weights)
        idxs, counts = np.unique(subs, return_counts=True)

        total = 0
        for idx, count in zip(idxs, counts, strict=True):
            d = self.dists[int(idx)]
            next_t = total + count
            out[total:next_t] = d.sample(rng, count)
            total = next_t

        rng.shuffle(out)
        return out

    def stratified_sample(self, rng: np.random.Generator, n: int):
        out = np.empty(n, dtype=np.int64)
        subs = rng.choice(len(self.dists), size=n, replace=True, p=self._weights)
        idxs, counts = np.unique(subs, return_counts=True)

        total = 0
        for idx, count in zip(idxs, counts, strict=True):
            d = self.dists[int(idx)]
            next_t = total + count
            out[total:next_t] = d.stratified_sample(rng, count)
            total = next_t

        rng.shuffle(out)
        return out

    def update(self, idxs: np.ndarray, values: np.ndarray):
        for d in self.dists:
            d.update(idxs, values)

    def update_single(self, idx: int, value: float):
        for d in self.dists:
            d.update_single(idx, value)
