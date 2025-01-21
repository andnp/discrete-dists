from __future__ import annotations
import numpy as np
import logging

from discrete_dists.utils.SumTree import SumTree

logger = logging.getLogger('discrete_dists')

class Distribution:
    def __init__(self, support: int):
        self._support = support
        self._tree: SumTree | None = None
        self._dim: int | None = None

        self._weights: np.ndarray | None = None

    @property
    def tree(self):
        assert self._tree is not None
        return self._tree

    @property
    def dim(self):
        assert self._dim is not None
        return self._dim

    @property
    def weights(self):
        assert self._weights is not None
        return self._weights

    def init(
        self,
        memory: SumTree | None = None,
        dim: int | None = None,
    ):
        if memory is None:
            assert self._size is not None
            memory = SumTree(self._size, 1)
            dim = 0

        assert dim is not None
        self._tree = memory
        self._dim = dim
        self._size = memory.size
        self._weights = np.zeros(memory.dims)
        self._weights[self._dim] = 1

    # ---------------
    # -- Accessing --
    # ---------------
    def probs(self, idxs: np.ndarray) -> np.ndarray:
        idxs = np.asarray(idxs)

        t = self.tree.dim_total(self.dim)
        if t == 0:
            return np.zeros(len(idxs))

        v = self.tree.get_values(self.dim, idxs)
        return v / t


    def isr(self, target: Distribution, idxs: np.ndarray):
        return target.probs(idxs) / self.probs(idxs)

    # --------------
    # -- Sampling --
    # --------------
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.tree.sample(rng, n, self.weights)

    def stratified_sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.tree.stratified_sample(rng, n, self.weights)

    def sample_without_replacement(
        self,
        rng: np.random.Generator,
        n: int,
        attempts: int = 25,
    ) -> np.ndarray:
        idxs = self.sample(rng, n)

        # fastpath for the common case that the first sample is already unique
        uniq = set(idxs)
        if len(uniq) == n:
            return idxs

        for _ in range(attempts):
            needed = n - len(uniq)
            sub = self.sample(rng, 2 * needed)
            uniq |= set(sub)

            if len(uniq) >= n:
                break

        if len(uniq) < n:
            logger.warning(f"Failed to get <{n}> required unique samples. Got <{len(uniq)}>")

        cutoff = min(n, len(uniq))
        out = np.array(uniq, dtype=np.int64)[:cutoff]
        return out
