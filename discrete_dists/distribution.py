from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger('discrete_dists')


class Distribution(ABC):
    @abstractmethod
    def probs(self, idxs: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def update(self, idxs: np.ndarray, values: np.ndarray) -> None: ...

    @abstractmethod
    def update_single(self, idx: int, value: float) -> None: ...

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...

    @abstractmethod
    def stratified_sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...

    def isr(self, target: Distribution, idxs: np.ndarray):
        return target.probs(idxs) / self.probs(idxs)

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
