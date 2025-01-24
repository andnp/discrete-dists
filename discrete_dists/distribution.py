from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger('discrete_dists')


Support = tuple[int, int]


class Distribution(ABC):
    @abstractmethod
    def probs(self, elements: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def update(self, elements: np.ndarray, values: np.ndarray) -> None: ...

    @abstractmethod
    def update_single(self, element: int, value: float) -> None: ...

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...

    @abstractmethod
    def stratified_sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...

    def isr(self, target: Distribution, elements: np.ndarray):
        return target.probs(elements) / self.probs(elements)

    def sample_without_replacement(
        self,
        rng: np.random.Generator,
        n: int,
        attempts: int = 25,
    ) -> np.ndarray:
        elements = self.sample(rng, n)

        # fastpath for the common case that the first sample is already unique
        uniq = set(elements)
        if len(uniq) == n:
            return elements

        for _ in range(attempts):
            needed = n - len(uniq)
            sub = self.sample(rng, 2 * needed)
            uniq |= set(sub)

            if len(uniq) >= n:
                break

        if len(uniq) < n:
            logger.warning(f"Failed to get <{n}> required unique samples. Got <{len(uniq)}>")

        cutoff = min(n, len(uniq))
        out = np.array(list(uniq), dtype=np.int64)[:cutoff]
        return out
