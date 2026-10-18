from __future__ import annotations
from abc import abstractmethod
import numpy as np
import logging
from discrete_dists.rust import Rng

logger = logging.getLogger('discrete_dists')


Support = tuple[int, int]


class Distribution:
    @abstractmethod
    def probs(self, elements: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def sample(self, rng: Rng, n: int) -> np.ndarray: ...

    @abstractmethod
    def stratified_sample(self, rng: Rng, n: int) -> np.ndarray: ...

    def isr(self, target: Distribution, elements: np.ndarray):
        """
        Compute importance sampling ratios from this distribution to `target`.

        For each queried element `x`, this returns `target.probs(x) / self.probs(x)`.
        The ratio is only well-defined when `self.probs(x) > 0`, which is the
        typical case when `elements` were sampled from `self`.

        If `self.probs(x) == 0` and `target.probs(x) > 0`, the returned ratio is
        `np.inf` to signal a support mismatch. If both probabilities are zero,
        the returned ratio is `np.nan` because the ratio is undefined.
        """
        elements = np.asarray(elements)
        source_probs = self.probs(elements)
        target_probs = target.probs(elements)

        ratios = np.full_like(target_probs, np.nan, dtype=np.float64)
        supported = source_probs > 0
        ratios[supported] = target_probs[supported] / source_probs[supported]
        ratios[~supported & (target_probs > 0)] = np.inf
        return ratios

    @abstractmethod
    def sample_without_replacement(self, rng: Rng, n: int) -> np.ndarray: ...


    @property
    def is_defunct(self) -> bool:
        return False
