import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from discrete_dists.distribution import Distribution
from discrete_dists.rust import Rng
from discrete_dists.rust import WindowMixture as _RustWindowMixture


@dataclass(frozen=True)
class Window:
    """
    Elements whose age (`cursor - stamp`) lies in `[min_age, max_age)`.
    With `half_life` set, an element's weight decays as `0.5 ** (age / half_life)`.
    """
    min_age: float = 0.0
    max_age: float = math.inf
    p: float = 1.0
    half_life: float | None = None


class WindowMixture(Distribution):
    """
    Weights elements by how long ago they were stamped.

    Stamps and the cursor live on any monotone axis (seconds, steps, ...).
    Advancing the cursor moves elements between windows in Rust.
    """
    def __init__(self, size: int, windows: Sequence[Window], cursor: float = 0.0):
        super().__init__()

        self._inner = _RustWindowMixture(
            size,
            [(w.min_age, w.max_age, w.half_life, w.p) for w in windows],
            cursor,
        )

    @property
    def cursor(self) -> float:
        return self._inner.cursor

    @property
    def counts(self) -> list[int]:
        """Number of elements currently in each window."""
        return self._inner.counts

    @property
    def is_defunct(self) -> bool:
        return self._inner.is_defunct

    def insert(self, elements, stamps, weights) -> None:
        self._inner.insert(
            np.asarray(elements, dtype=np.int64),
            np.asarray(stamps, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
        )

    def remove(self, elements) -> None:
        self._inner.remove(np.asarray(elements, dtype=np.int64))

    def advance(self, to: float) -> None:
        self._inner.advance(float(to))

    def probs(self, elements) -> np.ndarray:
        return self._inner.probs(np.asarray(elements, dtype=np.int64))

    def sample(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.sample(rng, n)

    def stratified_sample(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.stratified_sample(rng, n)

    def sample_without_replacement(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.sample_without_replacement(rng, n)
