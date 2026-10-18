import numpy as np

from discrete_dists.distribution import Distribution
from discrete_dists.rust import Rng
from discrete_dists.rust import UniformSet as _RustUniformSet


class UniformSet(Distribution):
    """
    A uniform distribution over a changing subset of `range(size)`.
    Inserting and removing elements costs constant time per element.
    """
    def __init__(self, size: int):
        super().__init__()

        self._inner = _RustUniformSet(size)

    @property
    def size(self) -> int:
        return self._inner.size

    @property
    def is_defunct(self) -> bool:
        return self._inner.is_defunct

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, element: int) -> bool:
        return self._inner.contains(int(element))

    def insert(self, elements) -> None:
        self._inner.insert(np.asarray(elements, dtype=np.int64))

    def remove(self, elements) -> None:
        self._inner.remove(np.asarray(elements, dtype=np.int64))

    def probs(self, elements) -> np.ndarray:
        return self._inner.probs(np.asarray(elements, dtype=np.int64))

    def sample(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.sample(rng, n)

    def stratified_sample(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.stratified_sample(rng, n)

    def sample_without_replacement(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.sample_without_replacement(rng, n)
