import numpy as np
import numpy.typing as npt

from discrete_dists._elements import as_elements
from discrete_dists.distribution import Distribution, Support
from discrete_dists.rust import Rng
from discrete_dists.rust import Alias as _RustAlias


class Alias(Distribution):
    """
    A fixed discrete distribution over `[lo, lo + len(weights))`, where each
    element is sampled with probability proportional to its weight.
    Draws take constant time after an O(n) build. The weights cannot change;
    build a new `Alias` to change them.
    """
    def __init__(self, weights: npt.ArrayLike, lo: int = 0):
        super().__init__()

        self._inner = _RustAlias(np.asarray(weights, dtype=np.float64), lo)

    @property
    def support(self) -> Support:
        return self._inner.support

    @property
    def is_defunct(self) -> bool:
        return self._inner.is_defunct

    def __len__(self) -> int:
        return len(self._inner)

    def probs(self, elements) -> np.ndarray:
        return self._inner.probs(as_elements(elements))

    def sample(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.sample(rng, n)

    def stratified_sample(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.stratified_sample(rng, n)

    def sample_without_replacement(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.sample_without_replacement(rng, n)
