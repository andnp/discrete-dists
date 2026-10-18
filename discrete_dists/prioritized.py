import numpy as np

from discrete_dists._elements import as_elements
from discrete_dists.distribution import Distribution
from discrete_dists.rust import Rng
from discrete_dists.rust import Prioritized as _RustPrioritized


class Prioritized(Distribution):
    """
    Elements are sampled with probability proportional to `priority ** alpha`.
    Importance weights correct the bias of that sampling; `beta = 1` corrects it fully.
    Priority 0 removes an element from the distribution.
    """
    def __init__(self, size: int, alpha: float):
        super().__init__()

        self._inner = _RustPrioritized(size, alpha)

    @property
    def size(self) -> int:
        return self._inner.size

    @property
    def alpha(self) -> float:
        return self._inner.alpha

    @property
    def max_priority(self) -> float | None:
        """Largest priority currently set, or None when no element has one."""
        return self._inner.max_priority

    @property
    def is_defunct(self) -> bool:
        return self._inner.is_defunct

    def update(self, elements, priorities) -> None:
        self._inner.update(
            as_elements(elements),
            np.asarray(priorities, dtype=np.float64),
        )

    def probs(self, elements) -> np.ndarray:
        return self._inner.probs(as_elements(elements))

    def importance_weights(self, elements, beta: float) -> np.ndarray:
        return self._inner.importance_weights(as_elements(elements), float(beta))

    def sample(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.sample(rng, n)

    def stratified_sample(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.stratified_sample(rng, n)

    def sample_without_replacement(self, rng: Rng, n: int) -> np.ndarray:
        return self._inner.sample_without_replacement(rng, n)

    def sample_with_weights(
        self, rng: Rng, n: int, beta: float, stratified: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draws `n` elements and returns them with their importance weights."""
        return self._inner.sample_with_weights(rng, n, float(beta), stratified)
