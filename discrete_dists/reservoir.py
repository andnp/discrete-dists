import numpy as np
import numpy.typing as npt

from discrete_dists._elements import as_elements
from discrete_dists.rust import Rng
from discrete_dists.rust import WeightedReservoir as _RustWeightedReservoir


class WeightedReservoir:
    """
    Keeps at most `capacity` elements from a stream, each kept with a probability
    that grows with its weight. A streaming sampler, not a `Distribution`.
    `offer` reports which elements were admitted and which stored elements were
    evicted to make room. Elements are opaque labels: offering one that is already
    stored keeps both entries.
    """
    def __init__(self, capacity: int):
        self._inner = _RustWeightedReservoir(capacity)

    @property
    def capacity(self) -> int:
        return self._inner.capacity

    @property
    def elements(self) -> np.ndarray:
        return self._inner.elements

    def __len__(self) -> int:
        return len(self._inner)

    def clear(self) -> None:
        self._inner.clear()

    def offer(self, rng: Rng, elements, weights: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Offers `elements` in order. Returns `(admitted, evicted)`: a bool per element, and the stored elements that were dropped."""
        return self._inner.offer(
            rng,
            as_elements(elements),
            np.asarray(weights, dtype=np.float64),
        )
