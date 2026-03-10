import numpy as np
from discrete_dists.distribution import Distribution, Support
from discrete_dists.utils.SumTree import SumTree

class Proportional(Distribution):
    """
    A distribution defined over some support
    [lo, hi) where elements within the support
    are sampled proportional to some value.

    By default, the ratios are initially set to 0,
    which inhibits sampling. In order to sample,
    this distribution must first receive updated
    ratios:
    ```python
    p = Proportional(10)
    p.update(
      # elements on the support to update
      elements=np.arange(5),
      # values with which to sample proportionally to
      values=np.array([1, 2, 1, 1, 1]),
    )
    ```
    """
    def __init__(self, support: Support | int):
        if isinstance(support, int):
            support = (0, support)

        self._support = support
        rang = support[1] - support[0]
        self.tree = SumTree(rang)

    # ---------------
    # -- Accessing --
    # ---------------
    def probs(self, elements: np.ndarray) -> np.ndarray:
        """
        Get the probabilities of the given elements
        in the distribution.
        """
        elements = np.asarray(elements)
        probs = np.zeros(len(elements), dtype=np.float64)

        in_support = (
            (elements >= self._support[0]) &
            (elements < self._support[1])
        )

        t = self.tree.total()
        if t == 0 or not np.any(in_support):
            return probs

        shifted = elements[in_support] - self._support[0]
        v = self.tree.get_values(shifted)
        probs[in_support] = v / t
        return probs


    @property
    def is_defunct(self) -> bool:
        return self.tree.total() == 0


    # --------------
    # -- Sampling --
    # --------------
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """
        Sample `n` values from the distribution. Return
        will be a np.array of integers.
        """
        return self.tree.sample(rng, n) + self._support[0]

    def stratified_sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """
        Sample `n` evenly spaced values from the distribution.
        Return will be a np.array of integers.
        """
        return self.tree.stratified_sample(rng, n) + self._support[0]

    def sample_without_replacement(
        self,
        rng: np.random.Generator,
        n: int,
        attempts: int = 25,
    ) -> np.ndarray:
        if n < 0:
            raise ValueError(f"n must be nonnegative, got {n}")

        width = self._support[1] - self._support[0]
        if width == 0:
            if n == 0:
                return np.array([], dtype=np.int64)
            raise ValueError("cannot sample from an empty proportional distribution")

        offsets = np.arange(width, dtype=np.int64)
        values = self.tree.get_values(offsets)
        active = values > 0

        if n > active.sum():
            raise ValueError(
                f"cannot sample {n} unique elements from {active.sum()} positive-mass elements"
            )

        support = offsets[active] + self._support[0]
        probs = values[active] / values[active].sum()
        return rng.choice(support, size=n, replace=False, p=probs)

    # --------------
    # -- Updating --
    # --------------
    def update(self, elements: np.ndarray, values: np.ndarray):
        """
        Update the the proportion values for a given set
        of elements. This changes the shape of the distribution.
        """
        elements = elements - self._support[0]
        self.tree.update(elements, values)

    def update_single(self, element: int, value: float):
        """
        Update the the proportion values for a given single
        element. This changes the shape of the distribution.
        """
        element = element - self._support[0]
        self.tree.update_single(element, value)

    def update_support(self, support: Support | int):
        """
        Shift the entire distribution to be over a new support.

        If the new support has the same width as the current support, the
        existing relative weights are shifted to the new interval.

        If the new support is wider and fully contains the current support,
        the existing absolute element weights are preserved and newly exposed
        elements start with zero mass.
        """
        if isinstance(support, int):
            support = (0, support)

        old_support = self._support
        old_width = old_support[1] - old_support[0]
        new_width = support[1] - support[0]

        if new_width < old_width:
            raise ValueError(
                f"cannot shrink proportional support from width {old_width} to {new_width}"
            )

        if new_width == old_width:
            self._support = support
            return

        if support[0] > old_support[0] or support[1] < old_support[1]:
            raise ValueError(
                "widened support must contain the existing support in order to preserve weights"
            )

        values = self.tree.get_values(np.arange(old_width, dtype=np.int64))
        new_tree = SumTree(new_width)
        offset = old_support[0] - support[0]
        new_tree.update(offset + np.arange(old_width, dtype=np.int64), values)

        self._support = support
        self.tree = new_tree
