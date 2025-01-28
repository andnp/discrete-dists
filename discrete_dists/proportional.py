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
        elements = elements - self._support[0]

        t = self.tree.total()
        if t == 0:
            return np.zeros(len(elements))

        v = self.tree.get_values(elements)
        return v / t


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
        It is undefined behavior to change the width of the support.
        """
        if isinstance(support, int):
            self._support = (0, support)
        else:
            self._support = support
