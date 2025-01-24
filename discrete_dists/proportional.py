import numpy as np
from discrete_dists.distribution import Distribution, Support
from discrete_dists.utils.SumTree import SumTree

class Proportional(Distribution):
    def __init__(self, support: Support | int):
        if isinstance(support, int):
            support = (0, support)

        self._support = support
        rang = support[1] - support[0]
        self.tree = SumTree(rang)

    # ---------------
    # -- Accessing --
    # ---------------
    def probs(self, idxs: np.ndarray) -> np.ndarray:
        idxs = np.asarray(idxs)
        idxs = idxs - self._support[0]

        t = self.tree.total()
        if t == 0:
            return np.zeros(len(idxs))

        v = self.tree.get_values(idxs)
        return v / t


    # --------------
    # -- Sampling --
    # --------------
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.tree.sample(rng, n) + self._support[0]

    def stratified_sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.tree.stratified_sample(rng, n) + self._support[0]

    # --------------
    # -- Updating --
    # --------------
    def update(self, idxs: np.ndarray, values: np.ndarray):
        idxs = idxs - self._support[0]
        self.tree.update(idxs, values)

    def update_single(self, idx: int, value: float):
        idx = idx - self._support[0]
        self.tree.update_single(idx, value)

    def update_support(self, support: Support | int):
        if isinstance(support, int):
            self._support = (0, support)
        else:
            self._support = support
