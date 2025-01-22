import numpy as np
from discrete_dists.distribution import Distribution
from discrete_dists.utils.SumTree import SumTree

class Proportional(Distribution):
    def __init__(self, support: int):
        self._support = support
        self.tree = SumTree(support)

    # ---------------
    # -- Accessing --
    # ---------------
    def probs(self, idxs: np.ndarray) -> np.ndarray:
        idxs = np.asarray(idxs)

        t = self.tree.total()
        if t == 0:
            return np.zeros(len(idxs))

        v = self.tree.get_values(idxs)
        return v / t


    # --------------
    # -- Sampling --
    # --------------
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.tree.sample(rng, n)

    def stratified_sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.tree.stratified_sample(rng, n)

    # --------------
    # -- Updating --
    # --------------
    def update(self, idxs: np.ndarray, values: np.ndarray):
        self.tree.update(idxs, values)

    def update_single(self, idx: int, value: float):
        self.tree.update_single(idx, value)
