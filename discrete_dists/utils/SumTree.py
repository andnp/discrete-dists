import numpy as np
from typing import Iterable
import discrete_dists.rust as ru

class SumTree(ru.SumTree):
    def __new__(
        cls,
        size: int | None = None,
    ):
        args = [size]
        if args[0] is None:
            args = []

        return super().__new__(cls, *args)

    def __init__(self, size: int):
        super().__init__()

    def update(self, idxs: Iterable[int], values: Iterable[float]):
        a_idxs = np.asarray(idxs, dtype=np.int64)
        a_values = np.asarray(values, dtype=np.float64)

        super().update(a_idxs, a_values)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        t = self.total()
        assert t > 0, "Cannot sample when the tree is empty or contains negative values"

        rs = rng.uniform(0, t, size=n)
        return self.query(rs)

    def stratified_sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        t = self.total()
        assert t > 0, "Cannot sample when the tree is empty or contains negative values"

        buckets = np.linspace(0., 1., n + 1)
        values = np.asarray([
            rng.uniform(buckets[i], buckets[i + 1]) for i in range(n)
        ])

        return self.query(t * values)

    def __getstate__(self):
        return {
            'st': super().__getstate__()
        }

    def __setstate__(self, state):
        super().__setstate__(state['st'])
