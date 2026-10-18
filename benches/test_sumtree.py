import numpy as np

from discrete_dists import Rng
from discrete_dists.utils.SumTree import SumTree

# ----------------
# -- Benchmarks --
# ----------------
class TestBenchmarks:
    def test_sumtree_update(self, benchmark):
        tree = SumTree(100_000)
        rng = np.random.default_rng(0)
        idxs = np.arange(32)
        vals = rng.uniform(0, 2, size=32)

        def _inner(tree: SumTree, idxs, vals):
            tree.update(idxs, vals)

        benchmark(_inner, tree, idxs, vals)

    def test_sumtree_sample(self, benchmark):
        tree = SumTree(100_000)
        rng = np.random.default_rng(0)

        idxs = np.arange(10_000)
        vals = rng.uniform(0, 2, size=10_000)
        tree.update(idxs, vals)

        def _inner(tree: SumTree, sampler: Rng):
            tree.sample(sampler, 32)

        benchmark(_inner, tree, Rng(0))

    def test_sumtree_get_values(self, benchmark):
        tree = SumTree(100_000)
        rng = np.random.default_rng(0)
        tree.update(np.arange(100_000), rng.uniform(0, 2, size=100_000))
        idxs = rng.integers(0, 100_000, size=32)

        benchmark(tree.get_values, idxs)

    def test_sumtree_query(self, benchmark):
        tree = SumTree(100_000)
        rng = np.random.default_rng(0)
        tree.update(np.arange(100_000), rng.uniform(0, 2, size=100_000))
        rs = rng.uniform(0, tree.total(), size=32)

        benchmark(tree.query, rs)
