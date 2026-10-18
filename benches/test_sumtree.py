import numpy as np

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

        def _inner(tree: SumTree, rng):
            tree.sample(rng, 32)

        benchmark(_inner, tree, rng)
