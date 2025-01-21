import pickle
import numpy as np

from discrete_dists.utils.SumTree import SumTree

class TestSumTree:
    def test_can_add_stuff(self):
        # human readable test
        v = np.array([1, 2, 3, 4, 5])
        truth = 15
        tree = SumTree(500)
        tree.update([0, 5, 8, 102, 358], v)

        assert tree.total() == truth

        # fuzz test
        tree = SumTree(100)
        rng = np.random.default_rng(0)
        for _ in range(100):
            v = rng.standard_normal(100)
            truth = v.sum()
            tree.update(np.arange(100), v)

            assert np.isclose(tree.total(), truth)


    def test_can_sample(self):
        tree = SumTree(50)
        tree.update(np.arange(50), np.ones(50))

        assert tree.total() == 50

        rng = np.random.default_rng(0)
        samples = tree.sample(rng, 10000)

        u, c = np.unique(samples, return_counts=True)
        assert np.all(u == np.arange(50))
        assert np.all(
            (c >= 150) & (c <= 250)
        )

    def test_can_sample_proportionally(self):
        tree = SumTree(10)
        tree.update(np.arange(10), [2**i for i in range(10)])

        # NOTE: it takes a shockingly high number of samples for the proportions
        # to converge even within a single decimal point..
        rng = np.random.default_rng(22)
        samples = tree.sample(rng, 10000000)

        u, c = np.unique(samples, return_counts=True)
        assert np.all(u == np.arange(10))

        for i in range(1, 10):
            assert np.isclose(c[i] / c[i - 1], 2, atol=0.1)

    def test_pickleable(self):
        tree = SumTree(123)
        tree.update(np.arange(123), np.arange(123))

        byt = pickle.dumps(tree)
        tree2 = pickle.loads(byt)

        assert np.all(
            tree.total() == tree2.total()
        )

        tree.update([2], [22])
        tree2.update([2], [22])

        assert np.all(
            tree.total() == tree2.total()
        )

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
