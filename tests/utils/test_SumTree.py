import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

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
            v = rng.uniform(0, 1, size=100)
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

    def test_threads_with_separate_trees_stay_consistent(self):
        # Each thread owns its tree, so concurrent use on free-threaded Python
        # must not interfere between trees.
        def worker(seed: int) -> tuple[float, float]:
            rng = np.random.default_rng(seed)
            n = 10_000
            tree = SumTree(n)
            expected = np.ones(n)
            tree.update(np.arange(n), expected)

            for _ in range(300):
                idxs = rng.choice(n, size=50, replace=False)
                values = rng.uniform(0, 1, size=50)
                tree.update(idxs, values)
                expected[idxs] = values
                tree.sample(rng, 32)

            return tree.total(), expected.sum()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(worker, range(8)))

        for total, expected in results:
            assert total == pytest.approx(expected, rel=1e-9)

    def test_rejects_negative_index(self):
        tree = SumTree(4)

        with pytest.raises(IndexError, match="out of bounds"):
            tree.update([-1], [1.0])

    def test_rejects_out_of_bounds_index(self):
        tree = SumTree(4)

        with pytest.raises(IndexError, match="out of bounds"):
            tree.update([4], [1.0])

    def test_rejects_length_mismatch(self):
        tree = SumTree(5)

        with pytest.raises(ValueError, match="same length"):
            tree.update([0, 1, 2], [10.0])

    def test_get_value_rejects_out_of_bounds_index(self):
        tree = SumTree(2)

        with pytest.raises(IndexError, match="out of bounds"):
            tree.get_value(-1)

    def test_rejects_negative_weight(self):
        tree = SumTree(4)

        with pytest.raises(ValueError, match="nonnegative"):
            tree.update([1], [-1.0])

    def test_rejects_nan_weight(self):
        tree = SumTree(4)

        with pytest.raises(ValueError, match="finite"):
            tree.update([1], [np.nan])

    def test_rejects_inf_weight(self):
        tree = SumTree(4)

        with pytest.raises(ValueError, match="finite"):
            tree.update([1], [np.inf])

    def test_zero_sized_tree_rejects_sampling(self):
        tree = SumTree(0)
        rng = np.random.default_rng(0)

        assert tree.total() == 0

        with pytest.raises(ValueError, match="Cannot sample"):
            tree.sample(rng, 1)

        with pytest.raises(ValueError, match="Cannot sample"):
            tree.stratified_sample(rng, 1)

    def test_updates_do_not_accumulate_drift(self):
        """
        Repeated updates must leave leaves and totals exact. Propagating deltas
        drifts the sums, which can make the total negative and break sampling.
        """
        tree = SumTree(4)
        rng = np.random.default_rng(1)
        for _ in range(200_000):
            i = int(rng.integers(4))
            tree.update([i], [float(rng.uniform(0, 1e6))])

        tree.update(np.arange(4), np.array([1e-9, 0, 0, 0]))

        assert tree.get_value(0) == 1e-9
        assert tree.total() == 1e-9

        samples = tree.sample(np.random.default_rng(0), 100)
        assert np.all(samples == 0)

    def test_query_at_zero_skips_zero_weight_leaf(self):
        """
        A query of exactly 0.0 must land on the positive-weight leaf, never on a
        zero-weight leaf to its left.
        """
        tree = SumTree(2)
        tree.update([0, 1], [0.0, 1.0])

        assert tree.query(np.array([0.0])).tolist() == [1]

    def test_query_on_dense_grid_never_returns_zero_weight_leaf(self):
        """
        Boundaries between positive-weight leaves must not step into the
        zero-weight leaf between them, and results must stay in range.
        """
        tree = SumTree(3)
        tree.update([0, 1, 2], [1.0, 0.0, 1.0])

        result = tree.query(np.linspace(0, 2, 10001))

        assert not np.any(result == 1)
        assert np.all((result >= 0) & (result < 3))

    def test_query_only_returns_positive_weight_leaves_when_sparse(self):
        """
        With most weights zero, every sampled or boundary query must resolve to
        a leaf with positive weight.
        """
        rng = np.random.default_rng(3)
        weights = rng.uniform(0.1, 1.0, size=1000)
        weights[rng.random(1000) < 0.9] = 0.0
        tree = SumTree(1000)
        tree.update(np.arange(1000), weights)

        total = tree.total()
        values = np.concatenate(([0.0, total], rng.uniform(0, total, size=100_000)))
        result = tree.query(values)

        assert np.all(tree.get_values(result) > 0)

    def test_corrupt_state_raises_value_error(self):
        """
        Corrupt pickle state must surface as a ValueError from the Python side,
        not as an uncaught panic.
        """
        tree = SumTree(4)

        with pytest.raises(ValueError, match="invalid SumTree state"):
            tree.__setstate__({"st": b"\x00garbage"})
