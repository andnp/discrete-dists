import pickle

import numpy as np
import pytest

from discrete_dists import Rng


class TestDraws:
    def test_same_seed_gives_same_draws(self):
        """
        Generators built from the same seed must produce identical streams,
        so sampling is reproducible across runs.
        """
        a = Rng(7).random(5)
        b = Rng(7).random(5)

        assert np.array_equal(a, b)

    def test_draws_are_float64_in_unit_interval(self):
        """
        `random(n)` returns exactly n float64 values in [0, 1).
        """
        draws = Rng(11).random(1000)

        assert draws.shape == (1000,)
        assert draws.dtype == np.float64
        assert np.all((draws >= 0) & (draws < 1))


class TestSeedValidation:
    def test_negative_seed_raises(self):
        """
        Negative seeds have no unsigned representation and must be rejected.
        """
        with pytest.raises(ValueError, match="seed"):
            Rng(-1)

    def test_seed_above_u64_range_raises(self):
        """
        Seeds at or above 2**64 do not fit the generator state and must be rejected.
        """
        with pytest.raises(ValueError, match="seed"):
            Rng(2**64)


class TestPickling:
    def test_round_trip_continues_the_same_stream(self):
        """
        An unpickled generator must resume the exact stream of the original,
        so a checkpointed run replays the same draws.
        """
        rng = Rng(3)
        rng.random(10)

        restored = pickle.loads(pickle.dumps(rng))

        assert np.array_equal(restored.random(20), rng.random(20))
