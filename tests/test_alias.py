import pickle

import numpy as np
import pytest

from discrete_dists import Alias, Rng


def test_probs_are_normalized_weights():
    a = Alias([1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(a.probs(np.arange(4)), [0.1, 0.2, 0.3, 0.4])


def test_seeded_sampling_is_reproducible():
    a = Alias([1.0, 2.0, 3.0])

    np.testing.assert_array_equal(a.sample(Rng(7), 50), a.sample(Rng(7), 50))


def test_zero_weight_elements_are_never_sampled():
    a = Alias([0.0, 3.0, 0.0, 1.0])

    for draws in (
        a.sample(Rng(0), 5000),
        a.stratified_sample(Rng(0), 5000),
        a.sample_without_replacement(Rng(0), 2),
    ):
        assert set(draws.tolist()) <= {1, 3}


def test_invalid_weights_raise_value_error():
    with pytest.raises(ValueError):
        Alias([1.0, -1.0])
    with pytest.raises(ValueError):
        Alias([1.0, np.nan])


def test_pickle_round_trip_preserves_probs_and_draws():
    a = Alias([1.0, 0.0, 2.0], lo=5)
    restored = pickle.loads(pickle.dumps(a))
    elements = np.arange(4, 9)

    np.testing.assert_allclose(restored.probs(elements), a.probs(elements))
    np.testing.assert_array_equal(
        restored.sample_without_replacement(Rng(4), 2),
        a.sample_without_replacement(Rng(4), 2),
    )


def test_offset_support_is_respected():
    a = Alias([1.0, 3.0], lo=10)

    assert a.support == (10, 12)
    np.testing.assert_allclose(a.probs(np.array([9, 10, 11, 12])), [0.0, 0.25, 0.75, 0.0])
    assert set(a.sample(Rng(1), 500).tolist()) <= {10, 11}
