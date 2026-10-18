import pickle

import numpy as np
import pytest

from discrete_dists import Rng, WeightedReservoir


def test_capacity_bounds_stored_elements_and_clear_empties_it():
    r = WeightedReservoir(3)

    r.offer(Rng(0), np.arange(20), np.ones(20))
    assert r.capacity == 3
    assert len(r) == 3
    assert r.elements.shape == (3,)

    r.clear()
    assert len(r) == 0


def test_admitted_and_evicted_match_stored_elements():
    r = WeightedReservoir(8)
    rng = Rng(1)

    for start in range(0, 500, 100):
        stream = np.arange(start, start + 100)
        before = set(r.elements.tolist())

        admitted, evicted = r.offer(rng, stream, 1.0 + stream % 5)

        stored = (before | set(stream[admitted].tolist())) - set(evicted.tolist())
        assert set(r.elements.tolist()) == stored
        assert len(r) == 8


def test_seeded_offers_are_reproducible():
    stream, weights = np.arange(300), 1.0 + np.arange(300) % 3
    a, b = WeightedReservoir(10), WeightedReservoir(10)

    admitted_a, evicted_a = a.offer(Rng(7), stream, weights)
    admitted_b, evicted_b = b.offer(Rng(7), stream, weights)

    np.testing.assert_array_equal(admitted_a, admitted_b)
    np.testing.assert_array_equal(evicted_a, evicted_b)
    np.testing.assert_array_equal(a.elements, b.elements)


@pytest.mark.parametrize("weights", [[1.0, -1.0], [1.0, np.nan], [1.0, np.inf]])
def test_invalid_weights_raise_value_error_and_change_nothing(weights):
    r = WeightedReservoir(2)

    with pytest.raises(ValueError):
        r.offer(Rng(0), [0, 1], weights)
    assert len(r) == 0


def test_length_mismatch_raises_value_error():
    r = WeightedReservoir(2)

    with pytest.raises(ValueError):
        r.offer(Rng(0), [0, 1], [1.0])


def test_pickle_round_trip_continues_the_same_stream():
    r = WeightedReservoir(4)
    r.offer(Rng(2), np.arange(50), 1.0 + np.arange(50) % 4)
    restored = pickle.loads(pickle.dumps(r))
    stream, weights = np.arange(50, 150), 1.0 + np.arange(100) % 4

    np.testing.assert_array_equal(restored.elements, r.elements)
    admitted_r, evicted_r = restored.offer(Rng(3), stream, weights)
    admitted, evicted = r.offer(Rng(3), stream, weights)

    np.testing.assert_array_equal(admitted_r, admitted)
    np.testing.assert_array_equal(evicted_r, evicted)
    np.testing.assert_array_equal(restored.elements, r.elements)


def test_empty_batch_changes_nothing():
    r = WeightedReservoir(3)
    r.offer(Rng(0), [1, 2], [1.0, 1.0])
    before = r.elements.copy()

    admitted, evicted = r.offer(Rng(0), [], [])

    assert admitted.dtype == np.bool_
    assert admitted.shape == (0,)
    assert evicted.shape == (0,)
    np.testing.assert_array_equal(r.elements, before)
