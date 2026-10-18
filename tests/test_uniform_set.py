import pickle

import numpy as np
import pytest

from discrete_dists import Rng, UniformSet


def test_insert_remove_and_membership_track_elements():
    u = UniformSet(10)
    u.insert([1, 3, 7])
    u.insert([3])

    assert len(u) == 3
    assert 3 in u
    assert 4 not in u

    u.remove([3, 5])

    assert len(u) == 2
    assert 3 not in u
    assert u.size == 10


def test_sampling_returns_only_members():
    u = UniformSet(20)
    u.insert([2, 5, 9, 11])

    for draws in (
        u.sample(Rng(0), 500),
        u.stratified_sample(Rng(0), 500),
        u.sample_without_replacement(Rng(0), 4),
    ):
        assert set(draws.tolist()) <= {2, 5, 9, 11}

    assert sorted(u.sample_without_replacement(Rng(1), 4).tolist()) == [2, 5, 9, 11]


def test_seeded_sampling_is_reproducible():
    u = UniformSet(20)
    u.insert([2, 5, 9, 11])

    np.testing.assert_array_equal(u.sample(Rng(7), 50), u.sample(Rng(7), 50))


def test_sampling_empty_set_raises_value_error():
    u = UniformSet(4)
    assert u.is_defunct

    with pytest.raises(ValueError):
        u.sample(Rng(0), 3)


def test_out_of_range_element_raises_index_error():
    u = UniformSet(10)

    with pytest.raises(IndexError):
        u.insert([10])
    with pytest.raises(IndexError):
        u.remove([-1])


def test_pickle_round_trip_preserves_membership_and_draws():
    u = UniformSet(20)
    u.insert([2, 5, 9])
    restored = pickle.loads(pickle.dumps(u))

    assert len(restored) == 3
    assert 5 in restored
    np.testing.assert_array_equal(
        restored.sample_without_replacement(Rng(4), 3),
        u.sample_without_replacement(Rng(4), 3),
    )
