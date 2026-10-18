import pickle

import numpy as np
import pytest

from discrete_dists import Rng, Window, WindowMixture


def test_element_moves_to_next_window_when_cursor_crosses_boundary():
    m = WindowMixture(4, [Window(0, 10, p=0.5), Window(10, np.inf, p=0.5)])
    m.insert([0], [0.0], [1.0])

    m.advance(9.5)
    assert m.counts == [1, 0]

    m.advance(10.0)
    assert m.counts == [0, 1]


def test_decaying_probs_follow_half_life():
    m = WindowMixture(3, [Window(half_life=2.0)], cursor=5.0)
    m.insert([0, 1, 2], [0.0, 1.0, 3.0], [1.0, 2.0, 0.5])

    ages = 5.0 - np.array([0.0, 1.0, 3.0])
    expected = np.array([1.0, 2.0, 0.5]) * 0.5 ** (ages / 2.0)
    np.testing.assert_allclose(m.probs(np.array([0, 1, 2])), expected / expected.sum(), rtol=1e-12)


def test_probs_follow_window_shares_and_empty_window_hands_its_share_over():
    m = WindowMixture(4, [Window(0, 10, p=0.75), Window(10, np.inf, p=0.25)])
    m.insert([0], [0.0], [1.0])
    m.insert([1, 2, 3], [-10.0, -10.0, -10.0], [1.0, 1.0, 1.0])

    np.testing.assert_allclose(m.probs(np.array([0, 1, 2, 3])), [0.75, 0.25 / 3, 0.25 / 3, 0.25 / 3])

    m.remove([0])
    np.testing.assert_allclose(m.probs(np.array([0, 1, 2, 3])), [0.0, 1 / 3, 1 / 3, 1 / 3])


def test_sampling_empty_mixture_raises_value_error():
    m = WindowMixture(4, [Window(0, 10)])
    assert m.is_defunct

    with pytest.raises(ValueError):
        m.sample(Rng(0), 3)


def test_invalid_window_raises_value_error():
    with pytest.raises(ValueError):
        WindowMixture(4, [Window(5, 5)])


def test_pickle_round_trip_preserves_probs_and_later_advances():
    m = WindowMixture(4, [Window(0, 10, p=0.5), Window(10, np.inf, p=0.5)])
    m.insert([0, 1], [0.0, 3.0], [1.0, 2.0])
    restored = pickle.loads(pickle.dumps(m))
    elements = np.array([0, 1, 2, 3])
    np.testing.assert_allclose(restored.probs(elements), m.probs(elements))

    m.advance(12.0)
    restored.advance(12.0)

    assert restored.counts == m.counts
    np.testing.assert_allclose(restored.probs(elements), m.probs(elements))


def test_seeded_sampling_is_reproducible():
    m = WindowMixture(4, [Window(0, 10, p=0.5), Window(10, np.inf, p=0.5)])
    m.insert([0, 1, 2, 3], [0.0, 0.0, -10.0, -10.0], [1.0, 2.0, 1.0, 1.0])

    np.testing.assert_array_equal(m.sample(Rng(7), 50), m.sample(Rng(7), 50))
