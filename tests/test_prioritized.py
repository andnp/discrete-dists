import pickle

import numpy as np
import pytest

from discrete_dists import Prioritized, Rng


def brute_weights(priorities, alpha, beta):
    p = np.asarray(priorities, dtype=np.float64)
    present = p > 0
    stored = np.where(present, p**alpha, 0.0)
    probs = stored / stored.sum()
    n = present.sum()
    safe = np.where(present, n * probs, 1.0)
    raw = np.where(present, safe**-beta, 0.0)
    return raw / raw.max()


def test_weights_match_brute_force_formula():
    rng = np.random.default_rng(0)
    priorities = rng.uniform(0.1, 10.0, size=200)
    priorities[rng.random(200) < 0.2] = 0.0
    p = Prioritized(200, 0.6)
    p.update(np.arange(200), priorities)

    weights = p.importance_weights(np.arange(200), 0.7)

    np.testing.assert_allclose(weights, brute_weights(priorities, 0.6, 0.7), rtol=1e-12)


@pytest.mark.parametrize("stratified", [False, True])
def test_sample_with_weights_is_reproducible_and_matches_importance_weights(stratified):
    p = Prioritized(100, 0.6)
    p.update(np.arange(0, 100, 2), np.arange(1, 51, dtype=np.float64))

    samples, weights = p.sample_with_weights(Rng(7), 64, 0.5, stratified=stratified)
    again, again_weights = p.sample_with_weights(Rng(7), 64, 0.5, stratified=stratified)

    np.testing.assert_array_equal(samples, again)
    np.testing.assert_array_equal(weights, again_weights)
    np.testing.assert_allclose(weights, p.importance_weights(samples, 0.5), rtol=1e-12)


def test_priority_zero_removes_element_from_probs_samples_and_max():
    p = Prioritized(4, 1.0)
    p.update([0, 1, 2], [1.0, 4.0, 2.0])

    p.update([1], [0.0])

    assert p.max_priority == 2.0
    np.testing.assert_allclose(p.probs(np.array([0, 1, 2])), [1 / 3, 0.0, 2 / 3])
    assert p.importance_weights([1], 1.0)[0] == 0.0
    assert 1 not in set(p.sample(Rng(0), 500).tolist())


def test_max_priority_is_none_until_an_element_has_positive_priority():
    p = Prioritized(3, 1.0)
    assert p.max_priority is None

    p.update([0, 2], [0.0, 0.0])

    assert p.max_priority is None
    assert p.is_defunct


def test_invalid_alpha_and_beta_raise_value_error():
    with pytest.raises(ValueError):
        Prioritized(4, -1.0)
    with pytest.raises(ValueError):
        Prioritized(4, np.nan)

    p = Prioritized(4, 1.0)
    p.update([0], [1.0])
    with pytest.raises(ValueError):
        p.importance_weights([0], -0.5)


def test_pickle_round_trip_preserves_weights_and_draws():
    p = Prioritized(8, 0.6)
    p.update([1, 4, 6], [2.0, 0.5, 3.0])
    restored = pickle.loads(pickle.dumps(p))

    assert restored.alpha == 0.6
    assert restored.size == 8
    assert restored.max_priority == 3.0
    np.testing.assert_allclose(
        restored.importance_weights([1, 4, 6], 0.5),
        p.importance_weights([1, 4, 6], 0.5),
    )
    s1, w1 = p.sample_with_weights(Rng(4), 16, 0.5)
    s2, w2 = restored.sample_with_weights(Rng(4), 16, 0.5)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(w1, w2)
