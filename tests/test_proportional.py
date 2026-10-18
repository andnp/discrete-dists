import pickle
from pathlib import Path

import pytest
import numpy as np
from discrete_dists.proportional import Proportional


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_proportional1(rng):
    p = Proportional(100)
    p.update(
        np.arange(10),
        np.ones(10),
    )

    data = p.sample(rng, 10000)
    vals, counts = np.unique(data, return_counts=True)

    expected_count = 10000 / 10
    thresh = expected_count * 0.05

    assert np.all(vals == np.arange(10))
    assert np.all(
        counts > (expected_count - thresh)
    ) and np.all(
        counts < (expected_count + thresh)
    )


def test_proportional2(rng):
    p = Proportional(100)
    p.update(
        np.arange(10),
        np.ones(10),
    )

    p.update(
        20 + np.arange(10),
        np.ones(10),
    )

    data = p.sample(rng, 100000)
    vals, counts = np.unique(data, return_counts=True)

    expected_count = 100000 / 20
    thresh = expected_count * 0.05

    support = np.concatenate((np.arange(10), 20 + np.arange(10)))
    assert np.all(vals == support)
    assert np.all(
        counts > (expected_count - thresh)
    ) and np.all(
        counts < (expected_count + thresh)
    )


def test_shifted_support(rng):
    p = Proportional((-20, 0))
    p.update(
        np.arange(10) - 10,
        np.ones(10),
    )

    data = p.sample(rng, 10000)
    vals, counts = np.unique(data, return_counts=True)

    expected_count = 10000 / 10
    thresh = expected_count * 0.05

    assert np.all(vals == (np.arange(10) - 10))
    assert np.all(
        counts > (expected_count - thresh)
    ) and np.all(
        counts < (expected_count + thresh)
    )


def test_proportional_stratified1(rng):
    p = Proportional(100)
    p.update(
        np.arange(10),
        np.ones(10),
    )

    p.update(
        20 + np.arange(10),
        np.ones(10),
    )

    data = p.stratified_sample(rng, 5)
    _, counts = np.unique(data, return_counts=True)

    assert np.all(counts == 1)


def test_proportional_sample_matches_probs(rng):
    p = Proportional(3)
    p.update(
        np.arange(3),
        np.array([1.0, 2.0, 3.0]),
    )

    data = p.sample(rng, 20000)
    empirical = np.bincount(data, minlength=3) / len(data)

    assert np.allclose(empirical, p.probs(np.arange(3)), atol=0.02)


def test_proportional_sample_without_replacement(rng):
    p = Proportional(5)
    p.update(
        np.arange(5),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )

    data = p.sample_without_replacement(rng, 3)

    assert len(data) == 3
    assert len(set(data)) == 3


def test_proportional_sample_without_replacement_too_many(rng):
    p = Proportional(5)
    p.update(
        np.array([0, 2]),
        np.array([1.0, 1.0]),
    )

    with pytest.raises(ValueError, match="positive-mass elements"):
        p.sample_without_replacement(rng, 3)


def test_proportional_probs_respect_support():
    p = Proportional((10, 15))
    p.update(
        np.arange(10, 15),
        np.ones(5),
    )

    probs = p.probs(np.array([9, 10, 12, 14, 15]))

    assert np.allclose(probs, np.array([0.0, 0.2, 0.2, 0.2, 0.0]))


def test_proportional_update_support_same_width_shift(rng):
    p = Proportional(5)
    p.update(
        np.arange(5),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )

    p.update_support((10, 15))

    assert np.allclose(p.probs(np.arange(10, 15)), np.array([1, 2, 3, 4, 5]) / 15)


def test_proportional_update_support_widen_preserves_absolute_weights():
    p = Proportional((10, 15))
    p.update(
        np.arange(10, 15),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )

    p.update_support((8, 17))

    assert np.allclose(
        p.probs(np.arange(8, 17)),
        np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]) / 15,
    )


def test_proportional_update_support_rejects_shrink():
    p = Proportional((10, 15))

    with pytest.raises(ValueError, match="cannot shrink"):
        p.update_support((10, 14))


def test_proportional_update_support_rejects_ambiguous_widen():
    p = Proportional((10, 15))

    with pytest.raises(ValueError, match="must contain the existing support"):
        p.update_support((12, 18))


def test_proportional_scale_then_update_mixes_scaled_and_new_weights():
    """
    A weight written after a scale is stored at its raw value, so the scaled
    old weights keep their relative mass against it.
    """
    p = Proportional(3)
    p.update(np.array([0, 1]), np.array([1.0, 1.0]))

    p.scale(0.5)
    p.update(np.array([2]), np.array([1.0]))

    assert np.allclose(p.probs(np.arange(3)), [0.25, 0.25, 0.5])


def test_proportional_scale_alone_leaves_probs_unchanged():
    """
    Multiplying every weight by the same factor cannot change the sampling
    distribution, so probabilities must be identical before and after.
    """
    p = Proportional(3)
    p.update(np.arange(3), np.array([1.0, 2.0, 3.0]))
    before = p.probs(np.arange(3))

    p.scale(0.25)

    assert np.allclose(p.probs(np.arange(3)), before)


@pytest.mark.parametrize("factor", [0.0, float("nan")])
def test_proportional_scale_rejects_invalid_factor(factor):
    """
    Zero and NaN factors would destroy or poison every weight, so they must
    raise and leave the distribution untouched.
    """
    p = Proportional(3)
    p.update(np.arange(3), np.array([1.0, 2.0, 3.0]))
    before = p.probs(np.arange(3))

    with pytest.raises(ValueError, match="scale factor must be finite and positive"):
        p.scale(factor)

    assert np.allclose(p.probs(np.arange(3)), before)


def test_proportional_pickle_round_trip_keeps_scaled_weights():
    """
    A pickled scaled distribution must reload with its weights, not just its
    probabilities, so later updates and totals stay consistent.
    """
    p = Proportional(3)
    p.update(np.arange(3), np.array([1.0, 2.0, 3.0]))
    p.scale(0.5)

    q = pickle.loads(pickle.dumps(p))

    assert np.allclose(q.tree.get_values(np.arange(3)), [0.5, 1.0, 1.5])
    assert q.tree.total() == pytest.approx(3.0)
    assert np.allclose(q.probs(np.arange(3)), p.probs(np.arange(3)))


def test_pickle_from_v1_2_1_loads_and_updates():
    """
    Pickles written by the released 1.2.1 wheel must keep loading and
    updating after core refactors. The fixture pins the bytes that a real
    1.2.1 install produces, so a change to the pickled layout fails here.
    """
    path = Path(__file__).parent / "fixtures" / "proportional_v1_2_1.pkl"
    with open(path, "rb") as f:
        p = pickle.load(f)

    weights = np.array([0.5, 1.0, 0.0, 2.0, 3.0, 0.0, 1.5, 4.0])
    assert np.allclose(p.probs(np.arange(10, 18)), weights / 12.0)
    assert p.tree.total() == 12.0

    p.update(np.array([12]), np.array([6.0]))

    new_weights = weights.copy()
    new_weights[2] = 6.0
    assert np.allclose(p.probs(np.arange(10, 18)), new_weights / 18.0)
