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
