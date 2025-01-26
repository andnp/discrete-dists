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
