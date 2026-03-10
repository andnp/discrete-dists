import pytest
import numpy as np
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional
from discrete_dists.uniform import Uniform


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_mixture1(rng):
    m = MixtureDistribution([
        SubDistribution(Uniform(5), p=0.5),
        SubDistribution(Uniform((10, 15)), p=0.5),
    ])

    data = m.sample(rng, 100000)
    vals, counts = np.unique(data, return_counts=True)

    expected_count = 100000 / 10
    thresh = expected_count * 0.05

    support = np.concatenate((np.arange(5), 10 + np.arange(5)))
    assert np.all(vals == support)
    assert np.all(
        counts > (expected_count - thresh)
    ) and np.all(
        counts < (expected_count + thresh)
    )


def test_mixture_defunct(rng):
    m = MixtureDistribution([
        SubDistribution(Uniform(5), p=0.5),
        SubDistribution(Uniform(0), p=0.25),
        SubDistribution(Proportional(10), p=0.25),
    ])

    data = m.sample(rng, 100000)
    vals, counts = np.unique(data, return_counts=True)

    expected_count = 100000 / 5
    thresh = expected_count * 0.05

    assert np.all(vals == np.arange(5))
    assert np.all(
        counts > (expected_count - thresh)
    ) and np.all(
        counts < (expected_count + thresh)
    )


def test_mixture_probs_defunct(rng):
    m = MixtureDistribution([
        SubDistribution(Uniform(5), p=0.5),
        SubDistribution(Uniform(0), p=0.25),
        SubDistribution(Proportional(10), p=0.25),
    ])

    probs = m.probs(np.arange(5))

    assert np.allclose(probs, np.full(5, 1 / 5))
    assert probs.sum() == pytest.approx(1.0)


def test_mixture_probs_all_defunct(rng):
    m = MixtureDistribution([
        SubDistribution(Uniform(0), p=0.5),
        SubDistribution(Proportional(10), p=0.5),
    ])

    probs = m.probs(np.arange(5))

    assert np.allclose(probs, np.zeros(5))


def test_mixture_sample_all_defunct(rng):
    m = MixtureDistribution([
        SubDistribution(Uniform(0), p=0.5),
        SubDistribution(Proportional(10), p=0.5),
    ])

    with pytest.raises(ValueError, match="all-defunct mixture"):
        m.sample(rng, 1)


def test_mixture_stratified_sample_all_defunct(rng):
    m = MixtureDistribution([
        SubDistribution(Uniform(0), p=0.5),
        SubDistribution(Proportional(10), p=0.5),
    ])

    with pytest.raises(ValueError, match="all-defunct mixture"):
        m.stratified_sample(rng, 1)
