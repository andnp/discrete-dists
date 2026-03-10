import numpy as np
import pytest

from discrete_dists import Categorical, MixtureDistribution, Proportional, SubDistribution, Uniform


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_mixture_probs_sum_to_one_over_live_support():
    p = Proportional((10, 15))
    p.update(np.arange(10, 15), np.ones(5))

    m = MixtureDistribution([
        SubDistribution(Uniform(5), p=0.4),
        SubDistribution(p, p=0.6),
    ])

    support = np.concatenate((np.arange(5), np.arange(10, 15)))
    assert m.probs(support).sum() == pytest.approx(1.0)


def test_sampling_stays_within_support(rng):
    u = Uniform((10, 20))
    samples = u.sample(rng, 1000)

    assert np.all(samples >= 10)
    assert np.all(samples < 20)


def test_categorical_alias_behaves_like_proportional(rng):
    c = Categorical(4)
    c.update(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))

    data = c.sample(rng, 10000)
    empirical = np.bincount(data, minlength=4) / len(data)

    assert np.allclose(empirical, c.probs(np.arange(4)), atol=0.03)


def test_sample_without_replacement_returns_exact_size(rng):
    u = Uniform(10)
    data = u.sample_without_replacement(rng, 10)

    assert len(data) == 10
    assert len(set(data)) == 10
