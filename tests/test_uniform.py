import pytest
import numpy as np
from discrete_dists.uniform import Uniform


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_uniform1(rng):
    u = Uniform(10)

    data = u.sample(rng, 1000)
    vals, counts = np.unique(data, return_counts=True)

    assert np.all(vals == np.arange(10))
    assert np.all(counts > 75) and np.all(counts < 125)


def test_uniform2(rng):
    u = Uniform((10, 20))

    data = u.sample(rng, 1000)
    vals, counts = np.unique(data, return_counts=True)

    assert np.all(vals == (10 + np.arange(10)))
    assert np.all(counts > 75) and np.all(counts < 125)


def test_uniform_update(rng):
    u = Uniform(10)
    u.update(np.array([1, 2, 4, 8, 19]))

    data = u.sample(rng, 10000)
    vals, counts = np.unique(data, return_counts=True)

    expected_count = 10000 / 20
    thresh = expected_count * 0.2

    assert np.all(vals == np.arange(20))
    assert np.all(
        counts > (expected_count - thresh)
    ) and np.all(
        counts < (expected_count + thresh)
    )


def test_uniform_sample_wo_replace(rng):
    u = Uniform(10)

    for _ in range(1000):
        data = u.sample_without_replacement(rng, 5)
        assert len(set(data)) == 5


def test_uniform_sample_wo_replace_too_many(rng):
    u = Uniform(3)

    with pytest.raises(ValueError, match="support of size 3"):
        u.sample_without_replacement(rng, 4)


def test_stratified_sample(rng):
    u = Uniform(10)

    data = u.stratified_sample(rng, 5)
    assert len(data) == 5


def test_uniform_probs_respect_support():
    u = Uniform((10, 15))

    probs = u.probs(np.array([9, 10, 12, 14, 15]))

    assert np.allclose(probs, np.array([0.0, 0.2, 0.2, 0.2, 0.0]))


def test_uniform_probs_defunct():
    u = Uniform(0)

    probs = u.probs(np.array([0, 1, 2]))

    assert np.allclose(probs, np.zeros(3))


def test_uniform_sample_matches_probs(rng):
    u = Uniform(5)

    data = u.sample(rng, 20000)
    empirical = np.bincount(data, minlength=5) / len(data)

    assert np.allclose(empirical, u.probs(np.arange(5)), atol=0.02)


def test_isr_within_support():
    source = Uniform(10)
    target = Uniform(5)

    ratios = source.isr(target, np.array([0, 1, 4]))

    assert np.allclose(ratios, np.array([2.0, 2.0, 2.0]))


def test_isr_zero_source_probability():
    source = Uniform(5)
    target = Uniform(10)

    ratios = source.isr(target, np.array([4, 8, 20]))

    assert ratios[0] == pytest.approx(0.5)
    assert ratios[1] == np.inf
    assert np.isnan(ratios[2])
