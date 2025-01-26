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


def test_stratified_sample(rng):
    u = Uniform(10)

    data = u.stratified_sample(rng, 5)
    assert len(data) == 5
