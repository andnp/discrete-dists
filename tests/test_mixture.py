import pytest
import numpy as np
from discrete_dists import Alias
from discrete_dists import Rng
from discrete_dists import Distribution
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional
from discrete_dists.uniform import Uniform
from discrete_dists.uniform_set import UniformSet


class Custom(Distribution):
    pass


@pytest.fixture
def rng():
    return Rng(0)


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

    with pytest.raises(ValueError, match="no mass"):
        m.sample(rng, 1)


def test_mixture_stratified_sample_all_defunct(rng):
    m = MixtureDistribution([
        SubDistribution(Uniform(0), p=0.5),
        SubDistribution(Proportional(10), p=0.5),
    ])

    with pytest.raises(ValueError, match="no mass"):
        m.stratified_sample(rng, 1)


def test_mixture_sample_matches_probs_with_defunct_children(rng):
    m = MixtureDistribution([
        SubDistribution(Uniform(5), p=0.5),
        SubDistribution(Uniform(0), p=0.25),
        SubDistribution(Proportional(10), p=0.25),
    ])

    support = np.arange(5)
    data = m.sample(rng, 20000)
    empirical = np.bincount(data, minlength=5) / len(data)

    assert np.allclose(empirical, m.probs(support), atol=0.02)


def test_mixture_is_defunct_false():
    m = MixtureDistribution([
        SubDistribution(Uniform(5), p=0.5),
        SubDistribution(Uniform((10, 15)), p=0.5),
    ])

    assert m.is_defunct is False


def test_mixture_is_defunct_partial():
    m = MixtureDistribution([
        SubDistribution(Uniform(5), p=0.5),
        SubDistribution(Uniform(0), p=0.5),
    ])

    assert m.is_defunct is False


def test_mixture_is_defunct_true():
    m = MixtureDistribution([
        SubDistribution(Uniform(0), p=0.5),
        SubDistribution(Proportional(10), p=0.5),
    ])

    assert m.is_defunct is True


def test_mixture_probs_disjoint_supports():
    p = Proportional((10, 15))
    p.update(
        np.arange(10, 15),
        np.ones(5),
    )

    m = MixtureDistribution([
        SubDistribution(Uniform(5), p=0.5),
        SubDistribution(p, p=0.5),
    ])

    probs = m.probs(np.array([0, 1, 10, 11, 50]))

    assert np.allclose(probs, np.array([0.1, 0.1, 0.1, 0.1, 0.0]))


def test_unsupported_sub_distribution_raises_type_error():
    """
    Only Uniform and Proportional parts can be sampled in Rust, so any other
    sub-distribution must be rejected at construction with its type named.
    """
    with pytest.raises(TypeError, match="Custom"):
        MixtureDistribution([
            SubDistribution(Custom(), p=1.0),
        ])


def test_sample_without_replacement_fills_most_of_two_uniforms_for_every_seed():
    """
    Drawing 900 distinct values from two Uniform(1000) parts must succeed for
    every seed, not only for most of them.
    """
    m = MixtureDistribution([
        SubDistribution(Uniform(1000), p=0.5),
        SubDistribution(Uniform(1000), p=0.5),
    ])

    for seed in range(20):
        draws = m.sample_without_replacement(Rng(seed), 900)

        assert len(np.unique(draws)) == 900


def test_sample_without_replacement_is_unbiased_by_value():
    """
    Without-replacement draws must not be biased toward small elements when the
    first batch contains duplicates and the result is truncated.
    """
    m = MixtureDistribution([
        SubDistribution(Uniform(1000), p=0.5),
        SubDistribution(Uniform(1000), p=0.5),
    ])
    rng = Rng(0)

    draws = [m.sample_without_replacement(rng, 500) for _ in range(200)]

    assert all(len(np.unique(d)) == 500 for d in draws)
    assert abs(np.concatenate(draws).mean() - 499.5) < 4


def test_uniform_set_and_proportional_probs_follow_shares():
    s = UniformSet(10)
    s.insert([1, 2])
    p = Proportional(10)
    p.update(np.array([5, 6]), np.array([1.0, 3.0]))

    m = MixtureDistribution([
        SubDistribution(s, p=0.5),
        SubDistribution(p, p=0.5),
    ])

    np.testing.assert_allclose(m.probs(np.array([1, 2, 5, 6])), [0.25, 0.25, 0.125, 0.375])


def test_samples_come_only_from_uniform_set_members_and_proportional_support():
    s = UniformSet(10)
    s.insert([1, 2])
    p = Proportional(10)
    p.update(np.array([5, 6]), np.array([1.0, 3.0]))
    m = MixtureDistribution([
        SubDistribution(s, p=0.5),
        SubDistribution(p, p=0.5),
    ])

    for draws in (m.sample(Rng(0), 2000), m.stratified_sample(Rng(0), 2000)):
        assert set(draws.tolist()) <= {1, 2, 5, 6}


def test_update_leaves_uniform_set_part_unchanged():
    s = UniformSet(10)
    s.insert([1, 2])
    p = Proportional(10)
    p.update(np.array([5]), np.array([1.0]))
    m = MixtureDistribution([
        SubDistribution(s, p=0.5),
        SubDistribution(p, p=0.5),
    ])

    m.update(np.array([7]), np.array([2.0]))

    assert len(s) == 2
    assert 7 not in s


def test_alias_part_probs_follow_its_share():
    m = MixtureDistribution([
        SubDistribution(Alias([1.0, 3.0]), p=0.5),
        SubDistribution(Uniform(2), p=0.5),
    ])

    np.testing.assert_allclose(m.probs(np.arange(3)), [0.375, 0.625, 0.0])


def test_samples_match_probs_with_alias_part(rng):
    m = MixtureDistribution([
        SubDistribution(Alias([1.0, 3.0]), p=0.5),
        SubDistribution(Uniform(2), p=0.5),
    ])

    data = m.sample(rng, 20000)
    empirical = np.bincount(data, minlength=2) / len(data)

    assert np.allclose(empirical, m.probs(np.arange(2)), atol=0.02)


def test_update_leaves_alias_part_unchanged():
    a = Alias([1.0, 3.0])
    p = Proportional(10)
    p.update(np.array([5]), np.array([1.0]))
    m = MixtureDistribution([
        SubDistribution(a, p=0.5),
        SubDistribution(p, p=0.5),
    ])

    m.update(np.array([0, 1]), np.array([2.0, 2.0]))

    np.testing.assert_allclose(a.probs(np.array([0, 1])), [0.25, 0.75])


def test_defunct_alias_part_is_excluded_from_probs():
    m = MixtureDistribution([
        SubDistribution(Alias([0.0, 0.0]), p=0.5),
        SubDistribution(Uniform(3), p=0.5),
    ])

    assert m.is_defunct is False
    np.testing.assert_allclose(m.probs(np.arange(3)), [1 / 3, 1 / 3, 1 / 3])
