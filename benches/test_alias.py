"""Benchmarks for Alias against a Proportional over the same weights."""

import numpy as np
import pytest

from discrete_dists import Alias, Proportional, Rng

SIZE = 1_000_000


@pytest.fixture
def sampler():
    return Rng(0)


@pytest.fixture
def weights():
    return 1.0 + np.arange(SIZE) % 7


@pytest.fixture
def a(weights: np.ndarray):
    return Alias(weights)


@pytest.fixture
def p(weights: np.ndarray):
    tree = Proportional(SIZE)
    tree.update(np.arange(SIZE), weights)
    return tree


@pytest.mark.parametrize("n", [32, 1024])
def test_alias_sample_batch(a: Alias, sampler: Rng, benchmark, n: int):
    benchmark(a.sample, sampler, n)


@pytest.mark.parametrize("n", [32, 1024])
def test_tree_sample_batch(p: Proportional, sampler: Rng, benchmark, n: int):
    benchmark(p.sample, sampler, n)


def test_alias_build(weights: np.ndarray, benchmark):
    benchmark(Alias, weights)
