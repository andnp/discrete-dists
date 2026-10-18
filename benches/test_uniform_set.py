"""Benchmarks for UniformSet against a weight-1 Proportional over the same subset."""

import numpy as np
import pytest

from discrete_dists import Proportional, Rng, UniformSet

SIZE = 1_000_000
BATCH = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def sampler():
    return Rng(0)


@pytest.fixture
def members():
    return np.arange(0, SIZE, 2)


@pytest.fixture
def s(members: np.ndarray):
    u = UniformSet(SIZE)
    u.insert(members)
    return u


@pytest.fixture
def p(members: np.ndarray):
    tree = Proportional(SIZE)
    tree.update(members, np.ones(len(members)))
    return tree


def test_sample_batch(s: UniformSet, sampler: Rng, benchmark):
    benchmark(s.sample, sampler, BATCH)


def test_tree_sample_batch(p: Proportional, sampler: Rng, benchmark):
    benchmark(p.sample, sampler, BATCH)


def test_insert_remove_batch(s: UniformSet, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE // 2, BATCH, replace=False) * 2 + 1

    def step():
        s.insert(idxs)
        s.remove(idxs)

    benchmark(step)


def test_tree_update_batch(p: Proportional, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE // 2, BATCH, replace=False) * 2 + 1
    ones = np.ones(BATCH)
    zeros = np.zeros(BATCH)

    def step():
        p.update(idxs, ones)
        p.update(idxs, zeros)

    benchmark(step)
