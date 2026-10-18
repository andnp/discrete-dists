"""Benchmarks for mixture sampling, mirroring the MaskedABDistribution shape."""

import numpy as np
import pytest

from discrete_dists import MixtureDistribution, Proportional, SubDistribution

SIZE = 1_000_000
BATCH = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _populated_proportional(rng: np.random.Generator) -> Proportional:
    dist = Proportional(SIZE)
    dist.update(np.arange(SIZE), rng.uniform(0.1, 2.0, SIZE))
    return dist


@pytest.fixture
def m(rng: np.random.Generator):
    return MixtureDistribution([
        SubDistribution(_populated_proportional(rng), 0.75),
        SubDistribution(_populated_proportional(rng), 0.25),
    ])


def test_sample_batch(m: MixtureDistribution, rng: np.random.Generator, benchmark):
    benchmark(m.sample, rng, BATCH)


def test_probs_batch(m: MixtureDistribution, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE, BATCH, replace=False)

    benchmark(m.probs, idxs)


def test_stratified_sample_batch(m: MixtureDistribution, rng: np.random.Generator, benchmark):
    benchmark(m.stratified_sample, rng, BATCH)


def test_update_batch(m: MixtureDistribution, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE, BATCH, replace=False)
    vals = rng.uniform(0.1, 2.0, BATCH)

    benchmark(m.update, idxs, vals)
