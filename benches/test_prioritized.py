"""Benchmarks for Prioritized updates and weighted sampling at a full 1M-element distribution."""

import numpy as np
import pytest

from discrete_dists import Prioritized, Rng

SIZE = 1_000_000
BATCH = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def sampler():
    return Rng(0)


@pytest.fixture
def p():
    prioritized = Prioritized(SIZE, 0.6)
    prioritized.update(np.arange(SIZE), np.arange(1, SIZE + 1, dtype=np.float64))
    return prioritized


def test_sample_with_weights_batch(p: Prioritized, sampler: Rng, benchmark):
    benchmark(p.sample_with_weights, sampler, BATCH, 0.5)


def test_update_batch(p: Prioritized, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE, BATCH, replace=False)
    priorities = np.full(BATCH, 2.0)

    benchmark(p.update, idxs, priorities)
