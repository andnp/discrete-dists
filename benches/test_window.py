"""Benchmarks for window mixtures, mirroring the mixture benchmark shapes."""

import numpy as np
import pytest

from discrete_dists import Rng, Window, WindowMixture

SIZE = 1_000_000
BATCH = 32

SHAPES = {
    "flat_windows": [Window(0, 1000, p=0.5), Window(1000, np.inf, p=0.5)],
    "flat_and_decaying": [Window(p=0.5), Window(0, 1500, p=0.5, half_life=100.0)],
}


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def sampler():
    return Rng(0)


@pytest.fixture(params=list(SHAPES))
def m(request, rng: np.random.Generator):
    # Ages are spread over [0, 2000) so the first 1000 steps of advancing fire window crossings.
    mixture = WindowMixture(SIZE, SHAPES[request.param])
    elements = np.arange(SIZE)
    stamps = -(elements % 2000).astype(np.float64)
    mixture.insert(elements, stamps, rng.uniform(0.1, 2.0, SIZE))
    return mixture


def test_sample_batch(m: WindowMixture, sampler: Rng, benchmark):
    benchmark(m.sample, sampler, BATCH)


def test_insert_advance_step(m: WindowMixture, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE, BATCH, replace=False)
    weights = rng.uniform(0.1, 2.0, BATCH)

    def step():
        m.insert(idxs, np.full(BATCH, m.cursor), weights)
        m.advance(m.cursor + 1.0)

    benchmark(step)
