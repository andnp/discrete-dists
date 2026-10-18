"""Benchmarks mirroring the replay-buffer call shapes on Proportional."""

import numpy as np
import pytest

from discrete_dists import Proportional

SIZE = 1_000_000
BATCH = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def p(rng: np.random.Generator):
    dist = Proportional(SIZE)
    dist.update(np.arange(SIZE), rng.uniform(0.1, 2.0, SIZE))
    return dist


def test_sample_batch(p: Proportional, rng: np.random.Generator, benchmark):
    benchmark(p.sample, rng, BATCH)


def test_update_single(p: Proportional, rng: np.random.Generator, benchmark):
    idx = int(rng.integers(SIZE))
    val = float(rng.uniform(0.1, 2.0))

    benchmark(p.update_single, idx, val)


def test_update_batch(p: Proportional, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE, BATCH, replace=False)
    vals = rng.uniform(0.1, 2.0, BATCH)

    benchmark(p.update, idxs, vals)


def test_probs_batch(p: Proportional, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE, BATCH, replace=False)

    benchmark(p.probs, idxs)


def _feed_one_at_a_time(dist: Proportional, idxs: np.ndarray, vals: np.ndarray):
    for i, w in zip(idxs, vals):
        dist.update(np.array([i]), np.array([w]))


def test_feed_loop(p: Proportional, rng: np.random.Generator, benchmark):
    idxs = rng.choice(SIZE, BATCH, replace=False)
    vals = rng.uniform(0.1, 2.0, BATCH)

    benchmark(_feed_one_at_a_time, p, idxs, vals)


def test_stratified_sample_batch(p: Proportional, rng: np.random.Generator, benchmark):
    benchmark(p.stratified_sample, rng, BATCH)


def test_sample_without_replacement(p: Proportional, rng: np.random.Generator, benchmark):
    benchmark(p.sample_without_replacement, rng, BATCH)


def test_scale(p: Proportional, benchmark):
    benchmark(p.scale, 0.999)
