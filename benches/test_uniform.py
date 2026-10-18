"""Benchmarks for the Uniform distribution and importance sampling ratios."""

import numpy as np
import pytest

from discrete_dists import Proportional, Uniform

SIZE = 1_000_000
BATCH = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def u():
    return Uniform(SIZE)


def test_sample_batch(u: Uniform, rng: np.random.Generator, benchmark):
    benchmark(u.sample, rng, BATCH)


def test_stratified_sample_batch(u: Uniform, rng: np.random.Generator, benchmark):
    benchmark(u.stratified_sample, rng, BATCH)


def test_sample_without_replacement(u: Uniform, rng: np.random.Generator, benchmark):
    benchmark(u.sample_without_replacement, rng, BATCH)


def test_probs_batch(u: Uniform, rng: np.random.Generator, benchmark):
    idxs = rng.integers(-SIZE // 2, SIZE + SIZE // 2, BATCH)

    benchmark(u.probs, idxs)


def test_isr_to_proportional(u: Uniform, rng: np.random.Generator, benchmark):
    target = Proportional(SIZE)
    target.update(np.arange(SIZE), rng.uniform(0.1, 2.0, SIZE))
    idxs = u.sample(rng, BATCH)

    benchmark(u.isr, target, idxs)
