"""Benchmarks for the Uniform distribution and importance sampling ratios."""

import numpy as np
import pytest

from discrete_dists import Proportional, Rng, Uniform

SIZE = 1_000_000
BATCH = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def sampler():
    return Rng(0)


@pytest.fixture
def u():
    return Uniform(SIZE)


def test_sample_batch(u: Uniform, sampler: Rng, benchmark):
    benchmark(u.sample, sampler, BATCH)


def test_stratified_sample_batch(u: Uniform, sampler: Rng, benchmark):
    benchmark(u.stratified_sample, sampler, BATCH)


def test_sample_without_replacement(u: Uniform, sampler: Rng, benchmark):
    benchmark(u.sample_without_replacement, sampler, BATCH)


def test_probs_batch(u: Uniform, rng: np.random.Generator, benchmark):
    idxs = rng.integers(-SIZE // 2, SIZE + SIZE // 2, BATCH)

    benchmark(u.probs, idxs)


def test_isr_to_proportional(u: Uniform, rng: np.random.Generator, sampler: Rng, benchmark):
    target = Proportional(SIZE)
    target.update(np.arange(SIZE), rng.uniform(0.1, 2.0, SIZE))
    idxs = u.sample(sampler, BATCH)

    benchmark(u.isr, target, idxs)
