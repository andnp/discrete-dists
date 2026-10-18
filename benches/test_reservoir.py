"""Benchmarks for WeightedReservoir offers into a full reservoir."""

import numpy as np
import pytest

from discrete_dists import Rng, WeightedReservoir

CAPACITY = 10_000
BATCH = 1024


@pytest.fixture
def full_reservoir():
    r = WeightedReservoir(CAPACITY)
    r.offer(Rng(0), np.arange(CAPACITY), np.ones(CAPACITY))
    return r


def test_reservoir_offer_batch(full_reservoir: WeightedReservoir, benchmark):
    elements = np.arange(BATCH)
    weights = 1.0 + elements % 7

    benchmark(full_reservoir.offer, Rng(1), elements, weights)
