import numpy as np
import pytest

from discrete_dists import (
    Alias,
    MixtureDistribution,
    Prioritized,
    Proportional,
    Rng,
    SubDistribution,
    Uniform,
    UniformSet,
    WeightedReservoir,
    Window,
    WindowMixture,
)

def filled_prioritized():
    p = Prioritized(10, 1.0)
    p.update(np.arange(10), np.ones(10))
    return p


# Each entry builds a fresh distribution and calls one method that takes element indices.
ELEMENT_CALLS = {
    "Alias.probs": (
        lambda: Alias([1.0, 1.0]),
        lambda d, e: d.probs(e),
    ),
    "Prioritized.probs": (
        lambda: Prioritized(10, 1.0),
        lambda d, e: d.probs(e),
    ),
    "Prioritized.update": (
        lambda: Prioritized(10, 1.0),
        lambda d, e: d.update(e, np.ones(len(e))),
    ),
    "Prioritized.importance_weights": (
        filled_prioritized,
        lambda d, e: d.importance_weights(e, 1.0),
    ),
    "Proportional.probs": (
        lambda: Proportional(10),
        lambda d, e: d.probs(e),
    ),
    "Proportional.update": (
        lambda: Proportional(10),
        lambda d, e: d.update(e, np.ones(len(e))),
    ),
    "Uniform.probs": (
        lambda: Uniform(10),
        lambda d, e: d.probs(e),
    ),
    "Uniform.update": (
        lambda: Uniform(10),
        lambda d, e: d.update(e),
    ),
    "UniformSet.insert": (
        lambda: UniformSet(10),
        lambda d, e: d.insert(e),
    ),
    "UniformSet.probs": (
        lambda: UniformSet(10),
        lambda d, e: d.probs(e),
    ),
    "WeightedReservoir.offer": (
        lambda: WeightedReservoir(10),
        lambda d, e: d.offer(Rng(0), e, np.ones(len(e))),
    ),
    "WindowMixture.insert": (
        lambda: WindowMixture(10, [Window()]),
        lambda d, e: d.insert(e, np.zeros(len(e)), np.ones(len(e))),
    ),
    "WindowMixture.probs": (
        lambda: WindowMixture(10, [Window()]),
        lambda d, e: d.probs(e),
    ),
    "MixtureDistribution.probs": (
        lambda: MixtureDistribution([SubDistribution(Uniform(10), 1.0)]),
        lambda d, e: d.probs(e),
    ),
}


@pytest.mark.parametrize("name", ELEMENT_CALLS)
@pytest.mark.parametrize(
    "elements",
    [
        np.array([1.0, 2.5]),
        np.array([1.0, 2.0]),
        np.array([True, False]),
    ],
    ids=["fractional", "integral-floats", "bool"],
)
def test_non_integer_elements_are_rejected(name, elements):
    make, call = ELEMENT_CALLS[name]

    with pytest.raises(TypeError, match="elements must be integers"):
        call(make(), elements)


@pytest.mark.parametrize("name", ELEMENT_CALLS)
@pytest.mark.parametrize(
    "elements",
    [[1, 2], np.array([1, 2], dtype=np.int32), []],
    ids=["int-list", "int32-array", "empty-list"],
)
def test_integer_and_empty_elements_are_accepted(name, elements):
    make, call = ELEMENT_CALLS[name]

    call(make(), elements)
