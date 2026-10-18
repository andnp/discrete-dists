from discrete_dists import (
    Categorical,
    Distribution,
    MixtureDistribution,
    Proportional,
    SubDistribution,
    Support,
    Uniform,
    UniformSet,
    Window,
    WindowMixture,
)


def test_public_api_exports():
    assert Categorical.__name__ == "Categorical"
    assert issubclass(Categorical, Proportional)
    assert Distribution.__name__ == "Distribution"
    assert MixtureDistribution.__name__ == "MixtureDistribution"
    assert SubDistribution.__name__ == "SubDistribution"
    assert Uniform.__name__ == "Uniform"
    assert UniformSet.__name__ == "UniformSet"
    assert Window.__name__ == "Window"
    assert WindowMixture.__name__ == "WindowMixture"
    assert Support == tuple[int, int]
