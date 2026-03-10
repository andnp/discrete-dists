from discrete_dists import (
    Categorical,
    Distribution,
    MixtureDistribution,
    Proportional,
    SubDistribution,
    Support,
    Uniform,
)


def test_public_api_exports():
    assert Categorical.__name__ == "Categorical"
    assert issubclass(Categorical, Proportional)
    assert Distribution.__name__ == "Distribution"
    assert MixtureDistribution.__name__ == "MixtureDistribution"
    assert SubDistribution.__name__ == "SubDistribution"
    assert Uniform.__name__ == "Uniform"
    assert Support == tuple[int, int]
