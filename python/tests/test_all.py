import pytest
import discrete_dists


def test_sum_as_string():
    assert discrete_dists.sum_as_string(1, 1) == "2"
