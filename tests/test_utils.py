"""Test utility functions."""

import numpy as np
import pytest

from sieve.utils import generate_factors


def test_basic_factorization():
    result = generate_factors(12, 3)
    expected = [[2, 2, 3], [2, 6], [3, 4], [12]]
    assert sorted(result) == sorted(expected)

    result = generate_factors(8, 2)
    expected = [[2, 4], [8]]
    assert sorted(result) == sorted(expected)


def test_prime_number():
    result = generate_factors(7, 3)
    assert result == [[7]]

    result = generate_factors(13, 5)
    assert result == [[13]]


def test_dimension_limit():
    result = generate_factors(24, 2)
    for factors in result:
        assert len(factors) <= 2

    assert [3, 8] in result
    assert [4, 6] in result
    assert [24] in result


def test_small_numbers():
    result = generate_factors(1, 2)
    assert result == []

    result = generate_factors(2, 1)
    assert result == [[2]]


def test_invalid_inputs():
    with pytest.raises(ValueError, match="n must be a positive integer"):
        generate_factors(0, 2)

    with pytest.raises(ValueError, match="n must be a positive integer"):
        generate_factors(-5, 2)

    with pytest.raises(ValueError, match="dim_limit must be a positive integer"):
        generate_factors(10, 0)

    with pytest.raises(ValueError, match="dim_limit must be a positive integer"):
        generate_factors(10, -1)


def test_large_dimension_limit():
    result = generate_factors(6, 10)
    expected = [[2, 3], [6]]
    assert sorted(result) == sorted(expected)


def test_perfect_square():
    result = generate_factors(16, 3)
    expected = [[2, 2, 2, 2], [2, 2, 4], [2, 8], [4, 4], [16]]
    expected = [f for f in expected if len(f) <= 3]
    assert sorted(result) == sorted(expected)


@pytest.mark.parametrize(
    "n,dim_limit,min_factors",
    [
        (30, 3, 3),
        (100, 2, 3),
        (60, 4, 5),
    ],
)
def test_multiple_factorizations(n, dim_limit, min_factors):
    result = generate_factors(n, dim_limit)
    assert len(result) >= min_factors

    for factors in result:
        assert len(factors) <= dim_limit
        assert np.prod(factors) == n
        assert all(f > 1 for f in factors)


def test_factorization_completeness():
    result = generate_factors(12, 4)

    all_factorizations = [[2, 2, 3], [2, 6], [3, 4], [12]]

    assert len(result) == len(all_factorizations)
    for fact in all_factorizations:
        assert fact in result
