"""Utility functions for sieve estimation methods."""

from __future__ import annotations

import numpy as np


def generate_factors(n: int, dim_limit: int) -> list[list[int]]:
    """Generate all possible ways to factor n into at most dim_limit factors.

    Parameters
    ----------
    n : int
        The positive integer to factorize.
    dim_limit : int
        Maximum number of factors allowed in each factorization.

    Returns
    -------
    list[list[int]]
        List of factorizations, where each factorization is a list of factors.
        Each inner list contains factors that multiply to n.

    Examples
    --------
    Generate all factorizations of 12 with at most 3 factors:

    .. ipython::

        In [1]: from sieve import generate_factors
           ...: generate_factors(12, 3)

    References
    ----------

    .. [1] Chen, X. (2007). *Large Sample Sieve Estimation of Semi-Nonparametric Models.*
        Handbook of Econometrics, 6, 5549-5632. https://doi.org/10.1016/S1573-4412(07)06076-X
    """
    _validate_inputs(n, dim_limit)

    if n == 1:
        return []

    all_factorizations = _find_factorizations(n)
    return _filter_by_dimension(all_factorizations, dim_limit)


def all_add_one(index: np.ndarray) -> np.ndarray:
    """Generate all vectors obtained by incrementing exactly one element of index.

    For each position in the input vector, create a new vector with that position
    incremented by 1 while keeping all other positions unchanged.

    Parameters
    ----------
    index : ndarray
        1-dimensional array of indices.

    Returns
    -------
    ndarray
        2D array of shape (len(index), len(index)) where each row is the input
        vector with exactly one element incremented.
    """
    index = np.asarray(index, dtype=int).ravel()
    xdim = len(index)

    added = np.tile(index, (xdim, 1))

    for i in range(xdim):
        added[i, i] += 1

    return added


def _validate_inputs(n: int, dim_limit: int) -> None:
    """Validate inputs for factorization."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if dim_limit <= 0:
        raise ValueError("dim_limit must be a positive integer")


def _find_factorizations(n: int, min_factor: int = 2) -> list[list[int]]:
    """Find all factorizations of n using factors greater than or equal to min_factor."""
    if n == 1:
        return []

    factorizations = []

    for factor in range(min_factor, int(n**0.5) + 1):
        if n % factor == 0:
            quotient = n // factor
            for sub_factorization in _find_factorizations(quotient, factor):
                factorizations.append([factor] + sub_factorization)

    factorizations.append([n])

    return factorizations


def _filter_by_dimension(factorizations: list[list[int]], dim_limit: int) -> list[list[int]]:
    """Filter factorizations by maximum dimension."""
    return [factors for factors in factorizations if len(factors) <= dim_limit]
