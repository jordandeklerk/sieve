"""Utility functions for sieve estimation methods."""

from __future__ import annotations


def generate_factors(n: int, dim_limit: int) -> list[list[int]]:
    """Generate all possible ways to factor n into at most dim_limit factors.

    Find all possible factorizations of n as a product of at most dim_limit
    positive integers greater than 1. This is useful for constructing tensor
    product basis functions in sieve estimation methods.

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
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if dim_limit <= 0:
        raise ValueError("dim_limit must be a positive integer")

    resultant: list[list[int]] = []

    def _factor_list_func(first: int, current_product: int, target: int, current_factors: list[int]) -> None:
        """Recursively find factor combinations."""
        if first > target or current_product > target:
            return

        if current_product == target:
            resultant.append(current_factors.copy())
            return

        for i in range(first, target):
            if i * current_product > target:
                break

            if target % i == 0:
                current_factors.append(i)
                _factor_list_func(i, i * current_product, target, current_factors)
                current_factors.pop()

    # Start recursive factorization from 2
    _factor_list_func(2, 1, n, [])

    # Add the single-element factorization if it fits within dim_limit
    if dim_limit >= 1 and n > 1:
        resultant.append([n])

    return [factors for factors in resultant if len(factors) <= dim_limit]
