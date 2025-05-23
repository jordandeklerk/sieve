# pylint: disable=wildcard-import
"""Penalized sieve estimation in tensor product spaces."""

from sieve._version import __version__
from sieve.basis import design_matrix, multi_psi, psi
from sieve.utils import generate_factors

__all__ = [
    "__version__",
    "design_matrix",
    "generate_factors",
    "multi_psi",
    "psi",
]
