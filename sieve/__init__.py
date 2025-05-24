# pylint: disable=wildcard-import
"""Penalized sieve estimation in tensor product spaces."""

from sieve._version import __version__
from sieve.basis import design_matrix, kernel, multi_psi, psi, tensor_kernel
from sieve.regression import kernel_matrix, krr_fit, krr_predict, least_squares, predict
from sieve.utils import generate_factors

__all__ = [
    "__version__",
    "design_matrix",
    "generate_factors",
    "kernel",
    "kernel_matrix",
    "krr_fit",
    "krr_predict",
    "least_squares",
    "multi_psi",
    "predict",
    "psi",
    "tensor_kernel",
]
