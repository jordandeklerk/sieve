# pylint: disable=wildcard-import
"""Penalized sieve estimation in tensor product spaces."""

from sieve._version import __version__
from sieve.basis import design_matrix, kernel, multi_psi, psi, tensor_kernel
from sieve.preprocessing import create_index_matrix, normalize_X, sieve_preprocess
from sieve.regression import kernel_matrix, krr_fit, krr_predict, least_squares, predict
from sieve.sieve import sieve_predict, sieve_solver
from sieve.utils import generate_factors

__all__ = [
    "__version__",
    "create_index_matrix",
    "design_matrix",
    "generate_factors",
    "kernel",
    "kernel_matrix",
    "krr_fit",
    "krr_predict",
    "least_squares",
    "multi_psi",
    "normalize_X",
    "predict",
    "psi",
    "sieve_predict",
    "sieve_preprocess",
    "sieve_solver",
    "tensor_kernel",
]
