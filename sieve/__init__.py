# pylint: disable=wildcard-import
"""Penalized sieve estimation in tensor product spaces."""

from sieve._version import __version__
from sieve.basis import design_matrix, kernel, multi_psi, psi, tensor_kernel
from sieve.dgp import generate_samples
from sieve.preprocessing import create_index_matrix, normalize_X, sieve_preprocess
from sieve.regression import (
    KernelRidgeRegression,
    kernel_matrix,
    krr_fit,
    krr_predict,
    krr_preprocess,
    least_squares,
    predict,
)
from sieve.sieve import sieve_predict, sieve_solver
from sieve.utils import all_add_one, generate_factors

__all__ = [
    "__version__",
    "all_add_one",
    "create_index_matrix",
    "design_matrix",
    "generate_factors",
    "generate_samples",
    "kernel",
    "kernel_matrix",
    "KernelRidgeRegression",
    "krr_fit",
    "krr_predict",
    "krr_preprocess",
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
