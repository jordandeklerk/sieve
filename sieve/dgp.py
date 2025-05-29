"""Data generation utilities for sieve estimation."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from scipy.stats import multivariate_normal

from .basis import multi_psi, psi
from .preprocessing import create_index_matrix


def generate_samples(
    n_samples: int,
    x_dim: int = 1,
    x_distribution: Literal["uniform", "block_diag", "block_diag_overlap", "dep12"] = "uniform",
    x_params: dict[str, Any] | None = None,
    true_function: str = "linear",
    true_function_params: Any = 1e2,
    y_type: Literal["continuous", "binary"] = "continuous",
    noise_distribution: Literal["normal"] = "normal",
    noise_level: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    r"""Generate synthetic data for testing sieve estimation methods.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    x_dim : int, default=1
        Dimension of feature vectors.
    x_distribution : {"uniform", "block_diag", "block_diag_overlap", "dep12"}, default="uniform"
        Distribution of features:

        - "uniform": Uniform on :math:`[0, 1]^d`
        - "block_diag": Block diagonal covariance structure
        - "block_diag_overlap": Adversarial example with overlapping features
        - "dep12": First two dimensions are dependent

    x_params : dict or None, default=None
        Parameters for feature distribution. For "block_diag":

        - "mean": Mean of each feature
        - "var": Marginal variance
        - "cov": Covariance within blocks
        - "block_size": Size of each block

    true_function : str, default="linear"
        True regression function. Options include:

        "linear", "constant", "piecewise_constant", "piecewise_linear",
        "additive", "theoretical_cosine", "interaction", "sinsin", "sparse",
        "lineartensor", "turntensor", "multiellipsoid", "STE", "sinproduct",
        "linear_binary", "nonlinear_binary"
    true_function_params : Any, default=1e2
        Parameters for the true function.
    y_type : {"continuous", "binary"}, default="continuous"
        Type of outcome variable.
    noise_distribution : {"normal"}, default="normal"
        Distribution of noise (for continuous outcomes).
    noise_level : float, default=0.5
        Standard deviation of noise.
    rng : np.random.Generator or None, default=None
        Random number generator. If None, uses default generator.

    Returns
    -------
    dict[str, ndarray]
        Dictionary containing:

        - 'y': Outcome variable (n_samples,)
        - 'X': Feature matrix (n_samples, x_dim)
    """
    if rng is None:
        rng = np.random.default_rng()

    X = _generate_features(n_samples, x_dim, x_distribution, x_params, rng)

    if y_type == "continuous":
        if noise_distribution == "normal":
            if true_function == "multiellipsoid":
                basis_n = true_function_params
                index_matrix = create_index_matrix(x_dim, basis_n=basis_n, interaction_order=x_dim)[:, 1:]
                y = np.array([_true_function(X[i], true_function, index_matrix, rng) for i in range(n_samples)])
                y += rng.standard_normal(n_samples) * noise_level

            elif true_function == "STE":
                basis_n = true_function_params.get("basisN", 50)
                D = true_function_params.get("D", x_dim)
                index_matrix = create_index_matrix(D, basis_n=basis_n, interaction_order=D)[:basis_n, 1:]
                params = {**true_function_params, "index_matrix": index_matrix}
                y = np.array([_true_function(X[i, :D], true_function, params, rng) for i in range(n_samples)])
                y += rng.standard_normal(n_samples) * noise_level

            elif true_function == "block_diag_sparse_linear":
                all_dim = np.arange(x_dim)
                block_size = x_params["block_size"]
                causal_dim = all_dim[all_dim % block_size == 0]
                causal_X = X[:, causal_dim]
                y = np.array(
                    [_true_function(causal_X[i], "linear", true_function_params, rng) for i in range(n_samples)]
                )
                y += rng.standard_normal(n_samples) * noise_level

            elif true_function == "block_diag_overlap_sparse_linear":
                causal_X = X[:, [0, 2]]
                y = np.array(
                    [_true_function(causal_X[i], "linear", true_function_params, rng) for i in range(n_samples)]
                )
                y += rng.standard_normal(n_samples) * noise_level

            else:
                y = np.array([_true_function(X[i], true_function, true_function_params, rng) for i in range(n_samples)])
                y += rng.standard_normal(n_samples) * noise_level

    else:
        y = np.array([_true_function(X[i], true_function, true_function_params, rng) for i in range(n_samples)])

    return {"y": y, "X": X}


def _generate_features(
    n_samples: int, x_dim: int, x_distribution: str, x_params: dict[str, Any] | None, rng: np.random.Generator
) -> np.ndarray:
    """Generate feature matrix according to specified distribution."""
    if x_distribution == "uniform":
        return rng.uniform(0, 1, size=(n_samples, x_dim))

    if x_distribution == "block_diag":
        if x_params is None:
            raise ValueError("x_params required for block_diag distribution")

        x_mean = x_params["mean"]
        x_var = x_params["var"]
        x_cov = x_params["cov"]
        x_block_size = x_params["block_size"]
        x_block_num = x_dim // x_block_size

        if x_dim != x_block_size * x_block_num:
            warnings.warn("x dimension cannot be divided by block size")

        Sigma = np.full((x_block_size, x_block_size), x_cov)
        np.fill_diagonal(Sigma, x_var)

        X = np.zeros((n_samples, x_block_size * x_block_num))
        for i in range(x_block_num):
            block_data = multivariate_normal.rvs(mean=np.repeat(x_mean, x_block_size), cov=Sigma, size=n_samples)
            if n_samples == 1:
                block_data = block_data.reshape(1, -1)
            X[:, i * x_block_size : (i + 1) * x_block_size] = block_data

        return X

    if x_distribution == "block_diag_overlap":
        if x_params is None:
            raise ValueError("x_params required for block_diag_overlap distribution")

        feature_noise = x_params.get("feature_noise", 1.0)

        x1 = rng.standard_normal(n_samples)
        x3 = rng.standard_normal(n_samples)
        x2 = 0.5 * x1 + 2 * x3 + rng.standard_normal(n_samples) * feature_noise

        return np.column_stack([x1, x2, x3])

    if x_distribution == "dep12":
        X = rng.uniform(0, 1, size=(n_samples, x_dim))
        temp = np.column_stack([X[:, 1], X[:, 0]])
        X[:, :2] = (X[:, :2] + 0.5 * temp) / 2
        return X

    raise ValueError(f"Unknown x_distribution: {x_distribution}")


def _true_function(
    x: np.ndarray, function_type: str, params: Any = None, rng: np.random.Generator | None = None
) -> float:
    """Evaluate true regression function at point x."""
    x = np.asarray(x).ravel()
    xdim = len(x)
    y = 0.0

    if function_type == "linear":
        return np.sum(x)

    if function_type == "constant":
        return 1.0

    if function_type == "piecewise_constant":
        return float(x[0] > 0.5)

    if function_type == "piecewise_linear":
        return float(x[0] > 0.5) + x[0]

    if function_type == "additive":
        D = params if params is not None else xdim
        for i in range(D):
            if i % 2 == 0:
                y += 0.5 - abs(x[i] - 0.5)
            else:
                y += np.exp(-x[i])
        return y

    if function_type == "theoretical_cosine":
        decay_rate = params if params is not None else 2.0
        for j in range(1, 31):
            y += j ** (-decay_rate) * np.cos((j - 1) * np.pi * x[0])
        return y

    if function_type == "interaction":
        D = params if params is not None else xdim
        for i in range(D - 1):
            y += psi(x[i], 2, "legendre") * psi(x[i + 1], 3, "legendre")
        return y

    if function_type == "sinsin":
        for i in range(xdim - 1):
            y += np.sin(2 * np.pi * x[i]) * np.sin(2 * np.pi * x[i + 1])
        return y

    if function_type == "sparse":
        y = 1.0
        y -= np.sin(1.5 * x[0]) * (x[1] ** 3) + 1.5 * (x[1] - 0.5) ** 2
        if xdim >= 4:
            y += np.sin(x[0]) * np.cos(x[1]) * (x[2] ** 3 + 1.5 * (x[2] - 0.5) ** 2) * np.sin(np.exp(-0.5 * x[3]))
            y += np.sin(x[1]) * np.cos(x[2]) * (x[3] ** 3 + 1.5 * (x[3] - 0.5) ** 2) * np.sin(np.exp(-0.5 * x[0]))
        return y

    if function_type == "lineartensor":
        D = params if params is not None else xdim
        for i in range(D - 1):
            y += psi(x[i], 3, "legendre") + psi(x[i], 2, "legendre") * psi(x[i + 1], 2, "legendre")
        return y

    if function_type == "turntensor":
        for i in range(xdim):
            for j in range(i, xdim):
                y += (0.5 - abs(x[i] - 0.5)) * (0.5 - abs(x[j] - 0.5))
        return y

    if function_type == "multiellipsoid":
        index_matrix = params
        for i in range(index_matrix.shape[0]):
            y += np.prod(index_matrix[i]) ** (-1.5) * multi_psi(x, index_matrix[i], "sobolev1")
        return y

    if function_type == "STE":
        index_matrix = params["index_matrix"]
        basis_n = params["basisN"]
        for i in range(basis_n):
            if np.prod(index_matrix[i]) <= 8:
                y += multi_psi(x, index_matrix[i], "cosine")
            else:
                y += 0.0
        return y

    if function_type == "sinproduct":
        y1 = 1.0
        y2 = 1.0
        for i in range(xdim):
            y1 *= np.cos(3 * x[i])
            y2 *= np.cos(6 * x[i])
        return y1 + y2

    if function_type == "linear_binary":
        if rng is None:
            rng = np.random.default_rng()
        prob = np.mean(x)
        return float(rng.binomial(1, prob))

    if function_type == "nonlinear_binary":
        if rng is None:
            rng = np.random.default_rng()
        prob = np.mean(np.abs(x - 0.5)) + 0.2
        return float(rng.binomial(1, prob))

    raise ValueError(f"Unknown function type: {function_type}")
