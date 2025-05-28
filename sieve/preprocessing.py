"""Preprocessing functions for sieve estimation."""

from __future__ import annotations

import warnings
from itertools import combinations, permutations
from typing import Any

import numpy as np

from .basis import BasisType, design_matrix
from .utils import generate_factors


def sieve_preprocess(
    X: np.ndarray | Any,
    basis_n: int | None = None,
    maxj: int | None = None,
    basis_type: BasisType = "cosine",
    interaction_order: int = 3,
    index_matrix: np.ndarray | None = None,
    norm_feature: bool = True,
    norm_para: np.ndarray | None = None,
) -> dict[str, Any]:
    r"""Preprocess data for sieve estimation.

    Generate the design matrix for downstream penalized regression using
    tensor product basis functions. This is the main preprocessing step
    for sieve estimation methods.

    Parameters
    ----------
    X : ndarray or array-like
        n x d matrix of original features.
    basis_n : int or None, default=None
        Number of basis functions. If None, defaults to 50 * d where d is
        the feature dimension. Larger values reduce approximation error but
        increase computational cost.
    maxj : int or None, default=None
        Maximum index product for basis functions. If basis_n is specified,
        this parameter is ignored.
    basis_type : {"cosine", "sine", "polytri", "poly"}, default="cosine"
        Type of univariate basis functions:

        - "cosine": Cosine basis, suitable for most purposes
        - "sine": Sine basis
        - "polytri": Mixed polynomial-trigonometric basis
        - "poly": Polynomial basis

    interaction_order : int, default=3
        Controls model complexity. 1 for additive models, 2 for pairwise
        interactions, etc. Must be <= feature dimension.
    index_matrix : ndarray or None, default=None
        Pre-generated index matrix. If None, one is generated automatically.
    norm_feature : bool, default=True
        Whether to normalize features to [0, 1]. Only set to False if
        features are already normalized.
    norm_para : ndarray or None, default=None
        Normalization parameters from training data. For training, use None.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - 'Phi': n x basis_n design matrix for regression
        - 'X': Normalized feature matrix
        - 'basis_type': Type of basis functions used
        - 'index_matrix': Matrix specifying basis function indices
        - 'basis_n': Number of basis functions
        - 'norm_para': Normalization parameters

    Warnings
    --------
    UserWarning
        If basis_n is not specified, warns about default choice and
        theoretical recommendations.

    Examples
    --------
    Basic usage with automatic basis selection:

    .. ipython::

        In [1]: from sieve.preprocessing import sieve_preprocess
           ...: import numpy as np
           ...: X = np.random.rand(100, 2)
           ...: result = sieve_preprocess(X, basis_n=20)
           ...: result['Phi'].shape

    Fit an additive model (no interactions):

    .. ipython::

        In [2]: result = sieve_preprocess(X, basis_n=30, interaction_order=1)
           ...: # Verify additive structure
           ...: index_mat = result['index_matrix']
           ...: ((index_mat[:, 1:] > 1).sum(axis=1) <= 1).all()

    References
    ----------

    .. [1] Chen, X. (2007). *Large Sample Sieve Estimation of Semi-Nonparametric Models.*
        Handbook of Econometrics, 6, 5549-5632.

    .. [2] Newey, W. K. (1997). *Convergence rates and asymptotic normality for
        series estimators.* Journal of Econometrics, 79(1), 147-168.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, xdim = X.shape

    # Set default basis_n if not specified
    if basis_n is None:
        if maxj is None:
            warnings.warn(
                "User did not specify number of basis functions, default is xdim*50. "
                f"A theoretically good number should be around (sample size)^(1/3) * (feature dimension)^3 "
                f"= {int(n_samples ** (1 / 3) * xdim**3)}"
            )
            basis_n = xdim * 50

    if basis_n is not None and basis_n <= 1:
        raise ValueError("basis_n must be > 1")

    if norm_feature:
        norm_result = normalize_X(X, norm_para)
        X = norm_result["X"]
        norm_para = norm_result["norm_para"]

    if index_matrix is None:
        index_matrix = create_index_matrix(
            xdim=xdim,
            basis_n=basis_n,
            maxj=maxj,
            interaction_order=interaction_order,
        )

    indices_only = index_matrix[:, 1:]

    if basis_type not in ["cosine", "sine", "polytri", "poly"]:
        raise ValueError(
            f"basis_type '{basis_type}' is not supported by the optimized design_matrix function. "
            "Supported types are: 'cosine', 'sine', 'polytri', 'poly'"
        )

    actual_basis_n = indices_only.shape[0]
    Phi = design_matrix(X, actual_basis_n, basis_type, indices_only)

    return {
        "Phi": Phi,
        "X": X,
        "basis_type": basis_type,
        "index_matrix": index_matrix,
        "basis_n": actual_basis_n,
        "norm_para": norm_para,
    }


def normalize_X(
    X: np.ndarray,
    norm_para: np.ndarray | None = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> dict[str, np.ndarray]:
    """Normalize features to [0, 1] range using quantiles.

    Rescale each feature dimension to the unit interval using specified quantiles
    to handle outliers robustly.

    Parameters
    ----------
    X : ndarray
        n x d matrix of features to normalize.
    norm_para : ndarray or None, default=None
        2 x d matrix containing normalization parameters. If None, parameters
        are computed from X. First row contains lower bounds, second row contains
        upper bounds.
    lower_q : float, default=0.01
        Lower quantile for normalization.
    upper_q : float, default=0.99
        Upper quantile for normalization.

    Returns
    -------
    dict[str, ndarray]
        Dictionary containing:
        - 'X': Normalized feature matrix with values in [0, 1]
        - 'norm_para': 2 x d matrix of normalization parameters used

    Examples
    --------
    Normalize features to unit interval:

    .. ipython::

        In [1]: from sieve.preprocessing import normalize_X
           ...: import numpy as np
           ...: X = np.random.randn(100, 3) * 10 + 5
           ...: result = normalize_X(X)
           ...: result['X'].min(), result['X'].max()
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]

    if norm_para is None:
        norm_para = np.zeros((2, n_features))
        X_normalized = X.copy()

        for i in range(n_features):
            lower_val = np.quantile(X[:, i], lower_q)
            upper_val = np.quantile(X[:, i], upper_q)

            if np.abs(upper_val - lower_val) < 1e-10:
                warnings.warn(f"Feature {i} has near-constant values after quantile clipping.")
                X_normalized[:, i] = 0.5
            else:
                X_normalized[:, i] = (X[:, i] - lower_val) / (upper_val - lower_val)

            norm_para[0, i] = lower_val
            norm_para[1, i] = upper_val
    else:
        if norm_para.shape != (2, n_features):
            raise ValueError(f"norm_para shape {norm_para.shape} doesn't match expected (2, {n_features})")

        X_normalized = X.copy()
        for i in range(n_features):
            lower_val = norm_para[0, i]
            upper_val = norm_para[1, i]

            if np.abs(upper_val - lower_val) < 1e-10:
                X_normalized[:, i] = 0.5
            else:
                X_normalized[:, i] = (X[:, i] - lower_val) / (upper_val - lower_val)

    # Clip to [0, 1]
    X_normalized = np.clip(X_normalized, 0, 1)

    return {"X": X_normalized, "norm_para": norm_para}


def create_index_matrix(
    xdim: int,
    basis_n: int | None = None,
    maxj: int | None = None,
    interaction_order: int = 3,
) -> np.ndarray:
    """Create index matrix for multivariate basis functions.

    Generate an index matrix that specifies which univariate basis functions
    to use for constructing tensor product basis functions. Each row represents
    one multivariate basis function.

    Parameters
    ----------
    xdim : int
        Dimension of the feature space.
    basis_n : int or None, default=None
        Number of basis functions to generate. If None, must specify maxj.
    maxj : int or None, default=None
        Maximum product of indices in any row. If None, must specify basis_n.
    interaction_order : int, default=3
        Maximum order of interactions (number of non-1 elements per row).
        1 means additive model, 2 allows pairwise interactions, etc.

    Returns
    -------
    ndarray
        (basis_n + 1) x (xdim + 1) matrix where:
        - First column contains the product of indices in that row
        - Remaining columns contain the indices for each dimension
        - First row is all ones (constant basis)

    Examples
    --------
    Create index matrix for 2D features with 10 basis functions:

    .. ipython::

        In [1]: from sieve.preprocessing import create_index_matrix
           ...: index_matrix = create_index_matrix(xdim=2, basis_n=10)
           ...: index_matrix

    Create index matrix for additive model (no interactions):

    .. ipython::

        In [2]: index_matrix = create_index_matrix(xdim=3, basis_n=15, interaction_order=1)
           ...: # Check that each row has at most one non-1 entry (excluding first column)
           ...: ((index_matrix[:, 1:] > 1).sum(axis=1) <= 1).all()
    """
    if basis_n is None and maxj is None:
        raise ValueError("Must specify either basis_n or maxj")

    if xdim <= 0:
        raise ValueError("xdim must be positive")

    if interaction_order <= 0:
        raise ValueError("interaction_order must be positive")

    interaction_order = min(interaction_order, xdim)

    # Start with constant basis (all ones)
    index_list = [np.ones(xdim, dtype=int)]

    if maxj is not None:
        # Generate up to maxj product value
        for product_v in range(2, maxj + 1):
            _add_product_indices(index_list, product_v, xdim, interaction_order)
    else:
        # Generate until we have enough basis functions
        product_v = 2
        while len(index_list) < basis_n:
            _add_product_indices(index_list, product_v, xdim, interaction_order)
            product_v += 1

        index_list = index_list[:basis_n]

    index_matrix = np.array(index_list, dtype=int)
    products = np.prod(index_matrix, axis=1)

    return np.column_stack([products, index_matrix])


def _add_product_indices(
    index_list: list[np.ndarray],
    product_v: int,
    xdim: int,
    interaction_order: int,
) -> None:
    """Add all valid index combinations for a given product value."""
    # Get all factorizations of product_v
    factorizations = generate_factors(product_v, interaction_order)

    # Add trivial factorization (product_v itself)
    factorizations.append([product_v])

    all_permutations = []
    for factors in factorizations:
        if len(factors) > interaction_order:
            continue

        seen = set()
        for perm in permutations(factors):
            if perm not in seen:
                seen.add(perm)
                all_permutations.append(list(perm))

    # For each permutation, create index vectors
    for perm in all_permutations:
        n_factors = len(perm)

        for positions in combinations(range(xdim), n_factors):
            index_vec = np.ones(xdim, dtype=int)
            for i, pos in enumerate(positions):
                index_vec[pos] = perm[i]
            index_list.append(index_vec)
