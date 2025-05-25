"""Basis functions for sieve estimation."""

from __future__ import annotations

from typing import Literal

import numpy as np

BasisType = Literal["sobolev1", "cosine", "sine", "tri", "legendre", "poly", "polytri"]


def psi(x: float, j: int, basis_type: BasisType) -> float:
    r"""Compute univariate basis function value.

    Evaluate the :math:`j`-th basis function of the specified type at point :math:`x`.
    Different basis types are suited for different smoothness assumptions
    and boundary conditions.

    Parameters
    ----------
    x : float
        Point at which to evaluate the basis function, typically in :math:`[0, 1]`.
    j : int
        Index of the basis function, starting from :math:`1`.
    basis_type : {"sobolev1", "cosine", "sine", "tri", "legendre", "poly", "polytri"}
        Type of basis function:

        - "sobolev1": Sobolev space basis using sine functions
        - "cosine": Cosine basis functions
        - "sine": Sine basis functions
        - "tri": Trigonometric polynomial basis
        - "legendre": Legendre polynomial basis
        - "poly": Power/polynomial basis
        - "polytri": Mixed polynomial-trigonometric basis

    Returns
    -------
    float
        Value of the :math:`j`-th basis function at :math:`x`.

    Examples
    --------
    Evaluate different basis functions:

    .. ipython::

        In [1]: from sieve.basis import psi
           ...: import numpy as np
           ...: x = 0.5
           ...: psi(x, 1, "sobolev1")  # First Sobolev basis (constant)

        In [2]: psi(x, 2, "sobolev1")  # Second Sobolev basis (sine)

        In [3]: psi(x, 3, "legendre")  # Third Legendre polynomial

    References
    ----------

    .. [1] Chen, X. (2007). *Large Sample Sieve Estimation of Semi-Nonparametric Models.*
        Handbook of Econometrics, 6, 5549-5632. https://doi.org/10.1016/S1573-4412(07)06076-X
    """
    if basis_type == "sobolev1":
        if j == 1:
            return 1.0
        return np.sin((2 * (j - 1) - 1) * np.pi * x / 2)

    if basis_type == "cosine":
        if j == 1:
            return 1.0
        return np.cos((j - 1) * np.pi * x)

    if basis_type == "sine":
        return np.sin((2 * j - 1) * np.pi * x / 2)

    if basis_type in ("tri", "polytri"):
        if j == 1:
            return 1.0
        if j == 2:
            return x
        if j % 2 == 1:
            return np.sqrt(2) * np.cos(2 * np.pi * (j - 1) / 2 * x)
        return np.sqrt(2) * np.sin(2 * np.pi * (j - 2) / 2 * x)

    if basis_type == "poly":
        return x ** (j - 1)

    if basis_type == "legendre":
        # Transform x from [0,1] to [-1,1] for standard Legendre polynomials
        x_transformed = (x - 0.5) * 2

        # Legendre polynomials up to order 9
        if j == 1:
            return 1.0
        if j == 2:
            return x_transformed
        if j == 3:
            return (3 * x_transformed**2 - 1) / 2
        if j == 4:
            return (5 * x_transformed**3 - 3 * x_transformed) / 2
        if j == 5:
            return (35 * x_transformed**4 - 30 * x_transformed**2 + 3) / 8
        if j == 6:
            return (63 * x_transformed**5 - 70 * x_transformed**3 + 15 * x_transformed) / 8
        if j == 7:
            return (231 * x_transformed**6 - 315 * x_transformed**4 + 105 * x_transformed**2 - 5) / 16
        if j == 8:
            return (429 * x_transformed**7 - 693 * x_transformed**5 + 315 * x_transformed**3 - 35 * x_transformed) / 16
        if j == 9:
            return (
                6435 * x_transformed**8
                - 12012 * x_transformed**6
                + 6930 * x_transformed**4
                - 1260 * x_transformed**2
                + 35
            ) / 128
        raise ValueError(f"Legendre polynomial of order {j} not implemented (max order is 9)")

    raise ValueError(f"Unknown basis type: {basis_type}")


def multi_psi(x: np.ndarray, index: np.ndarray, basis_type: BasisType) -> float:
    """Compute multivariate tensor product basis function.

    Evaluate the tensor product of univariate basis functions specified by
    the index vector at the multivariate point x.

    Parameters
    ----------
    x : ndarray
        d-dimensional point at which to evaluate the basis function.
    index : ndarray
        d-dimensional vector of basis function indices for each dimension.
    basis_type : {"sobolev1", "cosine", "sine", "tri", "legendre", "poly", "polytri"}
        Type of univariate basis functions to use.

    Returns
    -------
    float
        Value of the tensor product basis function at x.

    Examples
    --------
    Evaluate a 2D tensor product basis:

    .. ipython::

        In [1]: from sieve.basis import multi_psi
           ...: import numpy as np
           ...: x = np.array([0.5, 0.3])
           ...: index = np.array([2, 3])
           ...: multi_psi(x, index, "sobolev1")

    References
    ----------

    .. [1] Belloni, A., Chernozhukov, V., Chetverikov, D., & Kato, K. (2015).
        *Some new asymptotic theory for least squares series: Pointwise and
        uniform results.* Journal of Econometrics, 186(2), 345-366.
        https://doi.org/10.1016/j.jeconom.2015.02.014
        arXiv preprint https://arxiv.org/abs/1212.0442
    """
    if len(x) != len(index):
        raise ValueError("x and index must have the same dimension")

    result = 1.0
    for xi, idx in zip(x, index):
        result *= psi(float(xi), int(idx), basis_type)
    return result


def design_matrix(
    X: np.ndarray, basis_n: int, basis_type: Literal["cosine", "sine", "polytri", "poly"], index_matrix: np.ndarray
) -> np.ndarray:
    r"""Construct design matrix using tensor product basis functions.

    Build the design matrix :math:`\Phi` where each row corresponds to an observation
    and each column corresponds to a basis function evaluated at that observation.
    This implementation uses optimized computations for specific basis types.

    Parameters
    ----------
    X : ndarray
        :math:`n \times d` matrix of covariates, where :math:`n` is the number of observations
        and :math:`d` is the dimension.
    basis_n : int
        Number of basis functions to use.
    basis_type : {"cosine", "sine", "polytri", "poly"}
        Type of basis functions to use. Only these optimized types are supported
        in this function.
    index_matrix : ndarray
        :math:`basis_n \times d` matrix where each row specifies the indices of univariate
        basis functions for the tensor product.

    Returns
    -------
    ndarray
        :math:`n \times basis_n` design matrix.

    Examples
    --------
    Create a design matrix with cosine basis:

    .. ipython::

        In [1]: from sieve.basis import design_matrix
           ...: import numpy as np
           ...: X = np.random.rand(100, 2)  # 100 observations, 2 dimensions
           ...: index_matrix = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])
           ...: Phi = design_matrix(X, 4, "cosine", index_matrix)
           ...: Phi

    References
    ----------

    .. [1] Newey, W. K. (1997). *Convergence rates and asymptotic normality for
        series estimators.* Journal of Econometrics, 79(1), 147-168.
        https://doi.org/10.1016/S0304-4076(97)00011-0
    """
    n_obs, x_dim = X.shape

    if index_matrix.shape != (basis_n, x_dim):
        raise ValueError(f"index_matrix shape {index_matrix.shape} doesn't match expected ({basis_n}, {x_dim})")

    Phi = np.zeros((n_obs, basis_n))

    if basis_type == "cosine":
        for i in range(n_obs):
            for j in range(basis_n):
                psix = 1.0
                for k in range(x_dim):
                    if index_matrix[j, k] > 1:
                        psix *= np.cos((index_matrix[j, k] - 1) * np.pi * X[i, k])
                Phi[i, j] = psix

    elif basis_type == "sine":
        for i in range(n_obs):
            for j in range(basis_n):
                psix = 1.0
                for k in range(x_dim):
                    idx = index_matrix[j, k]
                    if idx >= 1:
                        psix *= np.sin((2 * idx - 1) * np.pi * X[i, k] / 2)
                Phi[i, j] = psix

    elif basis_type == "polytri":
        for i in range(n_obs):
            for j in range(basis_n):
                psix = 1.0
                for k in range(x_dim):
                    idx = index_matrix[j, k]
                    if idx <= 1:
                        continue
                    if idx == 2:
                        psix *= X[i, k]
                    elif idx % 2 == 1:
                        psix *= np.sqrt(2) * np.cos(2 * np.pi * (idx - 1) / 2 * X[i, k])
                    else:
                        psix *= np.sqrt(2) * np.sin(2 * np.pi * (idx - 2) / 2 * X[i, k])
                Phi[i, j] = psix

    elif basis_type == "poly":
        for i in range(n_obs):
            for j in range(basis_n):
                psix = 1.0
                for k in range(x_dim):
                    if index_matrix[j, k] > 1:
                        psix *= X[i, k] ** (index_matrix[j, k] - 1)
                Phi[i, j] = psix

    else:
        raise ValueError(f"Unsupported basis type for optimized computation: {basis_type}")

    return Phi


def kernel(x: float, z: float, kernel_type: Literal["sobolev1", "gaussian"], kernel_para: float = 1.0) -> float:
    r"""Compute univariate kernel function value.

    Evaluate a kernel function :math:`K(x, z)` that measures similarity between
    two points. Different kernel types correspond to different smoothness assumptions
    on the underlying function space.

    Parameters
    ----------
    x : float
        First input point.
    z : float
        Second input point.
    kernel_type : {"sobolev1", "gaussian"}
        Type of kernel function:

        - "sobolev1": Sobolev space kernel :math:`K(x,z) = 1 + \min(x, z)`
        - "gaussian": Gaussian RBF kernel :math:`K(x,z) = \exp(-\gamma ||x-z||^2)`

    kernel_para : float, default=1.0
        Kernel parameter. For Gaussian kernel, this is :math:`\gamma` controlling
        the bandwidth.

    Returns
    -------
    float
        Kernel function value :math:`K(x, z)`.

    Examples
    --------
    Evaluate different kernel functions:

    .. ipython::

        In [1]: from sieve.basis import kernel
           ...: kernel(0.3, 0.5, "sobolev1")

        In [2]: kernel(0.3, 0.5, "gaussian", kernel_para=2.0)
    """
    if kernel_type == "sobolev1":
        return 1.0 + min(x, z)
    if kernel_type == "gaussian":
        return np.exp(-kernel_para * (x - z) ** 2)
    raise ValueError(f"Unknown kernel type: {kernel_type}")


def tensor_kernel(
    x: np.ndarray,
    z: np.ndarray,
    kernel_type: Literal["sobolev1", "gaussian"],
    kernel_para: float = 1.0,
) -> float:
    r"""Compute tensor product kernel function.

    Evaluate the tensor product kernel :math:`K(x, z) = \prod_{i=1}^d K_i(x_i, z_i)`
    where :math:`K_i` is a univariate kernel function.

    Parameters
    ----------
    x : ndarray
        First d-dimensional input point.
    z : ndarray
        Second d-dimensional input point.
    kernel_type : {"sobolev1", "gaussian"}
        Type of univariate kernel functions to use.
    kernel_para : float, default=1.0
        Kernel parameter applied to each dimension.

    Returns
    -------
    float
        Tensor product kernel value.

    Examples
    --------
    Compute tensor product kernels:

    .. ipython::

        In [1]: from sieve.basis import tensor_kernel
           ...: import numpy as np
           ...: x = np.array([0.3, 0.7])
           ...: z = np.array([0.5, 0.4])
           ...: tensor_kernel(x, z, "sobolev1")

        In [2]: tensor_kernel(x, z, "gaussian", kernel_para=2.0)
    """
    if len(x) != len(z):
        raise ValueError("x and z must have the same dimension")

    result = 1.0
    for xi, zi in zip(x, z):
        result *= kernel(float(xi), float(zi), kernel_type, kernel_para)
    return result
