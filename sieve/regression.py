"""Regression methods for sieve estimation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import linalg

from .basis import tensor_kernel


def kernel_matrix(
    X: np.ndarray,
    kernel_type: Literal["sobolev1", "gaussian"],
    kernel_para: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Construct kernel matrix and perform singular value decomposition.

    Compute the kernel matrix :math:`K_{ij} = K(x_i, x_j)` for all pairs of observations
    and perform SVD decomposition :math:`K = U S V^T`.

    Parameters
    ----------
    X : ndarray
        :math:`n \times d` matrix of covariates.
    kernel_type : {"sobolev1", "gaussian"}
        Type of kernel function to use.
    kernel_para : float, default=1.0
        Kernel parameter.

    Returns
    -------
    K : ndarray
        :math:`n \times n` kernel matrix.
    U : ndarray
        :math:`n \times n` matrix of left singular vectors.
    s : ndarray
        :math:`n`-dimensional vector of singular values.

    Examples
    --------
    Compute kernel matrix and SVD:

    .. ipython::

        In [1]: from sieve.regression import kernel_matrix
           ...: import numpy as np
           ...: X = np.random.rand(50, 2)
           ...: K, U, s = kernel_matrix(X, "sobolev1")
           ...: K.shape

        In [2]: s[:5]  # First 5 singular values
    """
    n = X.shape[0]
    K = np.zeros((n, n))

    # Compute kernel matrix
    for i in range(n):
        for j in range(i, n):
            k_val = tensor_kernel(X[i], X[j], kernel_type, kernel_para)
            K[i, j] = k_val
            K[j, i] = k_val

    # SVD
    U, s, _ = linalg.svd(K)

    return K, U, s


def least_squares(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Solve least squares problem.

    Compute the least squares solution :math:`\hat{\beta} = (X^T X)^{-1} X^T y`
    using a numerically stable method.

    Parameters
    ----------
    Phi : ndarray
        :math:`n \times p` design matrix.
    y : ndarray
        :math:`n`-dimensional response vector.

    Returns
    -------
    ndarray
        :math:`p`-dimensional coefficient vector :math:`\hat{\beta}`.

    Examples
    --------
    Solve a least squares problem:

    .. ipython::

        In [1]: from sieve.regression import least_squares
           ...: import numpy as np
           ...: n, p = 100, 10
           ...: Phi = np.random.randn(n, p)
           ...: y = np.random.randn(n)
           ...: beta_hat = least_squares(Phi, y)
           ...: beta_hat
    """
    return linalg.lstsq(Phi, y)[0]


def predict(Phi: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    r"""Make predictions using fitted coefficients.

    Compute predictions :math:`\hat{y} = \Phi \hat{\beta}`.

    Parameters
    ----------
    Phi : ndarray
        :math:`n \times p` design matrix.
    beta_hat : ndarray
        :math:`p`-dimensional coefficient vector.

    Returns
    -------
    ndarray
        :math:`n`-dimensional prediction vector.

    Examples
    --------
    Make predictions:

    .. ipython::

        In [1]: from sieve.regression import predict
           ...: import numpy as np
           ...: n, p = 50, 10
           ...: Phi = np.random.randn(n, p)
           ...: beta_hat = np.random.randn(p)
           ...: y_pred = predict(Phi, beta_hat)
           ...: y_pred
    """
    return Phi @ beta_hat


def krr_fit(
    U: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    lambda_reg: float,
) -> np.ndarray:
    r"""Fit kernel ridge regression.

    Compute the KRR solution :math:`\hat{\beta} = (K + \lambda I)^{-1} y`
    using the SVD decomposition of the kernel matrix.

    Parameters
    ----------
    U : ndarray
        :math:`n \times n` matrix of left singular vectors from kernel matrix SVD.
    s : ndarray
        :math:`n`-dimensional vector of singular values.
    y : ndarray
        :math:`n`-dimensional response vector.
    lambda_reg : float
        Regularization parameter :math:`\lambda > 0`.

    Returns
    -------
    ndarray
        :math:`n`-dimensional coefficient vector :math:`\hat{\beta}`.

    Examples
    --------
    Fit kernel ridge regression:

    .. ipython::

        In [1]: from sieve.regression import kernel_matrix, krr_fit
           ...: import numpy as np
           ...: X = np.random.rand(50, 2)
           ...: y = np.random.randn(50)
           ...: K, U, s = kernel_matrix(X, "gaussian")
           ...: beta = krr_fit(U, s, y, lambda_reg=0.1)
           ...: beta
    """
    D_inv = np.diag(1.0 / (s + lambda_reg))
    return U @ D_inv @ U.T @ y


def krr_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    beta_hat: np.ndarray,
    kernel_type: Literal["sobolev1", "gaussian"],
    kernel_para: float = 1.0,
) -> np.ndarray:
    r"""Make predictions using kernel ridge regression.

    Compute predictions :math:`\hat{y}_{test} = K_{test,train} \hat{\beta}`
    where :math:`K_{test,train}` is the kernel matrix between test and training points.

    Parameters
    ----------
    X_train : ndarray
        :math:`n_{train} \times d` training covariate matrix.
    X_test : ndarray
        :math:`n_{test} \times d` test covariate matrix.
    beta_hat : ndarray
        :math:`n_{train}`-dimensional coefficient vector from KRR.
    kernel_type : {"sobolev1", "gaussian"}
        Type of kernel function.
    kernel_para : float, default=1.0
        Kernel parameter.

    Returns
    -------
    ndarray
        :math:`n_{test}`-dimensional prediction vector.

    Examples
    --------
    Make KRR predictions:

    .. ipython::

        In [1]: from sieve.regression import kernel_matrix, krr_fit, krr_predict
           ...: import numpy as np
           ...: X_train = np.random.rand(50, 2)
           ...: y_train = np.random.randn(50)
           ...: X_test = np.random.rand(20, 2)
           ...: K, U, s = kernel_matrix(X_train, "gaussian")
           ...: beta = krr_fit(U, s, y_train, lambda_reg=0.1)
           ...: y_pred = krr_predict(X_train, X_test, beta, "gaussian")
           ...: y_pred
    """
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    # Compute kernel matrix between test and training points
    K_test_train = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            K_test_train[i, j] = tensor_kernel(X_test[i], X_train[j], kernel_type, kernel_para)
    return K_test_train @ beta_hat


def krr_preprocess(
    X: np.ndarray,
    kernel_type: Literal["sobolev1", "gaussian"] = "sobolev1",
    kernel_para: float = 1.0,
    n_lambda: int = 10,
) -> dict[str, Any]:
    r"""Preprocess data for kernel ridge regression.

    Compute kernel matrix and perform SVD decomposition, preparing data for
    efficient KRR fitting with multiple regularization parameters.

    Parameters
    ----------
    X : ndarray
        :math:`n \times d` matrix of covariates.
    kernel_type : {"sobolev1", "gaussian"}, default="sobolev1"
        Type of kernel function to use.
    kernel_para : float, default=1.0
        Kernel parameter.
    n_lambda : int, default=10
        Number of regularization parameters to generate.

    Returns
    -------
    dict
        Dictionary containing:

        - 'K': Kernel matrix
        - 'U': Left singular vectors
        - 's': Singular values
        - 'lambdas': Array of regularization parameters
        - 'X': Original covariate matrix
        - 'kernel_type': Kernel type used
        - 'kernel_para': Kernel parameter used

    Examples
    --------
    Preprocess data for KRR:

    .. ipython::

        In [1]: from sieve.regression import krr_preprocess
           ...: import numpy as np
           ...: X = np.random.rand(100, 2)
           ...: model = krr_preprocess(X, kernel_type="gaussian")
           ...: model['lambdas']
    """
    K, U, s = kernel_matrix(X, kernel_type, kernel_para)

    lambda_min = 1e-4 * np.min(s)
    lambda_max = np.max(s)
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda)

    return {
        "K": K,
        "U": U,
        "s": s,
        "lambdas": lambdas,
        "X": X,
        "kernel_type": kernel_type,
        "kernel_para": kernel_para,
    }


class KernelRidgeRegression:
    """Kernel Ridge Regression estimator.

    Implements kernel ridge regression with automatic hyperparameter selection.

    Parameters
    ----------
    kernel_type : {"sobolev1", "gaussian"}, default="sobolev1"
        Type of kernel function.
    kernel_para : float, default=1.0
        Kernel parameter.
    lambda_reg : float or None, default=None
        Regularization parameter. If None, will be selected by cross-validation.

    Attributes
    ----------
    X_ : ndarray
        Training data.
    y_ : ndarray
        Training targets.
    beta_hat_ : ndarray
        Fitted coefficients.
    lambda_ : float
        Selected regularization parameter.
    kernel_type : str
        Kernel type.
    kernel_para : float
        Kernel parameter.

    Examples
    --------
    Basic usage:

    .. ipython::

        In [1]: from sieve.regression import KernelRidgeRegression
           ...: import numpy as np
           ...: X = np.random.rand(100, 2)
           ...: y = np.sin(X[:, 0]) + 0.1 * np.random.randn(100)
           ...: krr = KernelRidgeRegression(kernel_type="gaussian")
           ...: krr.fit(X, y)
           ...: y_pred = krr.predict(X[:10])
           ...: y_pred.shape
    """

    def __init__(
        self,
        kernel_type: Literal["sobolev1", "gaussian"] = "sobolev1",
        kernel_para: float = 1.0,
        lambda_reg: float | None = None,
    ):
        self.kernel_type = kernel_type
        self.kernel_para = kernel_para
        self.lambda_reg = lambda_reg
        self.X_ = None
        self.y_ = None
        self.beta_hat_ = None
        self.lambda_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KernelRidgeRegression:
        """Fit kernel ridge regression model.

        Parameters
        ----------
        X : ndarray
            Training data of shape (n_samples, n_features).
        y : ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        self
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_ = X
        self.y_ = y

        _, U, s = kernel_matrix(X, self.kernel_type, self.kernel_para)

        if self.lambda_reg is None:
            self.lambda_ = np.median(s) * 0.01
        else:
            self.lambda_ = self.lambda_reg

        self.beta_hat_ = krr_fit(U, s, y, self.lambda_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Parameters
        ----------
        X : ndarray
            Test data of shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Predicted values of shape (n_samples,).
        """
        if self.beta_hat_ is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return krr_predict(self.X_, X, self.beta_hat_, self.kernel_type, self.kernel_para)
