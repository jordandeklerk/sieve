"""Sieve estimation solver and prediction functions."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression

from .preprocessing import sieve_preprocess
from .regression import least_squares, predict


def sieve_solver(
    model: dict[str, Any],
    y: np.ndarray,
    l1: bool = True,
    family: Literal["gaussian", "binomial"] = "gaussian",
    lambda_vals: np.ndarray | None = None,
    n_lambda: int = 100,
) -> dict[str, Any]:
    r"""Fit sieve regression model.

    Calculate the coefficients by solving either a penalized lasso-type problem
    or a least squares problem. This is the main function for sieve estimation.

    Parameters
    ----------
    model : dict
        Output from :func:`sieve_preprocess` containing the design matrix
        and preprocessing information.
    y : ndarray
        Response variable of length n (training sample size).
    l1 : bool, default=True
        If True, use L1-penalized (lasso) regression. If False, use ordinary
        least squares.
    family : {"gaussian", "binomial"}, default="gaussian"
        Error distribution family:

        - "gaussian": Mean squared error regression
        - "binomial": Logistic regression for binary outcomes

    lambda_vals : ndarray or None, default=None
        Regularization parameters for lasso. If None, automatically chosen.
    n_lambda : int, default=100
        Number of lambda values to try if lambda_vals is None.

    Returns
    -------
    dict
        Updated model dictionary containing:

        - All fields from input model
        - 'beta_hat': Fitted coefficients (basis_n x n_lambda for lasso,
          basis_n x 1 for OLS)
        - 'lambda': Regularization parameters used (for lasso)
        - 'family': Error distribution family

    Examples
    --------
    Fit a sieve regression model:

    .. ipython::

        In [1]: from sieve import sieve_preprocess, sieve_solver
           ...: import numpy as np
           ...: np.random.seed(42)
           ...: X = np.random.rand(100, 2)
           ...: y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * np.random.randn(100)
           ...: model = sieve_preprocess(X, basis_n=20)
           ...: fit = sieve_solver(model, y)
           ...: fit['beta_hat'].shape

    Fit with least squares instead of lasso:

    .. ipython::

        In [2]: fit_ols = sieve_solver(model, y, l1=False)
           ...: fit_ols['beta_hat'].shape

    Fit logistic regression for binary outcomes:

    .. ipython::

        In [3]: y_binary = (y > 0).astype(int)
           ...: fit_logit = sieve_solver(model, y_binary, family="binomial")
           ...: fit_logit['family']
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) != model["Phi"].shape[0]:
        raise ValueError(f"Length of y ({len(y)}) doesn't match number of rows in Phi ({model['Phi'].shape[0]})")

    result = model.copy()
    result["family"] = family

    if not l1:
        Phi = model["Phi"]
        beta_hat = least_squares(Phi, y)
        result["beta_hat"] = beta_hat.reshape(-1, 1)
        return result

    Phi = model["Phi"]

    has_intercept_col = np.all(Phi[:, 0] == 1.0)
    if has_intercept_col:
        Phi_no_intercept = Phi[:, 1:]
    else:
        Phi_no_intercept = Phi

    if family == "gaussian":
        if lambda_vals is None:
            cv_model = LassoCV(
                cv=5,
                n_alphas=n_lambda,
                fit_intercept=True,
                max_iter=5000,
                tol=1e-4,
            )
            cv_model.fit(Phi_no_intercept, y)

            lambda_max = cv_model.alpha_ * 10
            lambda_min = cv_model.alpha_ / 100
            lambda_vals = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda)[::-1]

        n_features = Phi_no_intercept.shape[1]
        beta_hat = np.zeros((n_features + 1, len(lambda_vals)))

        for i, lam in enumerate(lambda_vals):
            lasso = Lasso(
                alpha=lam,
                fit_intercept=True,
                max_iter=5000,
                tol=1e-4,
            )
            lasso.fit(Phi_no_intercept, y)

            beta_hat[0, i] = lasso.intercept_
            beta_hat[1:, i] = lasso.coef_

        if has_intercept_col:
            result["beta_hat"] = beta_hat
        else:
            result["beta_hat"] = beta_hat

    else:  # binomial
        if lambda_vals is None:
            C_vals = np.logspace(-4, 2, n_lambda)
        else:
            C_vals = 1.0 / lambda_vals

        n_features = Phi_no_intercept.shape[1]
        beta_hat = np.zeros((n_features + 1, len(C_vals)))

        for i, C in enumerate(C_vals):
            logit = LogisticRegression(
                penalty="l1",
                C=C,
                solver="liblinear",
                fit_intercept=True,
                max_iter=5000,
                tol=1e-4,
            )
            logit.fit(Phi_no_intercept, y)

            beta_hat[0, i] = logit.intercept_[0]
            beta_hat[1:, i] = logit.coef_[0]

        lambda_vals = 1.0 / C_vals

        if has_intercept_col:
            result["beta_hat"] = beta_hat
        else:
            result["beta_hat"] = beta_hat

    result["lambda"] = lambda_vals

    return result


def sieve_predict(
    model: dict[str, Any],
    X_test: np.ndarray | Any,
    y_test: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    r"""Make predictions using fitted sieve model.

    Use the fitted sieve regression model to predict outcomes for new samples.
    Optionally compute mean squared error if test outcomes are provided.

    Parameters
    ----------
    model : dict
        Fitted model from :func:`sieve_solver`.
    X_test : ndarray or array-like
        Test features with shape (n_test, d). Should have the same format
        as training features provided to :func:`sieve_preprocess`.
    y_test : ndarray or None, default=None
        Test outcomes. If provided, MSE is calculated for regression problems.

    Returns
    -------
    dict
        Dictionary containing:

        - 'y_pred': Predicted values (n_test x n_lambda). For regression,
          these are conditional means. For classification, these are
          probabilities of class 1.
        - 'mse': Mean squared errors for each lambda value (regression only,
          when y_test is provided).

    Examples
    --------
    Make predictions on test data:

    .. ipython::

        In [1]: from sieve import sieve_preprocess, sieve_solver, sieve_predict
           ...: import numpy as np
           ...: np.random.seed(42)
           ...: X_train = np.random.rand(100, 2)
           ...: y_train = np.sin(2 * np.pi * X_train[:, 0]) + 0.1 * np.random.randn(100)
           ...: model = sieve_preprocess(X_train, basis_n=20)
           ...: fit = sieve_solver(model, y_train)
           ...: X_test = np.random.rand(50, 2)
           ...: y_test = np.sin(2 * np.pi * X_test[:, 0]) + 0.1 * np.random.randn(50)
           ...: pred = sieve_predict(fit, X_test, y_test)
           ...: pred['y_pred'].shape

    Predict with binary classification model:

    .. ipython::

        In [2]:
           ...: y_binary = (y_train > 0).astype(int)
           ...: fit_binary = sieve_solver(model, y_binary, family="binomial")
           ...: pred_binary = sieve_predict(fit_binary, X_test)
           ...:
           ...: (pred_binary['y_pred'] >= 0).all() and (pred_binary['y_pred'] <= 1).all()
    """
    if "lambda" in model:
        n_lambda = len(model["lambda"])
        beta_hat = np.asarray(model["beta_hat"])
        if beta_hat.ndim == 1:
            beta_hat = beta_hat.reshape(-1, 1)
    else:
        n_lambda = 1
        beta_hat = np.asarray(model["beta_hat"])
        if beta_hat.ndim == 1:
            beta_hat = beta_hat.reshape(-1, 1)

    test_model = sieve_preprocess(
        X=X_test,
        basis_n=model["basis_n"],
        basis_type=model["basis_type"],
        index_matrix=model["index_matrix"],
        norm_para=model.get("norm_para"),
        norm_feature=model.get("norm_para") is not None,
    )

    Phi_test = test_model["Phi"]
    n_test = Phi_test.shape[0]

    has_intercept_col = np.all(model["Phi"][:, 0] == 1.0)

    y_pred = np.zeros((n_test, n_lambda))

    for i in range(n_lambda):
        if has_intercept_col:
            y_pred[:, i] = predict(Phi_test, beta_hat[:, i])
        else:
            intercept = beta_hat[0, i]
            coefs = beta_hat[1:, i]
            y_pred[:, i] = intercept + predict(Phi_test, coefs)

    family = model.get("family", "gaussian")

    if family == "binomial":
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))

    result = {"y_pred": y_pred}

    if y_test is not None and family == "gaussian":
        y_test = np.asarray(y_test).ravel()
        if len(y_test) != n_test:
            raise ValueError(f"Length of y_test ({len(y_test)}) doesn't match number of test samples ({n_test})")

        mse = np.zeros(n_lambda)
        for i in range(n_lambda):
            mse[i] = np.mean((y_pred[:, i] - y_test) ** 2)
        result["mse"] = mse

    return result
