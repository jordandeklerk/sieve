"""Test sieve estimation functions."""

import numpy as np
import pytest

from sieve.preprocessing import sieve_preprocess
from sieve.sieve import sieve_predict, sieve_solver


def test_sieve_solver_basic():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * np.random.randn(100)

    model = sieve_preprocess(X, basis_n=20)
    fit = sieve_solver(model, y)

    assert isinstance(fit, dict)
    assert "beta_hat" in fit
    assert "lambda" in fit
    assert "family" in fit
    assert fit["family"] == "gaussian"

    assert fit["beta_hat"].shape[0] == 20
    assert fit["beta_hat"].shape[1] > 1


def test_sieve_solver_ols():
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(50)

    model = sieve_preprocess(X, basis_n=10)
    fit = sieve_solver(model, y, l1=False)

    assert "beta_hat" in fit
    assert fit["beta_hat"].shape == (10, 1)
    assert "lambda" not in fit


def test_sieve_solver_binomial():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y_continuous = 2 * X[:, 0] - 1
    y = (y_continuous > 0).astype(int)

    model = sieve_preprocess(X, basis_n=15)
    fit = sieve_solver(model, y, family="binomial")

    assert fit["family"] == "binomial"
    assert "beta_hat" in fit
    assert "lambda" in fit
    assert fit["beta_hat"].shape[0] == 15


def test_sieve_solver_custom_lambda():
    np.random.seed(42)
    X = np.random.rand(80, 2)
    y = np.random.randn(80)

    model = sieve_preprocess(X, basis_n=12)
    lambda_vals = np.array([0.1, 0.05, 0.01, 0.005, 0.001])

    fit = sieve_solver(model, y, lambda_vals=lambda_vals)

    assert len(fit["lambda"]) == 5
    assert fit["beta_hat"].shape == (12, 5)
    np.testing.assert_array_equal(fit["lambda"], lambda_vals)


def test_sieve_solver_invalid_y_length():
    X = np.random.rand(50, 2)
    y = np.random.randn(30)

    model = sieve_preprocess(X, basis_n=10)

    with pytest.raises(ValueError, match="Length of y"):
        sieve_solver(model, y)


def test_sieve_solver_small_n_lambda():
    np.random.seed(42)
    X = np.random.rand(60, 2)
    y = np.random.randn(60)

    model = sieve_preprocess(X, basis_n=10)
    fit = sieve_solver(model, y, n_lambda=10)

    assert len(fit["lambda"]) == 10
    assert fit["beta_hat"].shape == (10, 10)


def test_sieve_predict_basic():
    np.random.seed(42)
    X_train = np.random.rand(100, 2)
    y_train = np.sin(2 * np.pi * X_train[:, 0]) + 0.1 * np.random.randn(100)

    model = sieve_preprocess(X_train, basis_n=20)
    fit = sieve_solver(model, y_train)

    X_test = np.random.rand(50, 2)
    pred = sieve_predict(fit, X_test)

    assert isinstance(pred, dict)
    assert "y_pred" in pred
    assert pred["y_pred"].shape[0] == 50
    assert pred["y_pred"].shape[1] == len(fit["lambda"])


def test_sieve_predict_with_y_test():
    np.random.seed(42)
    X_train = np.random.rand(80, 2)
    y_train = X_train[:, 0] + X_train[:, 1] + 0.1 * np.random.randn(80)

    model = sieve_preprocess(X_train, basis_n=15)
    fit = sieve_solver(model, y_train, n_lambda=5)

    X_test = np.random.rand(40, 2)
    y_test = X_test[:, 0] + X_test[:, 1] + 0.1 * np.random.randn(40)

    pred = sieve_predict(fit, X_test, y_test)

    assert "y_pred" in pred
    assert "mse" in pred
    assert len(pred["mse"]) == 5
    assert np.all(pred["mse"] >= 0)


def test_sieve_predict_binomial():
    np.random.seed(42)
    X_train = np.random.rand(100, 2)
    y_continuous = 2 * X_train[:, 0] - 1
    y_train = (y_continuous > 0).astype(int)

    model = sieve_preprocess(X_train, basis_n=15)
    fit = sieve_solver(model, y_train, family="binomial", n_lambda=5)

    X_test = np.random.rand(30, 2)
    pred = sieve_predict(fit, X_test)

    assert "y_pred" in pred
    assert pred["y_pred"].shape == (30, 5)

    assert np.all(pred["y_pred"] >= 0)
    assert np.all(pred["y_pred"] <= 1)


def test_sieve_predict_ols():
    np.random.seed(42)
    X_train = np.random.rand(60, 2)
    y_train = np.random.randn(60)

    model = sieve_preprocess(X_train, basis_n=10)
    fit = sieve_solver(model, y_train, l1=False)

    X_test = np.random.rand(20, 2)
    pred = sieve_predict(fit, X_test)

    assert pred["y_pred"].shape == (20, 1)


def test_sieve_predict_invalid_y_test_length():
    np.random.seed(42)
    X_train = np.random.rand(50, 2)
    y_train = np.random.randn(50)

    model = sieve_preprocess(X_train, basis_n=10)
    fit = sieve_solver(model, y_train)

    X_test = np.random.rand(20, 2)
    y_test = np.random.randn(15)

    with pytest.raises(ValueError, match="Length of y_test"):
        sieve_predict(fit, X_test, y_test)


def test_sieve_predict_1d_input():
    np.random.seed(42)
    X_train = np.random.rand(100)
    y_train = np.sin(2 * np.pi * X_train) + 0.1 * np.random.randn(100)

    model = sieve_preprocess(X_train, basis_n=15)
    fit = sieve_solver(model, y_train, n_lambda=3)

    X_test = np.random.rand(25)
    pred = sieve_predict(fit, X_test)

    assert pred["y_pred"].shape == (25, 3)


def test_sieve_predict_array_like_input():
    np.random.seed(42)
    X_train = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    y_train = [0.3, 0.7, 1.1, 1.5]

    model = sieve_preprocess(X_train, basis_n=5)
    fit = sieve_solver(model, y_train, l1=False)

    X_test = [[0.2, 0.3], [0.4, 0.5]]
    pred = sieve_predict(fit, X_test)

    assert pred["y_pred"].shape == (2, 1)


def test_sieve_solver_predict_consistency():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = X[:, 0] ** 2 + X[:, 1] ** 2 + 0.05 * np.random.randn(100)

    for basis_n in [10, 20]:
        for basis_type in ["cosine", "sine"]:
            model = sieve_preprocess(X, basis_n=basis_n, basis_type=basis_type)
            fit = sieve_solver(model, y, n_lambda=3)

            pred = sieve_predict(fit, X)

            assert pred["y_pred"].shape == (100, 3)

            assert not np.all(pred["y_pred"] == 0)
            assert np.all(np.isfinite(pred["y_pred"]))


def test_sieve_solver_convergence_warning():
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.random.randn(20)

    model = sieve_preprocess(X, basis_n=50)

    fit = sieve_solver(model, y, n_lambda=5)
    assert "beta_hat" in fit
    assert fit["beta_hat"].shape == (50, 5)
