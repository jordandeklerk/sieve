"""Test regression functions."""

import numpy as np
import pytest

from sieve.regression import kernel_matrix, krr_fit, krr_predict, least_squares, predict


def test_kernel_matrix_basic():
    np.random.seed(42)
    X = np.random.rand(5, 2)

    result = kernel_matrix(X, "sobolev1")

    assert "K" in result
    assert "U" in result
    assert "s" in result

    K = result["K"]
    assert K.shape == (5, 5)
    assert np.allclose(K, K.T)

    assert result["U"].shape == (5, 5)
    assert result["s"].shape == (5,)


def test_kernel_matrix_gaussian():
    np.random.seed(42)
    X = np.random.rand(10, 3)

    result = kernel_matrix(X, "gaussian", kernel_para=2.0)

    K = result["K"]
    assert K.shape == (10, 10)
    assert np.allclose(K, K.T)
    assert np.all(np.diag(K) == 1.0)


def test_kernel_matrix_svd_properties():
    np.random.seed(42)
    X = np.random.rand(8, 2)

    result = kernel_matrix(X, "sobolev1")

    K_reconstructed = result["U"] @ np.diag(result["s"]) @ result["U"].T
    assert np.allclose(result["K"], K_reconstructed)


def test_least_squares_basic():
    np.random.seed(42)
    n, p = 50, 10
    Phi = np.random.randn(n, p)
    y = np.random.randn(n)

    beta_hat = least_squares(Phi, y)

    assert beta_hat.shape == (p,)
    assert not np.any(np.isnan(beta_hat))
    assert not np.any(np.isinf(beta_hat))


def test_least_squares_overdetermined():
    np.random.seed(42)
    Phi = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2, 3, 4])

    beta_hat = least_squares(Phi, y)

    assert beta_hat.shape == (2,)
    assert np.isclose(beta_hat[0], 1.0)
    assert np.isclose(beta_hat[1], 1.0)


def test_predict_basic():
    np.random.seed(42)
    n, p = 30, 5
    Phi = np.random.randn(n, p)
    beta_hat = np.random.randn(p)

    y_pred = predict(Phi, beta_hat)

    assert y_pred.shape == (n,)
    assert np.allclose(y_pred, Phi @ beta_hat)


def test_predict_identity():
    Phi = np.eye(5)
    beta_hat = np.array([1, 2, 3, 4, 5])

    y_pred = predict(Phi, beta_hat)

    assert np.allclose(y_pred, beta_hat)


def test_krr_fit_basic():
    np.random.seed(42)
    n = 20
    X = np.random.rand(n, 2)
    y = np.random.randn(n)

    km = kernel_matrix(X, "gaussian")
    beta = krr_fit(km["U"], km["s"], y, lambda_reg=0.1)

    assert beta.shape == (n,)
    assert not np.any(np.isnan(beta))
    assert not np.any(np.isinf(beta))


def test_krr_fit_regularization_effect():
    np.random.seed(42)
    n = 15
    X = np.random.rand(n, 2)
    y = np.random.randn(n)

    km = kernel_matrix(X, "gaussian")

    beta_small_lambda = krr_fit(km["U"], km["s"], y, lambda_reg=0.01)
    beta_large_lambda = krr_fit(km["U"], km["s"], y, lambda_reg=10.0)

    assert np.linalg.norm(beta_small_lambda) > np.linalg.norm(beta_large_lambda)


def test_krr_predict_basic():
    np.random.seed(42)
    n_train, n_test = 20, 10
    X_train = np.random.rand(n_train, 2)
    X_test = np.random.rand(n_test, 2)
    beta_hat = np.random.randn(n_train)

    y_pred = krr_predict(X_train, X_test, beta_hat, "gaussian", kernel_para=1.0)

    assert y_pred.shape == (n_test,)
    assert not np.any(np.isnan(y_pred))
    assert not np.any(np.isinf(y_pred))


def test_krr_predict_same_points():
    np.random.seed(42)
    X = np.random.rand(10, 2)
    beta_hat = np.ones(10)

    y_pred = krr_predict(X, X, beta_hat, "sobolev1")

    K = kernel_matrix(X, "sobolev1")["K"]
    expected = K @ beta_hat

    assert np.allclose(y_pred, expected)


def test_krr_pipeline():
    np.random.seed(42)
    n_train, n_test = 30, 10
    X_train = np.random.rand(n_train, 2)
    y_train = np.sin(2 * np.pi * X_train[:, 0]) + 0.1 * np.random.randn(n_train)
    X_test = np.random.rand(n_test, 2)

    km = kernel_matrix(X_train, "gaussian", kernel_para=5.0)
    beta = krr_fit(km["U"], km["s"], y_train, lambda_reg=0.01)
    y_pred = krr_predict(X_train, X_test, beta, "gaussian", kernel_para=5.0)

    assert y_pred.shape == (n_test,)
    assert not np.any(np.isnan(y_pred))


@pytest.mark.parametrize("kernel_type", ["sobolev1", "gaussian"])
def test_kernel_matrix_types(kernel_type):
    np.random.seed(42)
    X = np.random.rand(15, 3)

    result = kernel_matrix(X, kernel_type, kernel_para=1.5)

    assert result["K"].shape == (15, 15)
    assert np.allclose(result["K"], result["K"].T)
    assert np.all(result["s"] >= 0)
