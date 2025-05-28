"""Test preprocessing functions."""

import numpy as np
import pytest

from sieve.preprocessing import (
    create_index_matrix,
    normalize_X,
    sieve_preprocess,
)


def test_normalize_x_basic():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    result = normalize_X(X)

    assert isinstance(result, dict)
    assert "X" in result
    assert "norm_para" in result
    assert result["X"].min() >= 0
    assert result["X"].max() <= 1
    assert result["X"].shape == X.shape
    assert result["norm_para"].shape == (2, X.shape[1])


def test_normalize_x_1d_input():
    X = np.array([1, 2, 3, 4, 5])
    result = normalize_X(X)

    assert result["X"].shape == (5, 1)
    assert result["norm_para"].shape == (2, 1)


def test_normalize_x_with_norm_para():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    norm_para = np.array([[0, 1], [10, 5]])

    result = normalize_X(X, norm_para=norm_para)

    expected_col1 = np.array([0.1, 0.3, 0.5])
    expected_col2 = np.array([0.25, 0.75, 1.0])

    np.testing.assert_allclose(result["X"][:, 0], expected_col1)
    np.testing.assert_allclose(result["X"][:, 1], expected_col2)


def test_normalize_x_constant_feature_warning():
    X = np.array([[1.0, 2], [1.0 + 1e-15, 4], [1.0, 6], [1.0 - 1e-15, 8], [1.0, 10]])

    with pytest.warns(UserWarning, match="near-constant values"):
        result = normalize_X(X, lower_q=0.1, upper_q=0.9)

    assert np.allclose(result["X"][:, 0], 0.5)


def test_normalize_x_quantile_parameters():
    np.random.seed(42)
    X = np.random.randn(100, 2) * 10

    X[0, 0] = 1000
    X[-1, 0] = -1000

    result = normalize_X(X, lower_q=0.05, upper_q=0.95)

    assert result["X"].min() == 0
    assert result["X"].max() == 1


def test_normalize_x_invalid_norm_para_shape():
    X = np.array([[1, 2], [3, 4]])
    norm_para = np.array([[1, 2, 3]])

    with pytest.raises(ValueError, match="norm_para shape"):
        normalize_X(X, norm_para=norm_para)


def test_create_index_matrix_with_basis_n():
    index_matrix = create_index_matrix(xdim=2, basis_n=5)

    assert index_matrix.shape == (5, 3)
    assert np.all(index_matrix[0, 1:] == 1)

    products = np.prod(index_matrix[:, 1:], axis=1)
    np.testing.assert_array_equal(index_matrix[:, 0], products)


def test_create_index_matrix_with_maxj():
    index_matrix = create_index_matrix(xdim=2, maxj=3)

    assert np.all(index_matrix[:, 0] <= 3)

    unique_products = np.unique(index_matrix[:, 0])
    assert 1 in unique_products
    assert 2 in unique_products
    assert 3 in unique_products


def test_create_index_matrix_interaction_order_1():
    index_matrix = create_index_matrix(xdim=3, basis_n=10, interaction_order=1)

    non_one_counts = np.sum(index_matrix[:, 1:] > 1, axis=1)
    assert np.all(non_one_counts <= 1)


def test_create_index_matrix_interaction_order_2():
    index_matrix = create_index_matrix(xdim=3, basis_n=20, interaction_order=2)

    non_one_counts = np.sum(index_matrix[:, 1:] > 1, axis=1)
    assert np.all(non_one_counts <= 2)


def test_create_index_matrix_interaction_order_exceeds_xdim():
    index_matrix = create_index_matrix(xdim=2, basis_n=10, interaction_order=5)

    non_one_counts = np.sum(index_matrix[:, 1:] > 1, axis=1)
    assert np.all(non_one_counts <= 2)


def test_create_index_matrix_invalid_inputs():
    with pytest.raises(ValueError, match="Must specify either"):
        create_index_matrix(xdim=2)

    with pytest.raises(ValueError, match="xdim must be positive"):
        create_index_matrix(xdim=0, basis_n=10)

    with pytest.raises(ValueError, match="interaction_order must be positive"):
        create_index_matrix(xdim=2, basis_n=10, interaction_order=0)


def test_create_index_matrix_deterministic():
    index1 = create_index_matrix(xdim=3, basis_n=15, interaction_order=2)
    index2 = create_index_matrix(xdim=3, basis_n=15, interaction_order=2)

    np.testing.assert_array_equal(index1, index2)


def test_sieve_preprocess_basic():
    np.random.seed(42)
    X = np.random.rand(50, 3)

    result = sieve_preprocess(X, basis_n=20)

    assert isinstance(result, dict)
    required_keys = ["Phi", "X", "basis_type", "index_matrix", "basis_n", "norm_para"]
    for key in required_keys:
        assert key in result

    assert result["Phi"].shape == (50, 20)
    assert result["X"].shape == (50, 3)
    assert result["index_matrix"].shape == (20, 4)
    assert result["norm_para"].shape == (2, 3)
    assert result["basis_n"] == 20


def test_sieve_preprocess_default_basis_n_warning():
    X = np.random.rand(100, 2)

    with pytest.warns(UserWarning, match="did not specify number of basis functions"):
        result = sieve_preprocess(X)

    assert result["basis_n"] == 100


@pytest.mark.parametrize("basis_type", ["cosine", "sine", "polytri", "poly"])
def test_sieve_preprocess_basis_types(basis_type):
    X = np.random.rand(30, 2)

    result = sieve_preprocess(X, basis_n=10, basis_type=basis_type)
    assert result["basis_type"] == basis_type
    assert result["Phi"].shape == (30, 10)


def test_sieve_preprocess_invalid_basis_type():
    X = np.random.rand(30, 2)

    with pytest.raises(ValueError, match="not supported by the optimized"):
        sieve_preprocess(X, basis_n=10, basis_type="legendre")


def test_sieve_preprocess_1d_input():
    X = np.random.rand(50)

    result = sieve_preprocess(X, basis_n=15)

    assert result["Phi"].shape == (50, 15)
    assert result["X"].shape == (50, 1)


def test_sieve_preprocess_no_normalization():
    X = np.random.rand(30, 2)
    X_original = X.copy()

    result = sieve_preprocess(X, basis_n=10, norm_feature=False)

    np.testing.assert_array_equal(result["X"], X_original)
    assert result["norm_para"] is None


def test_sieve_preprocess_with_norm_para():
    X_train = np.random.rand(50, 2)
    train_result = sieve_preprocess(X_train, basis_n=10)

    X_test = np.random.rand(20, 2)
    test_result = sieve_preprocess(
        X_test, basis_n=10, norm_para=train_result["norm_para"], index_matrix=train_result["index_matrix"]
    )

    np.testing.assert_array_equal(test_result["norm_para"], train_result["norm_para"])


def test_sieve_preprocess_with_index_matrix():
    X = np.random.rand(30, 2)

    index_matrix = create_index_matrix(xdim=2, basis_n=8)

    result = sieve_preprocess(X, index_matrix=index_matrix)

    np.testing.assert_array_equal(result["index_matrix"], index_matrix)
    assert result["Phi"].shape == (30, 8)


def test_sieve_preprocess_interaction_orders():
    X = np.random.rand(40, 3)

    result1 = sieve_preprocess(X, basis_n=15, interaction_order=1)
    index1 = result1["index_matrix"]
    non_one_counts1 = np.sum(index1[:, 1:] > 1, axis=1)
    assert np.all(non_one_counts1 <= 1)

    result2 = sieve_preprocess(X, basis_n=15, interaction_order=2)
    index2 = result2["index_matrix"]
    non_one_counts2 = np.sum(index2[:, 1:] > 1, axis=1)
    assert np.any(non_one_counts2 > 1)


def test_sieve_preprocess_basis_n_validation():
    X = np.random.rand(30, 2)

    with pytest.raises(ValueError, match="basis_n must be > 1"):
        sieve_preprocess(X, basis_n=1)


def test_sieve_preprocess_array_like_input():
    X_list = [[1, 2], [3, 4], [5, 6]]
    result = sieve_preprocess(X_list, basis_n=5)
    assert result["Phi"].shape == (3, 5)

    X_tuple = ((1, 2), (3, 4), (5, 6))
    result = sieve_preprocess(X_tuple, basis_n=5)
    assert result["Phi"].shape == (3, 5)


def test_sieve_preprocess_consistent_output():
    np.random.seed(42)
    X = np.random.rand(25, 2)

    result1 = sieve_preprocess(X, basis_n=10, interaction_order=2)
    result2 = sieve_preprocess(X, basis_n=10, interaction_order=2)

    np.testing.assert_array_equal(result1["Phi"], result2["Phi"])
    np.testing.assert_array_equal(result1["index_matrix"], result2["index_matrix"])
