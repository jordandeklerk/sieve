"""Test basis functions."""

import numpy as np
import pytest

from sieve.basis import design_matrix, multi_psi, psi


def test_psi_sobolev1():
    assert psi(0.5, 1, "sobolev1") == 1.0
    assert np.isclose(psi(0.5, 2, "sobolev1"), np.sin(3 * np.pi * 0.5 / 2))
    assert np.isclose(psi(0.25, 3, "sobolev1"), np.sin(5 * np.pi * 0.25 / 2))


def test_psi_cosine():
    assert psi(0.5, 1, "cosine") == 1.0
    assert np.isclose(psi(0.5, 2, "cosine"), np.cos(np.pi * 0.5))
    assert np.isclose(psi(0.5, 3, "cosine"), np.cos(2 * np.pi * 0.5))


def test_psi_sine():
    assert np.isclose(psi(0.5, 1, "sine"), np.sin(np.pi * 0.5 / 2))
    assert np.isclose(psi(0.5, 2, "sine"), np.sin(3 * np.pi * 0.5 / 2))


def test_psi_tri():
    assert psi(0.5, 1, "tri") == 1.0
    assert psi(0.5, 2, "tri") == 0.5
    assert np.isclose(psi(0.5, 3, "tri"), np.sqrt(2) * np.cos(2 * np.pi * 0.5))
    assert np.isclose(psi(0.5, 4, "tri"), np.sqrt(2) * np.sin(2 * np.pi * 0.5))


def test_psi_poly():
    assert psi(0.5, 1, "poly") == 1.0
    assert psi(0.5, 2, "poly") == 0.5
    assert psi(0.5, 3, "poly") == 0.25
    assert psi(2.0, 4, "poly") == 8.0


def test_psi_legendre():
    x = 0.75
    x_transformed = (x - 0.5) * 2

    assert psi(x, 1, "legendre") == 1.0
    assert np.isclose(psi(x, 2, "legendre"), x_transformed)
    assert np.isclose(psi(x, 3, "legendre"), (3 * x_transformed**2 - 1) / 2)

    assert psi(0.5, 2, "legendre") == 0.0
    assert psi(0.5, 1, "legendre") == 1.0


def test_psi_legendre_max_order():
    for j in range(1, 10):
        result = psi(0.7, j, "legendre")
        assert isinstance(result, float)

    with pytest.raises(ValueError, match="Legendre polynomial of order 10 not implemented"):
        psi(0.5, 10, "legendre")


def test_psi_invalid_basis_type():
    with pytest.raises(ValueError, match="Unknown basis type: invalid"):
        psi(0.5, 1, "invalid")


def test_multi_psi_basic():
    x = np.array([0.5, 0.3])
    index = np.array([2, 3])

    expected = psi(0.5, 2, "sobolev1") * psi(0.3, 3, "sobolev1")
    result = multi_psi(x, index, "sobolev1")
    assert np.isclose(result, expected)


def test_multi_psi_dimension_mismatch():
    x = np.array([0.5, 0.3])
    index = np.array([2, 3, 4])

    with pytest.raises(ValueError, match="x and index must have the same dimension"):
        multi_psi(x, index, "sobolev1")


def test_multi_psi_different_basis_types():
    x = np.array([0.25, 0.75, 0.5])
    index = np.array([1, 2, 3])

    for basis_type in ["sobolev1", "cosine", "sine", "tri", "poly", "legendre"]:
        result = multi_psi(x, index, basis_type)
        expected = psi(0.25, 1, basis_type) * psi(0.75, 2, basis_type) * psi(0.5, 3, basis_type)
        assert np.isclose(result, expected)


def test_design_matrix_basic():
    np.random.seed(42)
    X = np.random.rand(10, 2)
    index_matrix = np.array([[1, 1], [2, 1], [1, 2]])

    Phi = design_matrix(X, 3, "cosine", index_matrix)

    assert Phi.shape == (10, 3)
    assert np.all(Phi[:, 0] == 1.0)


def test_design_matrix_invalid_dimensions():
    X = np.random.rand(10, 2)
    index_matrix = np.array([[1, 1, 1], [2, 1, 1]])

    with pytest.raises(ValueError, match="index_matrix shape .* doesn't match expected"):
        design_matrix(X, 2, "cosine", index_matrix)


@pytest.mark.parametrize("basis_type", ["cosine", "sine", "polytri", "poly"])
def test_design_matrix_different_basis_types(basis_type):
    np.random.seed(42)
    X = np.random.rand(5, 2)
    index_matrix = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])

    Phi = design_matrix(X, 4, basis_type, index_matrix)

    assert Phi.shape == (5, 4)
    assert not np.any(np.isnan(Phi))
    assert not np.any(np.isinf(Phi))


def test_design_matrix_manual_verification():
    X = np.array([[0.5, 0.5]])
    index_matrix = np.array([[1, 1], [2, 2]])

    Phi = design_matrix(X, 2, "cosine", index_matrix)

    assert Phi.shape == (1, 2)
    assert Phi[0, 0] == 1.0
    expected_val = np.cos(np.pi * 0.5) * np.cos(np.pi * 0.5)
    assert np.isclose(Phi[0, 1], expected_val)


def test_design_matrix_unsupported_basis_type():
    X = np.random.rand(5, 2)
    index_matrix = np.array([[1, 1]])

    with pytest.raises(ValueError, match="Unsupported basis type for optimized computation"):
        design_matrix(X, 1, "sobolev1", index_matrix)
