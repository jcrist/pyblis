import pytest

import numpy as np
from numpy.testing import assert_allclose

import pyblis

all_dtypes = pytest.mark.parametrize('dtype', ['f4', 'f8', 'c8', 'c16'])


@all_dtypes
def test_gemm(dtype):
    a = np.random.normal(size=(3, 4)).astype(dtype)
    b = np.random.normal(size=(4, 5)).astype(dtype)
    alpha = 4.5
    beta = 3.5
    if np.issubdtype(dtype, np.complexfloating):
        a += np.random.normal(size=a.shape) * 1j
        b += np.random.normal(size=b.shape) * 1j
        alpha += 1.3j
        beta += 2.3j

    # Standard call
    res = pyblis.gemm(a, b)
    sol = a.dot(b)
    assert_allclose(res, sol)

    # With out
    out = np.zeros(shape=(3, 5), dtype=dtype)
    res = pyblis.gemm(a, b, out=out)
    assert res is out
    assert_allclose(res, sol)

    # With alpha
    res = pyblis.gemm(a, b, alpha=alpha)
    assert_allclose(res, alpha * sol)

    # With beta
    out = np.ones(shape=(3, 5), dtype=dtype)
    pyblis.gemm(a, b, out=out, beta=beta)
    assert_allclose(out, beta + sol)

    # With transpose
    sol = pyblis.gemm(b, a, a_trans=True, b_trans=True)
    assert_allclose(sol, b.T.dot(a.T))

    # With conjugate
    sol = pyblis.gemm(a, b, a_conj=True, b_conj=True)
    assert_allclose(sol, a.conj().dot(b.conj()))

    # With transpose and conjugate
    sol = pyblis.gemm(b, a, a_trans=True, a_conj=True, b_trans=True, b_conj=True)
    assert_allclose(sol, b.conj().T.dot(a.conj().T))
