import pytest

import numpy as np
from numpy.testing import assert_allclose

import pyblis

all_dtypes = pytest.mark.parametrize('dtype', ['f4', 'f8', 'c8', 'c16'])


class Base(object):
    def rand(self, dtype, shape=()):
        a = np.random.normal(size=shape).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            a += np.random.normal(size=a.shape) * 1j
        return a if a.shape else a.reshape((1,))[0]

    def call_base(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class GEMMTests(Base):
    @all_dtypes
    def a_b(self, dtype):
        a = self.rand(dtype, (3, 4))
        b = self.rand(dtype, (4, 5))
        return a, b

    @all_dtypes
    def test_base(self, dtype):
        a, b = self.a_b(dtype)
        res = self.call_base(a, b)
        sol = a.dot(b)
        assert_allclose(res, sol)

    @all_dtypes
    def test_with_out(self, dtype):
        a, b = self.a_b(dtype)
        out = np.zeros(shape=(3, 5), dtype=dtype)
        res = self.call(a, b, out=out)
        assert res is out
        assert_allclose(res, a.dot(b))

    @all_dtypes
    def test_with_alpha(self, dtype):
        a, b = self.a_b(dtype)
        alpha = self.rand(dtype)
        res = self.call(a, b, alpha=alpha)
        assert_allclose(res, alpha * a.dot(b))

    @all_dtypes
    def test_with_beta(self, dtype):
        a, b = self.a_b(dtype)
        beta = self.rand(dtype)
        out = np.ones(shape=(3, 5), dtype=dtype)
        self.call(a, b, out=out, beta=beta)
        assert_allclose(out, beta + a.dot(b))

    @all_dtypes
    def test_with_transpose(self, dtype):
        a, b = self.a_b(dtype)
        sol = self.call(b, a, a_trans=True, b_trans=True)
        assert_allclose(sol, b.T.dot(a.T))

    @all_dtypes
    def test_with_conjugate(self, dtype):
        a, b = self.a_b(dtype)
        sol = self.call(a, b, a_conj=True, b_conj=True)
        assert_allclose(sol, a.conj().dot(b.conj()))

    @all_dtypes
    def test_with_transpose_conjugate(self, dtype):
        a, b = self.a_b(dtype)
        sol = self.call(b, a, a_trans=True, a_conj=True, b_trans=True, b_conj=True)
        assert_allclose(sol, b.conj().T.dot(a.conj().T))

    @all_dtypes
    def test_with_strides(self, dtype):
        a, b = self.a_b(dtype)

        res = self.call(a, a.T)
        assert_allclose(res, a.dot(a.T))

        res = self.call(a[::2], b[:, ::2])
        assert_allclose(res, a[::2].dot(b[:, ::2]))

    def test_errors_unsupported_dtype(self):
        a, b = self.a_b('i4')
        with pytest.raises(self.error_cls) as exc:
            self.call(a, b)
        assert "No implementation" in str(exc.value)

    def test_errors_mismatch_dtypes(self):
        a, b = self.a_b('f4')
        b = b.astype('f8')
        with pytest.raises(self.error_cls) as exc:
            self.call(a, b)
        assert "Non-uniform" in str(exc.value)

    def test_errors_not_ndarray(self):
        with pytest.raises(self.error_cls) as exc:
            self.call(1, 2)
        assert "NumPy ndarray" in str(exc.value)

    def test_errors_wrong_dimensions(self):
        with pytest.raises(self.error_cls) as exc:
            self.call(np.array([1, 2, 3.]), np.array([[1.]]))
        assert "2 dimensional" in str(exc.value)

    def test_errors_bad_flags(self):
        a, b = self.a_b('f4')
        with pytest.raises(self.error_cls) as exc:
            self.call(a, b, a_trans=1)
        assert "bool" in str(exc.value)

    def test_errors_bad_nthreads(self):
        a, b = self.a_b('f4')
        with pytest.raises(self.error_cls) as exc:
            self.call(a, b, nthreads='oops')
        assert "nthreads" in str(exc.value)

    def test_error_shape_mismatch(self):
        # Bad b
        a = self.rand('f4', (3, 4))
        b = self.rand('f4', (3, 5))
        with pytest.raises(ValueError) as exc:
            self.call(a, b)
        assert "shape mismatch" in str(exc.value)

        # Bad out
        b = self.rand('f4', (4, 5))
        out = np.zeros_like(a)
        with pytest.raises(ValueError) as exc:
            self.call(a, b, out=out)
        assert "shape mismatch" in str(exc.value)


class TestGEMMCtypes(GEMMTests):
    error_cls = TypeError

    def call(self, *args, **kwargs):
        return pyblis.gemm(*args, **kwargs)
