import pytest

import numpy as np
from numpy.testing import assert_allclose

import pyblis

from .utils import Base, all_dtypes


class GEMMTests(Base):
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
        return pyblis.lib.gemm(*args, **kwargs)


class SYRKTests(Base):
    def a(self, dtype):
        return self.rand(dtype, (3, 4))

    def sol(self, a, out=None, a_trans=False, a_conj=False, out_upper=False,
            alpha=1.0, beta=0.0):
        if a_conj:
            a = a.conj()
        aa = a.T.dot(a) if a_trans else a.dot(a.T)
        aa = np.triu(aa) if out_upper else np.tril(aa)
        alpha_aa = alpha * aa
        if out is not None:
            inds = np.triu_indices_from(out) if out_upper else np.tril_indices_from(out)
            np.multiply.at(out, inds, beta)
            out += alpha_aa
        else:
            out = alpha_aa
        return out

    @all_dtypes
    def test_base(self, dtype):
        a = self.a(dtype)
        res = self.call_base(a)
        sol = self.sol(a)
        assert_allclose(res, sol)

    @all_dtypes
    def test_with_out(self, dtype):
        a = self.a(dtype)
        out = np.zeros(shape=(3, 3), dtype=dtype)
        res = self.call(a, out=out)
        sol = self.sol(a)
        assert res is out
        assert_allclose(res, sol)

    @all_dtypes
    def test_with_alpha(self, dtype):
        a = self.a(dtype)
        alpha = self.rand(dtype)
        res = self.call(a, alpha=alpha)
        sol = self.sol(a, alpha=alpha)
        assert_allclose(res, sol)

    @all_dtypes
    def test_with_beta(self, dtype):
        a = self.a(dtype)
        beta = self.rand(dtype)
        res = np.ones(shape=(3, 3), dtype=dtype)
        sol = np.ones(shape=(3, 3), dtype=dtype)
        self.call(a, out=res, beta=beta)
        self.sol(a, out=sol, beta=beta)
        assert_allclose(res, sol)

    @all_dtypes
    def test_with_transpose(self, dtype):
        a = self.a(dtype)
        res = self.call(a, a_trans=True)
        sol = self.sol(a, a_trans=True)
        assert_allclose(res, sol)

    @all_dtypes
    def test_with_conjugate(self, dtype):
        a = self.a(dtype)
        res = self.call(a, a_conj=True)
        sol = self.sol(a, a_conj=True)
        assert_allclose(res, sol)

    @all_dtypes
    def test_with_transpose_conjugate(self, dtype):
        a = self.a(dtype)
        res = self.call(a, a_trans=True, a_conj=True)
        sol = self.sol(a, a_trans=True, a_conj=True)
        assert_allclose(res, sol)

    @all_dtypes
    def test_with_strides(self, dtype):
        a = self.a(dtype)
        res = self.call(a.T)
        sol = self.sol(a.T)
        assert_allclose(res, sol)

        res = self.call(a[::2])
        sol = self.sol(a[::2])
        assert_allclose(res, sol)

    def test_errors_unsupported_dtype(self):
        a = self.a('i4')
        with pytest.raises(self.error_cls) as exc:
            self.call(a)
        assert "No implementation" in str(exc.value)

    def test_errors_mismatch_dtypes(self):
        a = self.a('f4')
        out = np.zeros((3, 3), dtype='f8')
        with pytest.raises(self.error_cls) as exc:
            self.call(a, out=out)
        assert "Non-uniform" in str(exc.value)

    def test_errors_not_ndarray(self):
        with pytest.raises(self.error_cls) as exc:
            self.call(1)
        assert "NumPy ndarray" in str(exc.value)

    def test_errors_wrong_dimensions(self):
        with pytest.raises(self.error_cls) as exc:
            self.call(np.array([1, 2, 3.]))
        assert "2 dimensional" in str(exc.value)

    def test_errors_bad_flags(self):
        a = self.a('f4')
        with pytest.raises(self.error_cls) as exc:
            self.call(a, a_trans=1)
        assert "bool" in str(exc.value)

    def test_errors_bad_nthreads(self):
        a = self.a('f4')
        with pytest.raises(self.error_cls) as exc:
            self.call(a, nthreads='oops')
        assert "nthreads" in str(exc.value)

    def test_error_shape_mismatch(self):
        # Bad out
        a = self.a('f8')
        out = np.zeros((3, 4), dtype='f8')
        with pytest.raises(ValueError) as exc:
            self.call(a, out=out)
        assert "shape mismatch" in str(exc.value)


class TestSYRKCtypes(SYRKTests):
    error_cls = TypeError

    def call(self, *args, **kwargs):
        return pyblis.lib.syrk(*args, **kwargs)


class MKSYMMTests(Base):
    def a(self, dtype):
        return self.rand(dtype, (3, 3))

    def sol(self, a, upper=False):
        mask = (np.tril if upper else np.triu)(np.ones(a.shape, dtype='b'))
        return np.where(mask, a.T, a)

    @pytest.mark.parametrize('upper', [False, True])
    @all_dtypes
    def test_mksymm(self, dtype, upper):
        a = self.a(dtype)
        sol = self.sol(a, upper=upper)
        res = self.call(a, upper=upper)
        assert_allclose(res, sol)

    @pytest.mark.parametrize('upper', [False, True])
    @all_dtypes
    def test_with_strides(self, dtype, upper):
        a = self.a(dtype)
        sol = self.sol(a.T, upper=upper)
        res = self.call(a.T, upper=upper)
        assert_allclose(res, sol)

        a = self.a(dtype)
        sol = self.sol(a[::2, ::2], upper=upper)
        res = self.call(a[::2, ::2], upper=upper)
        assert_allclose(res, sol)

    def test_errors_unsupported_dtype(self):
        a = self.a('i4')
        with pytest.raises(self.error_cls) as exc:
            self.call(a)
        assert "No implementation" in str(exc.value)

    def test_errors_not_ndarray(self):
        with pytest.raises(self.error_cls) as exc:
            self.call(1)
        assert "NumPy ndarray" in str(exc.value)

    def test_errors_wrong_dimensions(self):
        with pytest.raises(self.error_cls) as exc:
            self.call(np.array([1, 2, 3.]))
        assert "2 dimensional" in str(exc.value)

    def test_error_not_square(self):
        a = np.zeros((3, 4), dtype='f8')
        with pytest.raises(ValueError) as exc:
            self.call(a)
        assert "square" in str(exc.value)


class TestMKSYMMCtypes(MKSYMMTests):
    error_cls = TypeError

    def call(self, *args, **kwargs):
        return pyblis.lib.mksymm(*args, **kwargs)
