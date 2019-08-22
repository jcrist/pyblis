import numpy as np
from numpy.testing import assert_allclose

import pyblis

from .utils import Base, all_dtypes


class DotTests(Base):
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
    def test_transpose(self, dtype):
        a, _ = self.a_b(dtype)
        res = self.call_base(a, a.T)
        sol = a.dot(a.T)
        assert_allclose(res, sol)


class TestDot(DotTests):
    error_cls = TypeError

    def call(self, *args, **kwargs):
        return pyblis.dot(*args, **kwargs)
