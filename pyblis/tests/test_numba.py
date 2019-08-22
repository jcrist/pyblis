import pytest

nb = pytest.importorskip("numba")

import pyblis
import pyblis._numba

from .test_core import GEMMTests, SYRKTests, MKSYMMTests
from .utils import NumbaMixin


class TestGEMMNumba(NumbaMixin, GEMMTests):
    @classmethod
    def compile(cls):
        @nb.jit(nopython=True)
        def base(a, b):
            return pyblis.lib.gemm(a, b)

        @nb.jit(nopython=True)
        def full(a, b, out=None, a_trans=False, a_conj=False, b_trans=False,
                 b_conj=False, alpha=1.0, beta=0.0, nthreads=-1):
            return pyblis.lib.gemm(a, b, out=out, a_trans=a_trans, a_conj=a_conj,
                                   b_trans=b_trans, b_conj=b_conj, alpha=alpha,
                                   beta=beta, nthreads=nthreads)
        return base, full


class TestSYRKNumba(NumbaMixin, SYRKTests):
    @classmethod
    def compile(cls):
        @nb.jit(nopython=True)
        def base(a):
            return pyblis.lib.syrk(a)

        @nb.jit(nopython=True)
        def full(a, out=None, a_trans=False, a_conj=False, out_upper=False,
                 alpha=1.0, beta=0.0, nthreads=-1):
            return pyblis.lib.syrk(a, out=out, a_trans=a_trans, a_conj=a_conj,
                                   out_upper=out_upper, alpha=alpha, beta=beta,
                                   nthreads=nthreads)
        return base, full


class TestMKSYMMNumba(NumbaMixin, MKSYMMTests):
    @classmethod
    def compile(cls):
        @nb.jit(nopython=True)
        def full(a, upper=False, nthreads=-1):
            return pyblis.lib.mksymm(a, upper=upper, nthreads=nthreads)

        return full, full
