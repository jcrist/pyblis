import pytest

nb = pytest.importorskip("numba")

import pyblis
import pyblis._numba

from .test_core import GEMMTests, SYRKTests, MKSYMMTests


class TestGEMMNumba(GEMMTests):
    error_cls = nb.errors.TypingError

    @classmethod
    def setup_class(cls):
        @nb.jit(nopython=True)
        def base(a, b):
            return pyblis.gemm(a, b)

        @nb.jit(nopython=True)
        def full(a, b, out=None, a_trans=False, a_conj=False, b_trans=False,
                 b_conj=False, alpha=1.0, beta=0.0, nthreads=-1):
            return pyblis.gemm(a, b, out=out, a_trans=a_trans, a_conj=a_conj,
                               b_trans=b_trans, b_conj=b_conj, alpha=alpha,
                               beta=beta, nthreads=nthreads)

        cls.base = staticmethod(base)
        cls.full = staticmethod(full)

    def call(self, *args, **kwargs):
        return self.full(*args, **kwargs)

    def call_base(self, *args, **kwargs):
        return self.base(*args, **kwargs)


class TestSYRKNumba(SYRKTests):
    error_cls = nb.errors.TypingError

    @classmethod
    def setup_class(cls):
        @nb.jit(nopython=True)
        def base(a):
            return pyblis.syrk(a)

        @nb.jit(nopython=True)
        def full(a, out=None, a_trans=False, a_conj=False, out_upper=False,
                 alpha=1.0, beta=0.0, nthreads=-1):
            return pyblis.syrk(a, out=out, a_trans=a_trans, a_conj=a_conj,
                               out_upper=out_upper, alpha=alpha, beta=beta,
                               nthreads=nthreads)

        cls.base = staticmethod(base)
        cls.full = staticmethod(full)

    def call(self, *args, **kwargs):
        return self.full(*args, **kwargs)

    def call_base(self, *args, **kwargs):
        return self.base(*args, **kwargs)


class TestMKSYMMNumba(MKSYMMTests):
    error_cls = nb.errors.TypingError

    @classmethod
    def setup_class(cls):
        @nb.jit(nopython=True)
        def full(a, upper=False, nthreads=-1):
            return pyblis.mksymm(a, upper=upper, nthreads=nthreads)

        cls.full = staticmethod(full)

    def call(self, *args, **kwargs):
        return self.full(*args, **kwargs)
