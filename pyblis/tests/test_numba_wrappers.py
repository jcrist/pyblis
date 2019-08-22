import pytest

nb = pytest.importorskip("numba")

import pyblis
import pyblis._numba

from .test_wrappers import DotTests
from .utils import NumbaMixin


class TestDotNumba(NumbaMixin, DotTests):
    @classmethod
    def compile(cls):
        @nb.jit(nopython=True)
        def base(a, b):
            return pyblis.dot(a, b)

        @nb.jit(nopython=True)
        def full(a, b, out=None, nthreads=-1):
            return pyblis.dot(a, b, out=out, nthreads=nthreads)

        return base, full
