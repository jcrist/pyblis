import numba as nb
from numba.extending import overload
from numba.errors import TypingError

from .core import gemm, TypingContext


class NumbaTyping(TypingContext):
    prefixes = {nb.float32: 's',
                nb.float64: 'd',
                nb.complex64: 'c',
                nb.complex128: 'z'}

    def error(self, msg):
        raise TypingError(msg)

    def is_bool(self, a):
        return isinstance(a, (bool, nb.types.Boolean))

    def is_int(self, a):
        return isinstance(a, (int, nb.types.Integer))

    def is_none(self, a):
        return a is None or isinstance(a, nb.types.NoneType)

    def is_ndarray(self, a):
        return isinstance(a, nb.types.Array)

    def is_contig(self, a):
        return a.is_contig


_CTX = NumbaTyping()


@overload(gemm)
def overload_gemm(a, b, out=None, a_trans=False, a_conj=False,
                  b_trans=False, b_conj=False, alpha=1.0,
                  beta=0.0, nthreads=-1):
    _, gemm = _CTX.check_gemm(a, b, out, a_trans, a_conj, b_trans, b_conj,
                              alpha, beta, nthreads)
    return gemm
