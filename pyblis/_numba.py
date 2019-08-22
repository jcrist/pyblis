import numba as nb
from numba.extending import overload
from numba.errors import TypingError

from .core import gemm, syrk, mksymm, TypingContext


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

    def check_cast_scalar(self, name, val, dtype):
        if val == dtype:
            return
        elif (isinstance(dtype, nb.types.Float) and
              isinstance(val, (float, int, nb.types.Float, nb.types.Integer))):
            return
        elif (isinstance(dtype, nb.types.Complex) and
              isinstance(val, (float, int, complex, nb.types.Float,
                               nb.types.Integer, nb.types.Complex))):
            return
        else:
            self.error("`%s` should have dtype %r, got %r" % (name, dtype, val))
        return val


_CTX = NumbaTyping()


@overload(gemm)
def overload_gemm(a, b, out=None, a_trans=False, a_conj=False,
                  b_trans=False, b_conj=False, alpha=1.0,
                  beta=0.0, nthreads=-1):
    return _CTX.check_gemm(
        a, b, out, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads
    )[0]


@overload(syrk)
def overload_syrk(a, out=None, a_trans=False, a_conj=False, out_upper=False,
                  alpha=1.0, beta=0.0, nthreads=-1):
    return _CTX.check_syrk(
        a, out, a_trans, a_conj, out_upper, alpha, beta, nthreads
    )[0]


@overload(mksymm)
def overload_mksymm(a, upper=False, nthreads=-1):
    return _CTX.check_mksymm(a, upper, nthreads)
