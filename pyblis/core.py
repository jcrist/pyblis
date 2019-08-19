import numpy as np
import numba as nb
from numba import types
from numba.errors import TypingError

from . import lib

__all__ = ("gemm",)


def check_is_2d_contig_array_of_type(name, a, dtype):
    if not isinstance(a, types.Array):
        raise TypingError("`%s` must be a NumPy ndarray" % name)
    elif not isinstance(a.dtype, types.Float):
        raise TypingError("`%s` must a floating-point dtype" % name)
    elif not a.dtype == dtype:
        raise TypingError("Non-uniform dtypes found, `%s`'s dtype is %r not %r"
                          % (name, a.dtype, dtype))
    elif not a.ndim == 2:
        raise TypingError("`%s` must be a 2 dimensional array" % name)
    elif not a.is_contig:
        raise TypingError("`%s` must be contiguous" % name)


@nb.generated_jit(nopython=True, nogil=True)
def gemm(a, b, out=None, trans_a=False, trans_b=False, alpha=1., beta=1.):
    """Multiply two matrices.

    Solves ``out = alpha * trans_a(a).dot(beta * trans_b(b))``.

    Where ``trans_a`` and ``trans_b`` indicate whether to take the transpose of
    ``a`` or ``b`` respectively.

    Parameters
    ----------
    a, b : np.ndarray[T]
        Two identically typed arrays, where ``T`` is one of (float64, float32).
    out : np.ndarray[T], optional
        An optional output array, must match the type of the input arrays. If
        not provided, a new array will be allocated.
    trans_a, trans_b : bool, optional
        Whether to transpose ``a`` and ``b`` respectively. Default is False.
    alpha, beta : float
        The ``alpha`` and ``beta`` scalars respectively. Default is 1.

    Returns
    -------
    out : np.ndarray[T]
    """
    arrays = [("a", a), ("b", b)]
    if not isinstance(out, (types.NoneType, types.Omitted)):
        arrays.append(("out", out))

    for param, x in arrays:
        check_is_2d_contig_array_of_type(param, x, a.dtype)

    if a.dtype == types.float32:
        lib_gemm = lib.sgemm
    else:
        lib_gemm = lib.dgemm

    for param, typ in [("trans_a", trans_a), ("trans_b", trans_b)]:
        if not isinstance(typ, (types.Boolean, types.Omitted)):
            raise TypingError("%s must be a boolean" % param)

    for param, typ in [("alpha", alpha), ("beta", beta)]:
        if not isinstance(typ, (types.Float, types.Omitted)):
            raise TypingError("%s must be a float" % param)

    def gemm(a, b, out=None, trans_a=False, trans_b=False, alpha=1., beta=1.):
        nM = a.shape[0] if not trans_a else a.shape[1]
        nK = a.shape[1] if not trans_a else a.shape[0]
        nN = b.shape[1] if not trans_b else b.shape[0]

        if out is None:
            out = np.zeros((nM, nN), dtype=a.dtype)
        elif out.shape[0] != nM or out.shape[1] != nN:
            raise ValueError("Output shape mismatch")

        lib_gemm(
            trans_a,
            trans_b,
            nM, nN, nK,
            alpha,
            a.ctypes, a.shape[1], 1,
            b.ctypes, b.shape[1], 1,
            beta,
            out.ctypes, out.shape[1], 1
        )
        return out

    return gemm
