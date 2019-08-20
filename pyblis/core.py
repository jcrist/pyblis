import numpy as np
import numba as nb
from numba.types import float32, float64, complex64, complex128, boolean, Optional, int32

from . import lib

__all__ = ("gemm",)


afloat32 = nb.types.Array(float32, 2, 'C')
afloat64 = nb.types.Array(float64, 2, 'C')
acomplex64 = nb.types.Array(complex64, 2, 'C')
acomplex128 = nb.types.Array(complex128, 2, 'C')


@nb.generated_jit(nopython=True, nogil=True)
def lib_gemm(a, b, out, m, n, k, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads):
    if a.dtype in (nb.float32, nb.float64):
        lib_gemm = lib.sgemm if a.dtype == nb.float32 else lib.dgemm

        def f(a, b, out, m, n, k, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads):
            lib_gemm(
                a_trans, a_conj,
                b_trans, b_conj,
                m, n, k,
                alpha,
                a.ctypes, a.shape[1], 1,
                b.ctypes, b.shape[1], 1,
                beta,
                out.ctypes, out.shape[1], 1,
                nthreads
            )

    elif a.dtype in (nb.complex64, nb.complex128):
        lib_gemm = lib.cgemm if a.dtype == nb.complex64 else lib.zgemm

        def f(a, b, out, m, n, k, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads):
            lib_gemm(
                a_trans, a_conj,
                b_trans, b_conj,
                m, n, k,
                alpha.real, alpha.imag,
                a.ctypes, a.shape[1], 1,
                b.ctypes, b.shape[1], 1,
                beta.real, beta.imag,
                out.ctypes, out.shape[1], 1,
                nthreads
            )

    else:
        raise TypeError
    return f


def _gemm_signatures():
    typs = [(afloat32, float32), (afloat64, float64),
            (acomplex64, complex64), (acomplex128, complex128)]
    return [aT(aT, aT, Optional(aT), boolean, boolean, boolean, boolean, T, T, int32)
            for (aT, T) in typs]


@nb.jit(_gemm_signatures(), nopython=True, nogil=True)
def _gemm(a, b, out, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads):
    m = a.shape[0] if not a_trans else a.shape[1]
    n = b.shape[1] if not b_trans else b.shape[0]
    k = a.shape[1] if not a_trans else a.shape[0]

    if out is None:
        out = np.zeros((m, n), dtype=a.dtype)
    elif out.shape[0] != m or out.shape[1] != n:
        raise ValueError("Output shape mismatch")

    lib_gemm(a, b, out, m, n, k,
             a_trans, a_conj,
             b_trans, b_conj,
             alpha, beta,
             nthreads)
    return out


@nb.jit(nopython=True, nogil=True)
def gemm(a, b, out=None, a_trans=False, a_conj=False,
         b_trans=False, b_conj=False, alpha=1.0, beta=0.0, nthreads=-1):
    """Multiply two matrices.

    Solves ``out = alpha * op_a(a).dot(beta * op_b(b))``.

    Where ``op_a`` and ``op_b`` indicate any transpose/conjugate operation
    specified on ``a`` or ``b`` respectively.

    Parameters
    ----------
    a, b : np.ndarray[T]
        Two identically typed arrays, where ``T`` is one of
        (float64, float32, complex128, complex64).
    out : np.ndarray[T], optional
        An optional output array, must match the type of the input arrays. If
        not provided, a new array will be allocated.
    a_trans, b_trans : bool, optional
        Whether to transpose ``a`` and ``b`` respectively. Default is False.
    a_conj, b_conj : bool, optional
        Whether to conjugate ``a`` and ``b`` respectively. Default is False.
    alpha : T
        The ``alpha`` factor. Default is 1.
    beta : T
        The ``beta`` factor. Default is 0.
    nthreads : int
        The number of threads to use. Defaults to deriving from environment
        variables (e.g. ``BLIS_NUM_THREADS``).

    Returns
    -------
    out : np.ndarray[T]
    """
    return _gemm(a, b, out, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads)
