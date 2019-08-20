import numpy as np

from . import lib

__all__ = ("gemm",)


class TypingContext(object):
    # Subclasses should define prefixes mapping, and override methods below
    def error(self, msg):
        raise NotImplementedError

    def is_bool(self, a):
        raise NotImplementedError

    def is_int(self, a):
        raise NotImplementedError

    def is_none(self, a):
        raise NotImplementedError

    def is_ndarray(self, a):
        raise NotImplementedError

    def is_contig(self, a):
        raise NotImplementedError

    def ndim(self, a):
        return a.ndim

    def dtype(self, a):
        return a.dtype

    def check_dtype(self, dtype):
        if dtype not in self.prefixes:
            self.error("No implementation for arrays of dtype %r" % dtype)

    def check_is_2d_contig_array(self, name, a):
        if not self.is_ndarray(a):
            self.error("`%s` must be a NumPy ndarray" % name)
        elif not self.ndim(a) == 2:
            self.error("`%s` must be 2 dimensional" % name)
        elif not self.is_contig(a):
            self.error("`%s` must be contiguous" % name)

    def check_uniform_dtype(self, **kwargs):
        params = list(kwargs.items())
        dtype = self.dtype(params[0][1])
        self.check_dtype(dtype)
        for k, v in params:
            if not self.dtype(v) == dtype:
                self.error("Non-uniform dtypes found, `%s`'s dtype is %r not %r"
                           % (k, self.dtype(v), dtype))
        return dtype

    def check_bools(self, **kwargs):
        for k, v in kwargs.items():
            if not self.is_bool(v):
                self.error("`%s` must be a bool" % k)

    def check_ints(self, **kwargs):
        for k, v in kwargs.items():
            if not self.is_int(v):
                self.error("`%s` must be an int" % k)

    def get_lib_func(self, name, dtype):
        prefix = self.prefixes[dtype]
        return getattr(lib, prefix + name)

    def check_gemm(
        self, a, b, out=None, a_trans=False, a_conj=False, b_trans=False,
        b_conj=False, alpha=1.0, beta=0.0, nthreads=-1
    ):
        arrays = {"a": a, "b": b}
        if not self.is_none(out):
            arrays["out"] = out
        for k, v in arrays.items():
            self.check_is_2d_contig_array(k, v)
        dtype = self.check_uniform_dtype(**arrays)

        self.check_bools(a_trans=a_trans, a_conj=a_conj, b_trans=b_trans, b_conj=b_conj)
        self.check_ints(nthreads=nthreads)

        return dtype, self.get_lib_func("gemm", dtype)


class PythonTyping(TypingContext):
    prefixes = {np.dtype('f4'): 's',
                np.dtype('f8'): 'd',
                np.dtype('c8'): 'c',
                np.dtype('c16'): 'z'}

    def error(self, msg):
        raise TypeError(msg)

    def is_bool(self, a):
        return isinstance(a, bool)

    def is_int(self, a):
        return isinstance(a, int)

    def is_none(self, a):
        return a is None

    def is_ndarray(self, a):
        return isinstance(a, np.ndarray)

    def is_contig(self, a):
        return a.flags.contiguous

    def cast_scalar(self, name, val, dtype):
        try:
            return dtype.type(val)
        except TypeError as exc:
            self.error("%s %s" % (name, exc))


_CTX = PythonTyping()


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
    dtype, gemm = _CTX.check_gemm(a, b, out, a_trans, a_conj, b_trans, b_conj,
                                  alpha, beta, nthreads)
    alpha = _CTX.cast_scalar("alpha", alpha, dtype)
    beta = _CTX.cast_scalar("beta", beta, dtype)
    return gemm(a, b, out, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads)
