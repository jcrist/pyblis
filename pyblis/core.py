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

    def check_cast_scalar(self, name, val, dtype):
        raise NotImplementedError

    def ndim(self, a):
        return a.ndim

    def dtype(self, a):
        return a.dtype

    def check_dtype(self, dtype):
        if dtype not in self.prefixes:
            self.error("No implementation for arrays of dtype %r" % dtype)

    def check_is_2d_array(self, **kwargs):
        for k, v in kwargs.items():
            if not self.is_ndarray(v):
                self.error("`%s` must be a NumPy ndarray" % k)
            elif not self.ndim(v) == 2:
                self.error("`%s` must be 2 dimensional" % k)

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
        self.check_is_2d_array(**arrays)
        dtype = self.check_uniform_dtype(**arrays)

        self.check_bools(a_trans=a_trans, a_conj=a_conj, b_trans=b_trans, b_conj=b_conj)
        self.check_ints(nthreads=nthreads)

        alpha = self.check_cast_scalar("alpha", alpha, dtype)
        beta = self.check_cast_scalar("beta", beta, dtype)

        gemm = self.get_lib_func("gemm", dtype)

        return gemm, alpha, beta

    def check_syrk(
        self, a, out=None, a_trans=False, a_conj=False, out_upper=False,
        alpha=1.0, beta=0.0, nthreads=-1
    ):
        arrays = {"a": a}
        if not self.is_none(out):
            arrays["out"] = out
        self.check_is_2d_array(**arrays)
        dtype = self.check_uniform_dtype(**arrays)

        self.check_bools(a_trans=a_trans, a_conj=a_conj, out_upper=out_upper)
        self.check_ints(nthreads=nthreads)

        alpha = self.check_cast_scalar("alpha", alpha, dtype)
        beta = self.check_cast_scalar("beta", beta, dtype)

        syrk = self.get_lib_func("syrk", dtype)

        return syrk, alpha, beta

    def check_mksymm(self, a, upper, nthreads=-1):
        self.check_is_2d_array(a=a)
        dtype = self.dtype(a)
        self.check_dtype(dtype)
        self.check_bools(upper=upper)
        self.check_ints(nthreads=nthreads)

        return self.get_lib_func("mksymm", dtype)


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

    def check_cast_scalar(self, name, val, dtype):
        try:
            return dtype.type(val)
        except TypeError as exc:
            self.error("%s %s" % (name, exc))


_CTX = PythonTyping()


def gemm(a, b, out=None, a_trans=False, a_conj=False,
         b_trans=False, b_conj=False, alpha=1.0, beta=0.0, nthreads=-1):
    """Multiply two matrices.

    Solves ``out = alpha * op_a(a).dot(op_b(b)) + beta * out``.

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
    gemm, alpha, beta = _CTX.check_gemm(
        a, b, out, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads
    )
    return gemm(a, b, out, a_trans, a_conj, b_trans, b_conj, alpha, beta, nthreads)


def syrk(a, out=None, a_trans=False, a_conj=False, out_upper=False, alpha=1.0,
         beta=0.0, nthreads=-1):
    """Multiply a matrix with its transpose.

    Solves ``out = alpha * op_a(a).dot(op_a(a).T) + beta * out``.

    Where ``op_a`` indicates any transpose/conjugate operation specified
    on ``a``, and ``out`` is an optional lower/upper triangular matrix.

    Parameters
    ----------
    a : np.ndarray[T]
        The input array, where ``T`` is one of (float64, float32, complex128,
        complex64).
    out : np.ndarray[T], optional
        An optional output array, must match the type of the input array. If
        not provided, a new array will be allocated.
    a_trans : bool, optional
        Whether to transpose ``a``. Default is False.
    a_conj : bool, optional
        Whether to conjugate ``a``. Default is False.
    out_upper : bool, optional
        Whether ``out`` is an upper (``True``) or lower (``False``) triangular
        matrix. Default is False.
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
    syrk, alpha, beta = _CTX.check_syrk(
        a, out, a_trans, a_conj, out_upper, alpha, beta, nthreads
    )
    return syrk(a, out, a_trans, a_conj, out_upper, alpha, beta, nthreads)


def mksymm(a, upper=False, nthreads=-1):
    """Convert a triangular matrix into a symmetric matrix.

    Parameters
    ----------
    a : np.ndarray[T]
        A triangular square matrix, where ``T`` is one of (float64, float32,
        complex128, complex64).
    upper : bool, optional
        Whether ``a`` is an upper (``True``) or lower (``False``) triangular
        matrix. Default is False.
    nthreads : int
        The number of threads to use. Defaults to deriving from environment
        variables (e.g. ``BLIS_NUM_THREADS``).

    Returns
    -------
    a : np.ndarray[T]
    """
    mksymm = _CTX.check_mksymm(a, upper, nthreads)
    return mksymm(a, upper, nthreads)
