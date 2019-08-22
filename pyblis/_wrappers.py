from . import lib


def dot(a, b, out=None, nthreads=-1):
    """Perform a matrix multiplication.

    Parameters
    ----------
    a, b : np.ndarray[T]
        Two identically typed arrays, where ``T`` is one of
        (float64, float32, complex128, complex64).
    out : np.ndarray[T]
        An optional output array, must match the type of the input arrays. If
        not provided, a new array will be allocated.
    nthreads : int
        The number of threads to use. Defaults to deriving from environment
        variables (e.g. ``BLIS_NUM_THREADS``).
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2 dimensional")
    if (a.ctypes.data == b.ctypes.data and
            a.shape[0] == b.shape[1] and
            a.shape[1] == b.shape[0] and
            a.strides[0] == b.strides[1] and
            a.strides[1] == b.strides[0]):
        return lib.mksymm(lib.syrk(a, out=out, nthreads=nthreads))
    else:
        return lib.gemm(a, b, out=out, nthreads=nthreads)
