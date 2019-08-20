import ctypes as ct
import os
import sys


def load_libblis():
    if sys.platform == "darwin":
        ext = ".dylib"
    elif sys.platform == "win32":
        ext = ".dll"
    else:
        ext = ".so"
    path = os.path.join(os.path.dirname(__file__), "_lib" + ext)
    return ct.CDLL(path)


libblis = load_libblis()


# GEMM
def _load_gemm():
    for name, T, is_complex in [
        ("pybli_sgemm", ct.c_float, False),
        ("pybli_dgemm", ct.c_double, False),
        ("pybli_cgemm", ct.c_float, True),
        ("pybli_zgemm", ct.c_double, True)
    ]:
        func = getattr(libblis, name)
        argtypes = [
            ct.c_bool,      # a_trans
            ct.c_bool,      # a_conj
            ct.c_bool,      # b_trans
            ct.c_bool,      # b_conj
            ct.c_long,      # m
            ct.c_long,      # n
            ct.c_long,      # k
            T,              # alpha
        ]
        if is_complex:
            argtypes.append(T)  # alpha_imag
        argtypes.extend([
            ct.c_void_p,    # a
            ct.c_long,      # rsa
            ct.c_long,      # csa
            ct.c_void_p,    # b
            ct.c_long,      # rsb
            ct.c_long,      # csb
            T,              # beta
        ])
        if is_complex:
            argtypes.append(T)  # beta_imag
        argtypes.extend([
            ct.c_void_p,    # c
            ct.c_long,      # rsc
            ct.c_long,      # csc
            ct.c_long       # nthreads
        ])
        func.argtypes = argtypes
        yield func


sgemm, dgemm, cgemm, zgemm = _load_gemm()
