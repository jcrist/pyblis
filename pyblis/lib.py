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
    for name, T in [("pybli_sgemm", ct.c_float), ("pybli_dgemm", ct.c_double)]:
        func = getattr(libblis, name)
        func.argtypes = (
            ct.c_bool,      # trans_a
            ct.c_bool,      # trans_b
            ct.c_long,      # m
            ct.c_long,      # n
            ct.c_long,      # k
            T,              # alpha
            ct.c_void_p,    # a
            ct.c_long,      # rsa
            ct.c_long,      # csa
            ct.c_void_p,    # b
            ct.c_long,      # rsb
            ct.c_long,      # csb
            T,              # beta
            ct.c_void_p,    # c
            ct.c_long,      # rsc
            ct.c_long       # csc
        )
        yield func


sgemm, dgemm = _load_gemm()
