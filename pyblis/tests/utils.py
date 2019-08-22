import pytest

import numpy as np


all_dtypes = pytest.mark.parametrize('dtype', ['f4', 'f8', 'c8', 'c16'])


class Base(object):
    def rand(self, dtype, shape=()):
        a = np.random.normal(size=shape).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            a += np.random.normal(size=a.shape) * 1j
        return a if a.shape else a.reshape((1,))[0]

    def call_base(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class NumbaMixin(object):
    @property
    def error_cls(self):
        import numba
        return numba.errors.TypingError

    @classmethod
    def setup_class(cls):
        base, full = cls.compile()
        cls.base = staticmethod(base)
        cls.full = staticmethod(full)

    def call(self, *args, **kwargs):
        return self.full(*args, **kwargs)

    def call_base(self, *args, **kwargs):
        return self.base(*args, **kwargs)
