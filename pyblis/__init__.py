from . import lib
from ._wrappers import dot

def _init_numba():
    """Initialize the numba extension"""
    from . import _numba

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
