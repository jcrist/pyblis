pyblis
======

A Python wrapper for BLIS_.

Provides raw bindings to ``BLIS`` functions, as well as familiar higher-level
wrappers (like ``pyblis.dot``). All functions can be used from Python, or
inside ``numba`` code, even in ``nopython`` mode.

.. code-block:: python

    import pyblis

    a = np.random.normal(size=(2000, 2000))

    # Can call methods from Python
    res = pyblis.dot(a, a.T)

    # Or from within a numba-compiled function
    import numba as nb

    @nb.jit(nopython=True)
    def myfunc(a):
        return pyblis.dot(a, a.T)

    res = myfunc(a)


The library can be built either with a self contained ``libblis`` (for PyPI
support), or linking to a separate ``libblis`` (for conda support).


.. _BLIS: https://github.com/flame/blis/
.. _numba: http://numba.pydata.org/
