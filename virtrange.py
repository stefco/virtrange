"A lazy numpy ``arange`` that only generates values as needed."

__version__ = "0.1.0"

import functools
from operator import ge, le

DTYPE_MAXES = [(2**(8*i-1)-1, 'i%s'%i) for i in map(lambda x: 2**x, range(4))]


def default_type(start, stop):
    maxval = max(abs(start), abs(stop))
    for typemax, typename in DTYPE_MAXES:
        if maxval < typemax:
            import numpy
            return numpy.dtype(typename)
    raise ValueError("Cannot represent provided ")


# https://numpy.org/doc/stable/user/basics.dispatch.html
# https://numpy.org/doc/stable/user/basics.subclassing.html
class VirtRange:
    _slice: tuple
    _dtype: object

    def __init__(self, startstop, stop=None, step=1, dtype=None):
        self._slice = ((0, startstop, 1) if stop is None
                       else (startstop, stop, step))
        self._dtype = dtype

    def __array__(self):
        raise NotImplementedError()

    def all(self):
        s = self._slice
        return bool(s[0]) and any(any(c(i, 0) for i in s)
                                  or bool(s[0]%s[2]) for c in (ge, le))

    def any(self):
        return bool(self._slice[0] or len(self))

    def __len__(self):
        return max((self._slice[1]-self._slice[0])//self._slice[2], 0)
