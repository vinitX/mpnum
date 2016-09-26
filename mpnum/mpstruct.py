# encoding: utf-8
"""TODO"""
from __future__ import absolute_import, division, print_function

import itertools as it
import numpy as np


class LocalTensors(object):
    """Docstring for LocalTensors. """

    def __init__(self, ltens, nform=(None, None)):
        """@todo: to be defined1. """
        self._ltens = list(ltens)
        lnormalized, rnormalized = nform
        self._lnormalized = lnormalized or 0
        self._rnormalized = rnormalized or len(self._ltens)

        assert 0 <= self._lnormalized < len(self._ltens)
        assert 0 < self._rnormalized <= len(self._ltens)

        if __debug__:
            for i, (ten, nten) in enumerate(zip(self._ltens[:-1], self._ltens[1:])):
                assert ten.shape[-1] == nten.shape[0]

    def update(self, index, tens, normalization=None, unsafe=False):
        """Replaces the local tensor at position `index` with the tensor `tens`.
        by an in-place update

        :param index: Position of the tensor in the chain
        :param tens: New local tensor as numpy.ndarray
        :param normalization: If `tens` is left-/right-normalized, pass `'left'`
            /`'right'`, respectively. Otherwise, pass `None` (default `None`)
        :returns: @todo

        """
        current = self._ltens[index]
        if not unsafe:
            assert current.shape[0] == tens.shape[0]
            assert current.shape[-1] == tens.shape[-1]
            assert tens.ndim >= 2

        self._ltens[index] = tens
        if normalization == 'left' and self._lnormalized - index >= 0:
            self._lnormalized = max(index + 1, self._lnormalized)
        elif normalization == 'right' and index - self._rnormalized >= -1:
            self._rnormalized = min(index, self._rnormalized)
        else:
            self._lnormalized = min(index, self._lnormalized)
            self._rnormalized = max(index + 1, self._rnormalized)

    def __len__(self):
        return len(self._ltens)

    def __iter__(self):
        """Use only for read-only access! Do not change arrays in place!

        Subclasses should not override this method because it will
        break basic MPA functionality such as :func:`dot`.

        """
        for ltens in self._ltens:
            view = ltens.view()
            view.setflags(write=False)
            yield view

    def __getitem__(self, index):
        if isinstance(index, slice):
            return it.islice(self, index.start, index.stop, index.step)
        else:
            view = self._ltens[index].view()
            view.setflags(write=False)
            return view

    @property
    def normal_form(self):
        """Tensors which are currently in left/right-canonical form."""
        return self._lnormalized, self._rnormalized
