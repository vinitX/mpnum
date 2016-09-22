# encoding: utf-8
"""TODO"""
from __future__ import absolute_import, division, print_function

from collections.abc import Sequence

import numpy as np


class LocalTensors(Sequence):
    """Object handling all the local tensor business"""

    def __init__(self, ltens, lnormalized=None, rnormalized=None):
        """@todo: to be defined1.

        :param ltens: Sequence of local tensors for the MPA. In order
            to be valid the elements of `lttenss` need to be
            :code:`N`-dimensional arrays with :code:`N > 1` and need
            to fullfill::

                shape(tens[i])[-1] == shape(tens[i])[0].
        """
        Sequence.__init__(self)
        self._ltens = list(ltens)
        self._lnormalized = lnormalized
        self._rnormalized = rnormalized

        if __debug__:
            for i, (ten, nten) in enumerate(zip(self._ltens[:-1], self._ltens[1:])):
                assert ten.shape[-1] == nten.shape[0]

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

    def __getitem(self, index):
        view = self._ltens[index].view()
        view.setflags(write=False)
        return view

    def update(self, index, tens, normalization=None):
        """Replaces the local tensor at position `index` with the tensor `tens`.
        by an in-place update

        :param index: Position of the tensor in the chain
        :param tens: New local tensor as numpy.ndarray
        :param normalization: If `tens` is left-/right-normalized, pass `'left'`
            /`'right'`, respectively. Otherwise, pass `None` (default `None`)
        :returns: @todo

        """
        pass
