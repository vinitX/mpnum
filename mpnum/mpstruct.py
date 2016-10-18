# encoding: utf-8
"""TODO"""
from __future__ import absolute_import, division, print_function

import itertools as it
import collections

from six.moves import range, zip


def _roview(array):
    """Creates a read only view of the numpy array `view`."""
    view = array.view()
    view.setflags(write=False)
    return view


class LocalTensors(object):
    """Docstring for LocalTensors. """

    def __init__(self, ltens, nform=(None, None)):
        """@todo: to be defined1. """
        self._ltens = list(ltens)
        lnormalized, rnormalized = nform
        self._lnormalized = lnormalized or 0
        self._rnormalized = rnormalized or len(self._ltens)

        assert len(self._ltens) > 0
        assert 0 <= self._lnormalized < len(self._ltens)
        assert 0 < self._rnormalized <= len(self._ltens)

        if __debug__:
            for i, (ten, nten) in enumerate(zip(self._ltens[:-1], self._ltens[1:])):
                assert ten.shape[-1] == nten.shape[0]

    def _update(self, index, tens, normalization=None):
        """ Actually updates

        For parameters see :func:`update`.
        """
        self._ltens[index] = tens
        # If a normalized tensor is set next to a normalized slice,
        # the size of the normalized slice will increase by one
        # (equality case; first argument to max/min). If a normalized
        # tensor is set inside a normalized slice, its size will
        # remain the same (inequality case; second argument).
        if normalization == 'left' and self._lnormalized - index >= 0:
            self._lnormalized = max(index + 1, self._lnormalized)
        elif normalization == 'right' and index - self._rnormalized >= -1:
            self._rnormalized = min(index, self._rnormalized)
        else:
            # If a non-normalized tensor is provided, the sizes of the
            # normalized slices may decrease.
            self._lnormalized = min(index, self._lnormalized)
            self._rnormalized = max(index + 1, self._rnormalized)

    def update(self, index, tens, normalization=None):
        """Replaces the local tensor at position `index` with the tensor `tens`.
        by an in-place update

        :param index: Position of the tensor in the chain
        :param tens: New local tensor as numpy.ndarray
        :param normalization: If `tens` is left-/right-normalized, pass `'left'`
            /`'right'`, respectively. Otherwise, pass `None` (default `None`)

        """
        if isinstance(index, slice):
            indices = index.indices(len(self))
            # In Python 3, we can do range(*indices).start etc. Python 2 compat:
            start, stop, step = indices
            # Allow bond dimension changes if multiple consecutive
            # local tensors are changed. Callers should switch to
            tens = list(tens)
            assert self[start].shape[0] == tens[0].shape[0]
            assert all(t.shape[-1] == u.shape[0] for t, u in zip(
                tens[:-1], tens[1:]))
            assert self[stop - 1].shape[-1] == tens[-1].shape[-1]

            if not isinstance(normalization, collections.Sequence):
                normalization = it.repeat(normalization)

            for ten, pos, norm in zip(tens, range(*indices), normalization):
                self._update(pos, ten, normalization=norm)

        else:
            current = self._ltens[index]
            assert tens.ndim >= 2
            assert current.shape[0] == tens.shape[0]
            assert current.shape[-1] == tens.shape[-1]
            self._update(index, tens, normalization=normalization)

    def __len__(self):
        return len(self._ltens)

    def __iter__(self):
        """Use only for read-only access! Do not change arrays in place!

        Subclasses should not override this method because it will
        break basic MPA functionality such as :func:`dot`.

        """
        for ltens in self._ltens:
            yield _roview(ltens)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return (_roview(lten) for lten in self._ltens[index])
        else:
            view = self._ltens[index].view()
            view.setflags(write=False)
            return view

    def __setitem__(self, index, value):
        self.update(index, value)

    @property
    def normal_form(self):
        """Tensors which are currently in left/right-canonical form."""
        return self._lnormalized, self._rnormalized

    def copy(self):
        ltens = (lt.copy() for lt in self._ltens)
        return type(self)(ltens, nform=self.normal_form)
