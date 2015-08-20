#!/usr/bin/env python
# encoding: utf-8
from __future__ import absolute_import, division, print_function

import itertools as it

import mpnum.mparray as mp
import mpnum.mpsmpo as mpsmpo


class MPPovm(mp.MPArray):
    """Docstring for MPPovm. """

    def __iter__(self):
        return self.paxis_iter(axes=0)

    @classmethod
    def from_local_povm(cls, lelems, width):
        """@todo: Docstring for from_local_povm.

        :param lelems: @todo
        :param width: @todo
        :returns: @todo

        """
        return cls.from_kron(it.repeat(lelems, width))

    @property
    def probability_map(self):
        return mp.MPArray(mp._local_reshape(ten, (ten.shape[1], -1)).conj()
                          for ten in self._ltens)

    def expectations(self, mpa):
        """@todo: Docstring for expectation.

        :param rho: @todo
        :returns: @todo

        """
        if all(pleg == 1 for pleg in mpa.plegs):
            raise NotImplementedError("MPS expectations come soon")
        elif all(pleg == 2 for pleg in mpa.plegs):
            pmap = self.probability_map
            for ssite, rho_red in mpsmpo.reductions_mpo(mpa, len(self)):
                yield ssite, mp.dot(pmap, rho_red.ravel())
            return

        raise ValueError("Could not understand data dype.")
