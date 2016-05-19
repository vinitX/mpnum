# encoding: utf-8
from __future__ import absolute_import, division, print_function

import itertools as it

import mpnum.mparray as mp
import mpnum.mpsmpo as mpsmpo


class MPPovm(mp.MPArray):
    """MPArray representation of multipartite POVM

    There are two different ways to write down a POVM in matrix product form

    1) As a list of matrix product operators, where each entry corresponds to
        a single POVM element

    2) As a matrix proudct array with 3 physical legs:

                [POVM index, column index, row index]

        that is, the first physical leg of the MPArray corresponds to the index
        of the POVM element. This representation is especially helpful for
        computing expectation values with MPSs/MPDOs.

    Here, we choose the second.

    .. todo:: This class should provide a function which returns
        expectation values as full array. (Even though computing
        expectation values using the POVM struture brings advantages,
        we usually need the result as full array.) This function
        should also replace small negative probabilities by zero and
        normalize the sum of all probabilities to unity (if the
        deviation is non-zero but small). The same checks should also
        be implemented in localpovm.POVM.

    .. todo:: Right now we use this class for multi-site POVMs with
        elements obtained from every possible combination of the
        elements of single-site POVMs: The POVM index is split across
        all sites. Explore whether and how this concept can also be
        useful in other cases.

    """

    @property
    def elements(self):
        """Returns an iterator over all POVM elements. The result is the i-th
        POVM element in MPO form.

        It would be nice to call this method `__iter__`, but this
        breaks `mp.dot(mppovm, ...)`. In addition,
        `next(iter(mppovm))` would not be equal to `mppovm[0]`.

        """
        return self.paxis_iter(axes=0)

    @classmethod
    def from_local_povm(cls, lelems, width):
        """Generates a product POVM on `width` sites.

        :param lelems: POVM elements as an iterator over all local elements
            (i.e. an iterator over numpy arrays representing the latter)
        :param int width: Number of sites the POVM lives on
        :returns: :class:`MPPovm` which is a product POVM of the `lelems`

        """
        return cls.from_kron(it.repeat(lelems, width))

    @property
    def probability_map(self):
        """Map that takes a raveled MPDO to the POVM probabilities

        You can use :func:`MPPovm.expectations()` as a convenient
        wrapper around this map.

        If `rho` is a matrix product density operator (MPDO), then

        .. code::

            mp.dot(a_povm.probability_map, rho.ravel())

        produces the POVM probabilities as MPA (similar to
        :func:`mpnum.povm.localpovm.POVM.probability_map`).

        """
        # See :func:`.localpovm.POVM.probability_map` for explanation
        # of the transpose.
        return self.transpose((0, 2, 1)).reshape(
            (pdim[0], -1) for pdim in self.pdims)

    def expectations(self, mpa, mode='auto'):
        """Computes the exp. values of the POVM elements with given state

        :param mpa: State given as MPDO, MPS, or PMPS
        :param mode: In which form `mpa` is given. Possible values: 'mpdo',
            'pmps', 'mps', or 'auto'. If 'auto' is passed, we choose between
            'mps' or 'mpdo' depending on the number of physical legs
        :returns: Iterator over the expectation values, the n-th element is
            the expectation value correponding to the reduced state on sites
            [n,...,n + len(self) - 1]

        """
        assert len(self) <= len(mpa)
        if mode == 'auto':
            if all(pleg == 1 for pleg in mpa.plegs):
                mode = 'mps'
            elif all(pleg == 2 for pleg in mpa.plegs):
                mode = 'mpdo'

        pmap = self.probability_map

        if mode == 'mps':
            for psi_red in mpsmpo.reductions_mps_as_pmps(mpa, len(self)):
                rho_red = mpsmpo.pmps_to_mpo(psi_red)
                yield mp.dot(pmap, rho_red.ravel())
            return
        elif mode == 'mpdo':
            for rho_red in mpsmpo.reductions_mpo(mpa, len(self)):
                yield mp.dot(pmap, rho_red.ravel())
            return
        elif mode == 'pmps':
            for psi_red in mpsmpo.reductions_pmps(mpa, len(self)):
                rho_red = mpsmpo.pmps_to_mpo(psi_red)
                yield mp.dot(pmap, rho_red.ravel())
            return
        else:
            raise ValueError("Could not understand data dype.")
