# encoding: utf-8
from __future__ import absolute_import, division, print_function

import itertools as it
import numpy as np

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

    def __init__(self, *args, **kwargs):
        mp.MPArray.__init__(self, *args, **kwargs)
        assert all(plegs == 3 for plegs in self.plegs), \
            "Need 3 physical legs at each site: {!r}".format(self.pdims)
        assert all(pdims[1] == pdims[2] for pdims in self.pdims), \
            "Hilbert space dimension mismatch: {!r}".format(self.pdims)

    @classmethod
    def from_local_povm(cls, lelems, width):
        """Generates a product POVM on `width` sites.

        :param lelems: POVM elements as an iterator over all local elements
            (i.e. an iterator over numpy arrays representing the latter)
        :param int width: Number of sites the POVM lives on
        :returns: :class:`MPPovm` which is a product POVM of the `lelems`

        """
        return cls.from_kron(it.repeat(lelems, width))

    @classmethod
    def eye(cls, local_dims):
        return cls.from_kron((np.eye(dim).reshape((1, dim, dim)) for dim in local_dims))

    @property
    def outdims(self):
        """Tuple of outcome dimensions"""
        # First physical leg dimension
        return tuple(lt.shape[1] for lt in self._ltens)

    @property
    def hdims(self):
        """Tuple of local Hilbert space dimensions"""
        # Second physical leg dimension (equals third physical leg dimension)
        return tuple(lt.shape[2] for lt in self._ltens)

    @property
    def elements(self):
        """Returns an iterator over all POVM elements. The result is the i-th
        POVM element in MPO form.

        It would be nice to call this method `__iter__`, but this
        breaks `mp.dot(mppovm, ...)`. In addition,
        `next(iter(mppovm))` would not be equal to `mppovm[0]`.

        """
        return self.paxis_iter(axes=0)

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

    def find_matching_elements(self, other, eps=1e-10):
        """Find POVM elements in `other` which have information on `self`

        We find all POVM sites in `self` which have only one possible
        outcome. We discard these outputs in `other` and afterwards
        check `other` and `self` for any common POVM elements.

        :param other: Another MPPovm
        :param eps: Threshould for values which should be treated as zero

        :returns: (`sites`, `matches`, `prefactors`)

         `sites` contains the positions of all sites where `self`
         performs measurements.

        `matches[i1, ..., ik, j1, ..., jk]` specifies whether outcome
         `(i1, ..., ik)` of `self` has the same POVM element as the
         partial outcome `(j1, ..., jk)` of `other`; outcomes are
         specified only on the sites mentioned in `sites` and `k =
         len(sites)`.

        `prefactors[i1, ..., ik]` specifies how samples from `other`
        have to be weighted to correspond to samples for `self`.

        """
        if self.hdims != other.hdims:
            raise ValueError('Incompatible input Hilbert space: {!r} vs {!r}'
                             .format(self.hdims, other.hdims))
        # Drop measurement outcomes in `other` if there is only one
        # measurement outcome in `self`
        support = tuple(outdim > 1 for outdim in self.outdims)
        tr = mp.MPArray.from_kron([
            np.eye(outdim, dtype=lt.dtype)
            if keep else np.ones((1, outdim), dtype=lt.dtype)
            for keep, lt, outdim in zip(support, other, other.outdims)
        ])
        other = MPPovm(mp.dot(tr, other))

        # Compute all inner products between elements from self and other
        inner = mp.dot(self.conj(), other, axes=((1, 2), (1, 2)))
        # Compute squared norms of all elements from inner products
        snormsq = mp.dot(self.conj(), self, axes=((1, 2), (1, 2)))
        eye3d = mp.MPArray.from_kron(
            # Drop inner products between different elements
            np.fromfunction(lambda i, j, k: (i == j) & (j == k), [outdim] * 3)
            for outdim in self.outdims
        )
        snormsq = mp.dot(eye3d, snormsq, axes=((1, 2), (0, 1)))
        onormsq = mp.dot(other.conj(), other, axes=((1, 2), (1, 2)))
        eye3d = mp.MPArray.from_kron(
            # Drop inner products between different elements
            np.fromfunction(lambda i, j, k: (i == j) & (j == k), [outdim] * 3)
            for outdim in other.outdims
        )
        onormsq = mp.dot(eye3d, onormsq, axes=((1, 2), (0, 1)))
        inner = abs(mp.prune(inner, True).to_array_global())**2
        snormsq = mp.prune(snormsq, True).to_array().real
        onormsq = mp.prune(onormsq, True).to_array().real
        assert (snormsq > 0).all()
        assert (onormsq > 0).all()
        assert inner.shape == snormsq.shape + onormsq.shape
        # Compute the product of the norms of each element from self and other
        normprod = np.outer(snormsq, onormsq).reshape(inner.shape)
        assert ((normprod - inner) / normprod >= -eps).all()
        # Equality in the Cauchy-Schwarz inequality implies that the
        # vectors are linearly dependent
        match = abs(inner/normprod - 1) <= eps

        # Compute the prefactors by which matching elements differ
        snormsq_shape = snormsq.shape
        snormsq = snormsq.reshape(snormsq_shape + (1,) * onormsq.ndim)
        onormsq = onormsq.reshape((1,) * len(snormsq_shape) + onormsq.shape)
        prefactors = (snormsq / onormsq)**0.5
        assert prefactors.shape == match.shape
        prefactors[~match] = np.nan

        sites = np.nonzero(support)[0]
        return sites, match, prefactors
