#!/usr/bin/env python
# encoding: utf-8
"""Module containing routines for dealing with general matrix product arrays.

TODO
"""

from __future__ import division, print_function

import numpy as np
from numpy.linalg import qr, svd
from itertools import izip


def _extract_factors(tens, plegs):
    """Extract iteratively the leftmost MPO tensor with given number of
    legs by a qr-decomposition

    :param np.ndarray tens: Full tensor to be factorized
    :param int plegs: Number of physical legs per site
    :returns: List of local tensors with given number of legs yielding a
        factorization of tens
    """
    if tens.ndim == plegs + 1:
        return [tens.reshape(tens.shape + (1,))]
    elif tens.ndim < plegs + 1:
        raise ValueError("Number of remaining legs insufficient.")
    else:
        unitary, rest = qr(tens.reshape((np.prod(tens.shape[:plegs + 1]),
                                         np.prod(tens.shape[plegs + 1:]))))

        unitary = unitary.reshape(tens.shape[:plegs + 1] + rest.shape[:1])
        rest = rest.reshape(rest.shape[:1] + tens.shape[plegs + 1:])

        return [unitary] + _extract_factors(rest, plegs)


class MPArray(object):
    """Efficient representation of a general N-partite array A in matrix
    product form with closed boundary conditions:

            A^((i1),...,(iN)) = prod_k A^[k]_(ik)   (*)

    where the A^[k] are local tensors (with N legs). The matrix products in
    (*) are taken with respect to the left and right leg and the multi-
    index (ik) corresponds to the physical legs. Open boundary conditions
    imply that shape(A[0])[0] == shape(A[-1])[-1] == 1.

    By convention, the 0th and last dimension of the local tensors are reserved
    for the auxillary legs.
    """

    def __init__(self, ltens):
        """
        :param list ltens: List of local tensors for the MPA. In order to be
            valid the elements of `tens` need to be N-dimensional arrays
            with N > 1 and need to fullfill

                    shape(tens[i])[-1] == shape(tens[i])[0].

        """
        for i, (ten, nten) in enumerate(izip(ltens[:-1], ltens[1:])):
            if ten.shape[-1] != nten.shape[0]:
                raise ValueError("Shape mismatch on {}: {} != {}"
                                 .format(i, ten.shape[-1], nten.shape[0]))
        self._ltens = np.asarray(ltens)

    @classmethod
    def from_array(cls, array, plegs):
        """Computes the (exact) representation of `array` as MPA with open
        boundary conditions, i.e. bond dimension 1 at the boundary. This
        is done by factoring the off the left and the "physical" legs from
        the rest of the tensor by a QR decomposition and working its way
        through the tensor from the left. This yields a left-canonical
        representation of `array`.

        The result is a chain of local tensors with `plegs` physical legs at
        each location and has array.ndim // plegs number of sites.

        :param np.ndarray array: Array representation with global structure
            array[(i1), ..., (iN)], i.e. the legs which are factorized into
            the same factor are already adiacent. (For me details see
            :func:`_qmtools.global_to_local`)
        :param int plegs: Number of physical legs per site

        """
        if array.ndim % plegs != 0:
            raise ValueError("plegs invalid: {} is not multiple of {}"
                             .format(array.ndim, plegs))
        return cls(_extract_factors(array[None], plegs=plegs))

    def __len__(self):
        return len(self._ltens)

    def __mul__(self, fact):
        if np.isscalar(fact):
            return MPArray(fact * self._ltens)

        raise NotImplementedError("Multiplication by non-scalar not supported")

    @property
    def dims(self):
        """Tuple of shapes for the local tensors"""
        return tuple(m.shape for m in self._ltens)

    @property
    def bdims(self):
        """Tuple of bond dimensions; 0th entry is 1 for open boundary cond."""
        return tuple(m.shape[0] for m in self._ltens)

    @property
    def pdims(self):
        """Tuple of physical dimensions"""
        return tuple((m.shape[1:-1]) for m in self._ltens)

    @property
    def legs(self):
        """Tuple of total number of legs per site"""
        return tuple(lten.ndim for lten in self._ltens)

    @property
    def plegs(self):
        """Tuple of number of physical legs per site"""
        return tuple(lten.ndim - 2 for lten in self._ltens)

    def to_array(self):
        """Returns the full array representation of the MPT
        :returns: Full matrix A as array of shape [(i1),...,(iN)]

        WARNING: This can be slow for large MPTs!
        """
        res = self._ltens[0]
        for tens in self._ltens[1:]:
            res = np.tensordot(res, tens, axes=(-1, 0))
        # trace doesnt really do anything here, since we are dealing with
        # open boundary conditions anyway
        return np.trace(res, axis1=0, axis2=-1)

    @staticmethod
    def _local_transpose(ltens):
        """Transposes the physical legs of the local tensor `ltens`

        :param ltens: Local tensor as numpy.ndarray with ndim >= 2
        :returns: Transpose of ltens except for first and last dimension

        """
        return np.transpose(ltens, axes=[0] + range(ltens.ndim - 2, 0, -1) +
                            [ltens.ndim - 1])

    def T(self):
        """Transpose of the physical legs"""
        return type(self)([self._local_transpose(tens)
                           for tens in self._ltens])

    def adj(self):
        """Hermitian adjoint"""
        return type(self)([self._local_transpose(tens).conjugate()
                           for tens in self._ltens])

    def C(self):
        """Complex conjugate"""
        return type(self)(np.conjugate(self._ltens))

    @staticmethod
    def _local_dot(ltens_l, ltens_r, axes):
        """Computes the local tensors of a dot product dot(l, r).

        Besides computing the normal dot product, this function rearranges the
        bond legs in such a way that the result is a valid local tensor again.

        :param ltens_l: Array with ndim > 1
        :param ltens_r: Array with ndim > 1
        :param axes: Axes to compute dot product using the convention of
            np.tensordot. Note that these correspond to the true (and not the
            physical) legs of the local tensors

        """
        res = np.tensordot(ltens_l, ltens_r, axes=axes)
        # Rearrange the bond-dimension legs
        res = np.rollaxis(res, ltens_l.ndim - 1, 1)
        res = np.rollaxis(res, ltens_l.ndim - 1, ltens_l.ndim + ltens_r.ndim - 3)
        return res.reshape((ltens_l.shape[0] * ltens_r.shape[0], ) +
                           res.shape[2:-2] +
                           (ltens_l.shape[-1] * ltens_r.shape[-1],))

    def dot(self, fact, axes=(-1, 0)):
        """Compute the matrix product representation of a.b over the given
        (physical) axes.

        :param fact: Second factor
        :param axes: 2-tuple of axes to sum over. Note the difference in
            convention compared to np.tensordot(default: last axis of `left`
            and first axis of `fact`)
        :returns: @todo

        """
        if len(self) != len(fact):
            raise ValueError("mparrays have different lengths: {} != {}"
                            .format(len(self), len(fact)))

        # adapt the axes from physical to true legs
        ax_l, ax_r = axes
        ax_l = ax_l + 1 if ax_l >= 0 else ax_l - 1
        ax_r = ax_r + 1 if ax_r >= 0 else ax_r - 1

        ltens = [self._local_dot(l, r, (ax_l, ax_r))
                 for l, r in izip(self._ltens, fact._ltens)]

        return type(self)(ltens)


###################################################
#  Alternative functions to call member function  #
###################################################
dot = MPArray.dot

