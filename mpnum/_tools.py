#!/usr/bin/env python
# encoding: utf-8
"""General helper functions for dealing with arrays (esp. for quantum mechanics
"""

from __future__ import division, print_function
import itertools as it
import numpy as np


SI = np.eye(2)
SX = np.array([[0, 1], [1, 0]])
SY = np.array([[0, 1j], [-1j, 0]])
SZ = np.array([[1, 0], [0, -1]])
SP = 0.5 * (SX + 1j * SY)
SM = 0.5 * (SX - 1j * SY)
PAULI = np.asarray([SI, SX, SY, SZ])


def global_to_local(array, sites):
    """Converts a general `sites`-local array with fixed number p of physical
    legs per site from the global form

                A[i_1,..., i_N, j_1,..., j_N, ...]

    (i.e. grouped by physical legs) to the local form

                A[i_1, j_1, ..., i_2, j_2, ...]

    (i.e. grouped by site).

    :param np.ndarray array: Array with ndim, such that ndim % sites = 0
    :param int sites: Number of distinct sites
    :returns: Array with same ndim as array, but reshaped

    >>> global_to_local(np.zeros((1, 2, 3, 4, 5, 6)), 3).shape
    (1, 4, 2, 5, 3, 6)
    >>> global_to_local(np.zeros((1, 2, 3, 4, 5, 6)), 2).shape
    (1, 3, 5, 2, 4, 6)

    """
    assert array.ndim % sites == 0, \
        "ndim={} is not a multiple of {}".format(array.ndim, sites)
    plegs = array.ndim // sites
    order = [i // plegs + sites * (i % plegs) for i in range(plegs * sites)]
    return np.transpose(array, order)


def local_to_global(array, sites, left_skip=0, right_skip=0):
    """Inverse to local_to_global

    :param np.ndarray array: Array with ndim, such that ndim % sites = 0
    :param int sites: Number of distinct sites
    :param int left_skip: Ignore that many axes on the left
    :param int right_skip: Ignore that many axes on the right
    :returns: Array with same ndim as array, but reshaped

    >>> ltg, gtl = local_to_global, global_to_local
    >>> ltg(gtl(np.zeros((1, 2, 3, 4, 5, 6)), 3), 3).shape
    (1, 2, 3, 4, 5, 6)
    >>> ltg(gtl(np.zeros((1, 2, 3, 4, 5, 6)), 2), 2).shape
    (1, 2, 3, 4, 5, 6)

    Transform all or only the inner axes:

    >>> ltg = local_to_global
    >>> ltg(np.zeros((1, 2, 3, 4, 5, 6)), 3).shape
    (1, 3, 5, 2, 4, 6)
    >>> ltg(np.zeros((1, 2, 3, 4, 5, 6)), 2, left_skip=1, right_skip=1).shape
    (1, 2, 4, 3, 5, 6)

    """
    skip = left_skip + right_skip
    ndim = array.ndim - skip
    assert ndim % sites == 0, \
        "ndim={} is not a multiple of {}".format(array.ndim, sites)
    plegs = ndim // sites
    order = (left_skip + plegs*i + j for j in range(plegs) for i in range(sites))
    order = tuple(it.chain(
        range(left_skip), order, range(array.ndim-right_skip, array.ndim)))
    return np.transpose(array, order)


def partial_trace(array, traceout):
    """Return the partial trace of an array over the sites given in traceout.

    :param np.ndarray array: Array in global form (see :func:`global_to_local`
        above) with exactly 2 legs per site
    :param traceout: List of sites to trace out, must be in _ascending_ order
    :returns: Partial trace over input array

    """
    if len(traceout) == 0:
        return array
    sites, rem = divmod(array.ndim, 2)
    assert rem == 0, 'ndim={} is not a multiple of 2'.format(array.ndim)

    # Recursively trace out the last site given
    array_red = np.trace(array, axis1=traceout[-1], axis2=traceout[-1] + sites)
    return partial_trace(array_red, traceout=traceout[:-1])


def matdot(A, B, axes=((-1,), (0,))):
    """np.tensordot with sane defaults for matrix multiplication"""
    return np.tensordot(A, B, axes=axes)


def mkron(*args):
    """np.kron() with an arbitrary number of n >= 1 arguments"""
    if len(args) == 1:
        return args[0]
    return mkron(np.kron(args[0], args[1]), *args[2:])


def norm_2(x):
    """l2 norm of the vector x"""
    return np.sqrt(np.vdot(x, x))
