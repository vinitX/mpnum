#!/usr/bin/env python
# encoding: utf-8
"""General helper functions for dealing with arrays (esp. for quantum mechanics
"""

from __future__ import division, print_function
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


def local_to_global(array, sites):
    """Inverse to local_to_global

    :param np.ndarray array: Array with ndim, such that ndim % sites = 0
    :param int sites: Number of distinct sites
    :returns: Array with same ndim as array, but reshaped

    >>> ltg, gtl = local_to_global, global_to_local
    >>> ltg(gtl(np.zeros((1, 2, 3, 4, 5, 6)), 3), 3).shape
    (1, 2, 3, 4, 5, 6)
    >>> ltg(gtl(np.zeros((1, 2, 3, 4, 5, 6)), 2), 2).shape
    (1, 2, 3, 4, 5, 6)

    """
    assert array.ndim % sites == 0, \
        "ndim={} is not a multiple of {}".format(array.ndim, sites)
    plegs = array.ndim // sites
    order = sum(([plegs*i + j for i in range(sites)] for j in range(plegs)),
                [])
    return np.transpose(array, order)


def partial_trace(array, traceout):
    """Return the partial trace of array over the sites given in traceout.
    
    :param np.ndarray array: Array in global from (see :func:`global_to_local` above)
    :param traceout: List of sites to trace out, must be in *ascending* order
    :returns: Partial trace over input array
    """
    if len(traceout) == 0:
        return array
    sites, rem = divmod(array.ndim, 2)
    assert rem == 0, 'ndim={} is not a multiple of 2'.format(array.ndim)
    return partial_trace(np.trace(array, axis1=traceout[-1], axis2=traceout[-1]+sites), traceout=traceout[:-1])


def matdot(A, B, axes=((-1,), (0,))):
    """np.tensordot with sane defaults for matrix multiplication"""
    return np.tensordot(A, B, axes=axes)


def norm_2(x):
    """l2 norm of the vector x"""
    return np.sqrt(np.vdot(x, x))
