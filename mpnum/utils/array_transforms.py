# encoding: utf-8
"""Helper functions for transforming arrays"""

from __future__ import division, print_function

import itertools as it

import numpy as np


__all__ = ['global_to_local', 'local_to_global']


def global_to_local(array, sites, left_skip=0, right_skip=0):
    """Converts a general `sites`-local array with fixed number p of physical
    legs per site from the global form

                A[i_1,..., i_N, j_1,..., j_N, ...]

    (i.e. grouped by physical legs) to the local form

                A[i_1, j_1, ..., i_2, j_2, ...]

    (i.e. grouped by site).

    :param np.ndarray array: Array with ndim, such that ndim % sites = 0
    :param int sites: Number of distinct sites
    :param int left_skip: Ignore that many axes on the left
    :param int right_skip: Ignore that many axes on the right
    :returns: Array with same ndim as array, but reshaped

    >>> global_to_local(np.zeros((1, 2, 3, 4, 5, 6)), 3).shape
    (1, 4, 2, 5, 3, 6)
    >>> global_to_local(np.zeros((1, 2, 3, 4, 5, 6)), 2).shape
    (1, 3, 5, 2, 4, 6)

    """
    skip = left_skip + right_skip
    ndim = array.ndim - skip
    assert ndim % sites == 0, \
        "ndim={} is not a multiple of {}".format(array.ndim, sites)
    plegs = ndim // sites
    order = (left_skip + i + sites * j for i in range(sites) for j in range(plegs))
    order = tuple(it.chain(
        range(left_skip), order, range(array.ndim - right_skip, array.ndim)))
    return np.transpose(array, order)


def local_to_global(array, sites, left_skip=0, right_skip=0):
    """Inverse of local_to_global

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
        range(left_skip), order, range(array.ndim - right_skip, array.ndim)))
    return np.transpose(array, order)
