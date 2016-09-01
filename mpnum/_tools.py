# encoding: utf-8
"""General helper functions for dealing with arrays (esp. for quantum
mechanics)

"""

from __future__ import division, print_function

import itertools as it

import numpy as np

from six.moves import range, zip


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


def block_diag(summands, axes=(0, 1)):
    """Block-diagonal sum for n-dimensional arrays.

    Perform something like a block diagonal sum (if len(axes) == 2)
    along the specified axes. All other axes must have identical
    sizes.

    :param axes: Along these axes, perform a block-diagonal sum. Can
        be negative.

    >>> a = np.arange(8).reshape((2, 2, 2))
    >>> b = np.arange(8, 16).reshape((2, 2, 2))
    >>> a
    array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[4, 5],
            [6, 7]]])
    >>> b
    array([[[ 8,  9],
            [10, 11]],
    <BLANKLINE>
           [[12, 13],
            [14, 15]]])
    >>> block_diag((a, b), axes=(1, -1))
    array([[[ 0,  1,  0,  0],
            [ 2,  3,  0,  0],
            [ 0,  0,  8,  9],
            [ 0,  0, 10, 11]],
    <BLANKLINE>
           [[ 4,  5,  0,  0],
            [ 6,  7,  0,  0],
            [ 0,  0, 12, 13],
            [ 0,  0, 14, 15]]])

    """
    axes = np.array(axes)
    axes += (axes < 0) * summands[0].ndim

    nr_axes = len(axes)
    axes_order = list(axes)
    axes_order += [i for i in range(summands[0].ndim)
                   if i not in axes]
    summands = [array.transpose(axes_order) for array in summands]

    shapes = np.array([array.shape[:nr_axes] for array in summands])
    res = np.zeros(tuple(shapes.sum(axis=0)) + summands[0].shape[nr_axes:],
                   dtype=summands[0].dtype)
    startpos = np.zeros(nr_axes, dtype=int)

    for array, shape in zip(summands, shapes):
        endpos = startpos + shape
        pos = [slice(start, end) for start, end in zip(startpos, endpos)]
        res[pos] += array
        startpos = endpos

    old_axes_order = np.argsort(axes_order)
    res = res.transpose(old_axes_order)
    return res


def verify_real_nonnegative(values, zero_tol=1e-6, zero_cutoff=None):
    """Check that values are real and non-negative.

    :param numpy.ndarray values: An ndarray of complex or real values
        (or a single value).

    :param zero_tol: Replace negative real values and imaginary parts
        with modulus smaller than or equal to `zero_tol` with zero.

    :param zero_cutoff: Replace positive real values smaller than or
       equal to zero_cutoff with zero. Default: None (`zero_tol` will
       be used).

    :returns: An ndarray of real values (or a single real value).

    """
    single_value = False
    if getattr(values, 'ndim', 0) == 0:
        single_value = True
        values = np.array([values])
    if zero_cutoff is None:
        zero_cutoff = zero_tol
    if hasattr(values, 'imag'):
        assert all(abs(values.imag) <= zero_tol), \
            'non-real values found: {}'.format(values)
    values = values.real
    assert all(values >= -zero_tol), 'negative values found: {}'.format(values)
    values[values <= zero_cutoff] = 0
    if single_value:
        return values[0]
    return values
