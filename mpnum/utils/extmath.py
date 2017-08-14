# encoding: utf-8
"""Additional math functions for dealing with dense arrays"""

from __future__ import division, print_function

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import aslinearoperator
from six.moves import range, zip

__all__ = ['block_diag', 'matdot', 'mkron', 'partial_trace',
           'truncated_svd']


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


def truncated_svd(A, k):
    """Compute the truncated SVD of the matrix `A` i.e. the `k` largest
    singular values as well as the corresponding singular vectors. It might
    return less singular values/vectors, if one dimension of `A` is smaller
    than `k`.

    In the background it performs a full SVD. Therefore, it might be
    inefficient when `k` is much smaller than the dimensions of `A`.

    :param A: A real or complex matrix
    :param k: Number of singular values/vectors to compute
    :returns: u, s, v, where
        u: left-singular vectors
        s: singular values in descending order
        v: right-singular vectors

    """
    u, s, v = np.linalg.svd(A)
    k_prime = min(k, len(s))
    return u[:, :k_prime], s[:k_prime], v[:k_prime]
