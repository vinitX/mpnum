# encoding: utf-8
"""Optimized functions

Module contains some specialiced versions of some functions from mparray.
They are tuned for speed with special applications in mind
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as ssp

from . import mparray as mp
from .utils import truncated_svd
from .mpstruct import LocalTensors

__all__ = ['inner_prod_mps', 'sumup']


def inner_prod_mps(mpa1, mpa2):
    """Same as :func:`mparray.inner`, but assumes that `mpa1` is a product MPS

    :param mpa1: MPArray with one leg per site and rank 1
    :param mpa2: MPArray with same shape as mpa1 but arbitrary rank
    :returns: <mpa1|mpa2>

    """
    assert all(rank == 1 for rank in mpa1.ranks)
    assert all(dim == 1 for dim in mpa1.ndims)
    assert all(dim == 1 for dim in mpa2.ndims)

    # asssume mpa1 is product
    ltens1 = iter(mpa1.lt)
    ltens2 = iter(mpa2.lt)

    res = np.dot(next(ltens1)[0, :, 0].conj(), next(ltens2))
    for l1, l2 in zip(ltens1, ltens2):
        res = np.dot(res, np.dot(l1[0, :, 0].conj(), l2))
    return res[0, 0]


def sumup(mpas, rank, weights=None, svdfunc=truncated_svd):
    """Same as :func:`mparray.sumup` with a consequent compression, but with
    in-place svd compression.  Also, we use a sparse-matrix format for the
    intermediate local tensors of the sum. Therefore, the memory footprint
    scales only linearly in the number of summands (instead of quadratically).

    Right now, only the sum of product tensors is supported.

    :param mpas: Iterator over MPArrays
    :param rank: Rank of the final result.
    :param weights: Iterator of same length as mpas containing weights for
        computing weighted sum (default: None)
    :param svdfunc: Function implementing the truncated svd, for required
        signature see :func:`truncated_svd`.
    :returns: Sum of `mpas` with max. rank `rank`

    Possible values for :code:`svdfunc` include:

    - :func:`truncated_svd`: Almost no speedup compared to the standard
      sumup and compression, since it computes the full SVD
    - :func:`scipy.sparse.linalg.svds`: Only computes the necessary
      singular values/vectors, but slow if `rank` is not small enough
    - :func:`mpnum.utils.extmath.randomized_svd`: Randomized truncated
      SVD, fast and efficient, but only approximation.
    """
    mpas = list(mpas)
    length = len(mpas[0])
    nr_summands = len(mpas)
    weights = weights if weights is not None else np.ones(nr_summands)

    assert all(len(mpa) == length for mpa in mpas)
    assert len(weights) == len(mpas)

    if length == 1:
        # The code below assumes at least two sites.
        return mp.MPArray((sum(w * mpa.lt[0] for w, mpa in zip(weights, mpas)),))

    assert all(max(mpa.ranks) == 1 for mpa in mpas)

    ltensiter = [iter(mpa.lt) for mpa in mpas]

    if weights is None:
        summands = [next(lt) for lt in ltensiter]
    else:
        summands = [(w * next(lt)) for w, lt in zip(weights, ltensiter)]

    current = np.concatenate(summands, axis=-1)
    u, sv, v = svdfunc(current.reshape((-1, nr_summands)), rank)
    ltens = [u.reshape((1, -1, len(sv)))]

    for sites in range(1, length - 1):
        current = _local_add_sparse([next(lt).ravel() for lt in ltensiter])
        current = ((sv[:, None] * v) * current).reshape((-1, nr_summands))
        rank_t = min(rank, *current.shape)
        u, sv, v = svdfunc(current.reshape((-1, nr_summands)), rank_t)
        ltens.append(u.reshape((ltens[-1].shape[-1], -1, rank_t)))

    current = np.concatenate([next(lt) for lt in ltensiter], axis=0) \
        .reshape((nr_summands, -1))
    current = np.dot(sv[:, None] * v, current)
    ltens.append(current.reshape((len(sv), -1, 1)))

    result_ltens = LocalTensors(ltens, cform=(len(ltens) - 1, None))
    result = mp.MPArray(result_ltens)
    return result.reshape(mpas[0].shape)


def _local_add_sparse(ltenss):
    """Computes the local tensors of a sum of MPArrays (except for the boundary
    tensors). Works only for products right now

    :param ltenss: Raveled local tensors
    :returns: Correct local tensor representation

    """
    dim = len(ltenss[0])
    nr_summands = len(ltenss)

    indptr = np.arange(nr_summands * dim + 1)
    indices = np.concatenate((np.arange(nr_summands),) * dim)
    data = np.concatenate([lt[None, :] for lt in ltenss])
    data = np.rollaxis(data, 1).ravel()

    return ssp.csc_matrix((data, indices, indptr),
                          shape=(nr_summands, dim * nr_summands))
