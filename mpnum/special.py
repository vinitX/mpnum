# encoding: utf-8
"""Module contains some specialiced versions of some functions from mparray.
They are tuned for speed with special applications in mind
"""
from __future__ import absolute_import, division, print_function

from math import ceil

import numpy as np

from . import mparray as mp


def inner_prod_mps(mpa1, mpa2):
    """Same as :func:`mparray.inner`, but assumes that `mpa1` is a product MPS

    :param mpa1: MPArray with one physical leg per site and bond dimension 1
    :param mpa2: MPArray with same physical shape as mpa1
    :returns: <mpa1|mpa2>

    """
    assert all(bdim == 1 for bdim in mpa1.bdims)
    assert all(pleg == 1 for pleg in mpa1.plegs)
    assert all(pleg == 1 for pleg in mpa2.plegs)

    # asssume mpa1 is product
    ltens1 = iter(mpa1.lt)
    ltens2 = iter(mpa2.lt)

    res = np.dot(next(ltens1)[0, :, 0].conj(), next(ltens2))
    for l1, l2 in zip(ltens1, ltens2):
        res = np.dot(res, np.dot(l1[0, :, 0].conj(), l2))
    return res[0, 0]


def sumup2(mpas, weights=None, max_bdim=None, compargs=None,
           min_overlap=None, max_n_sum=None):
    mpas = list(mpas)
    assert len(mpas) > 0
    if len(mpas) == 1:
        return mpas[0] if weights is None else weights[0] * mpas[0]
    weights_now = None
    out = []
    bdims = np.array([m.bdims for m in mpas])
    old_n = len(mpas)
    while mpas:
        cbdims = np.cumsum(bdims, axis=0)
        n = np.nonzero((cbdims <= max_bdim).all(axis=1))[0][-1] + 1
        if max_n_sum is not None:
            n = min(max_n_sum, n)
        mpas_now, mpas = mpas[:n], mpas[n:]
        bdims = bdims[n:]
        if weights is not None:
            weights_now, weights = weights[:n], weights[n:]
        print('summing', n, 'mps')
        s = mp.sumup(mpas_now, weights_now)
        s_norm = mp.norm(s)
        old_bdim = s.bdim
        overlap = abs(s.compress(**compargs))
        s_norm2 = mp.norm(s)
        overlap /= s_norm * s_norm2
        if min_overlap is not None:
            print('comp bdim {} -> {}'.format(old_bdim, s.bdim))
            print('observed / minimal overl:', overlap, min_overlap)
            print('observed / maximal error:', 1 - overlap, 1 - min_overlap)
            assert overlap >= min_overlap
        s *= s_norm / s_norm2
        out.append(s)
    print('sumup2', old_n, '->', len(out), 'with bdim', [m.bdim for m in out])
    return sumup2(out, None, max_bdim, compargs, min_overlap, max_n_sum)


def sumup(mpas, weights=None, target_bdim=None, max_bdim=None,
          compargs={'method': 'svd'}):
    """Same as :func:`mparray.sumup`, but with extended weighting & compression
    options. Supports intermediate compression.

    Make sure that max_bdim >> target_bdim. Also, this function really flies
    only with python optimizations enabled.

    :param mpas: Iterator over MPArrays
    :param weights: Iterator of same length as mpas containing weights for
        computing weighted sum.
    :param target_bdim: Bond dimension the result should have (default: `None`)
    :param max_bdim: Maximum bond dimension the intermediate steps may assume
        (default: `None`)
    :param compargs: Arguments for compression function
        (default: `{'method': 'svd'}`)
    :returns: Sum of `mpas` with max. bond dimension `target_bdim`

    """
    mpas = list(mpas)
    length = len(mpas[0])
    weights = weights if weights is not None else np.ones(len(mpas))

    assert all(len(mpa) == length for mpa in mpas)
    assert len(weights) == len(mpas)
    assert 'bdim' not in compargs

    if length == 1:
        # The code below assumes at least two sites.
        return mp.MPArray((sum(w * mpa.lt[0] for w, mpa in zip(weights, mpas)),))

    assert all(mpa.bdim == 1 for mpa in mpas)
    return _sum_n_compress(mpas, weights, 1, target_bdim, max_bdim, compargs)


def _sum_n_compress(mpas, weights, current_bdim, target_bdim, max_bdim, compargs):
    """Recursively sum and compress the MPArrays' in `mpas`. The end result
    will have bond dimension smaller than `target_bdim` and during the process,
    the intermediate results always have bond dimension smaller than
    `max_bdim`.

    :param mpas: List of MPArrays to sum
    :param weights: Optional weights to compute weighted sum. If `None` is
        passed, they are all assumed to be 1.
    :param current_bdim: The maximal bond dimension of any MPArray in `mpas`
    :param target_bdim: Bond dimension the end result should have at most.
    :param max_bdim: Max bond dimension any intermediate result should have
    :param compargs: Compression args to be used.

    """
    length = len(mpas[0])
    # no subpartition ne
    if (max_bdim is None) or (len(mpas) * current_bdim <= max_bdim):
        ltensiter = [iter(mpa.lt) for mpa in mpas]
        if weights is None:
            ltens = [np.concatenate([next(lt) for lt in ltensiter], axis=-1)]
        else:
            ltens = [np.concatenate([w * next(lt) for w, lt in zip(weights, ltensiter)], axis=-1)]
        ltens += [mp._local_add([next(lt) for lt in ltensiter])
                for _ in range(length - 2)]
        ltens += [np.concatenate([next(lt) for lt in ltensiter], axis=0)]

        summed = mp.MPArray(ltens)
        summed.compress(bdim=target_bdim, **compargs)
        return summed

    else:
        nodes = max(max_bdim // target_bdim, 1)
        stride = ceil(len(mpas) / nodes)
        partition = [slice(n * stride, (n + 1) * stride) for n in range(nodes)
                     if n * stride < len(mpas)]
        mpas = [_sum_n_compress(mpas[sel], weights[sel], 1, target_bdim,
                                max_bdim, compargs)
                for sel in partition]
        return _sum_n_compress(mpas, None, target_bdim, target_bdim, max_bdim,
                               compargs)
