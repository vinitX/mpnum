# encoding: utf-8
"""Module contains some specialiced versions of some functions from mparray.
They are tuned for speed with special applications in mind
"""
from __future__ import absolute_import, division, print_function

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
    ltens1 = iter(mpa1)
    ltens2 = iter(mpa2)

    res = np.dot(next(ltens1)[0, :, 0].conj(), next(ltens2))
    for l1, l2 in zip(ltens1, ltens2):
        res = np.dot(res, np.dot(l1[0, :, 0].conj(), l2))
    return res[0, 0]


def sumup(mpas, weights=None):
    """Same as :func:`mparray.sumup`, but with extended weighting & compression
    options

    :param mpas: Iterator over MPArrays
    :param weights: Iterator of same length as mpas containing weights for
        computing weighted sum.
    :returns: Sum of `mpas`

    """
    mpas = list(mpas)
    length = len(mpas[0])
    weights = weights if weights is not None else np.ones(len(mpas))

    assert all(len(mpa) == length for mpa in mpas)
    assert len(weights) == len(mpas)

    if length == 1:
        # The code below assumes at least two sites.
        return mp.MPArray((sum(w * mpa[0] for w, mpa in zip(weights, mpas)),))

    ltensiter = [iter(mpa) for mpa in mpas]
    ltens = [np.concatenate([w * next(lt) for w, lt in zip(weights, ltensiter)], axis=-1)]
    ltens += [mp._local_add([next(lt) for lt in ltensiter])
              for _ in range(length - 2)]
    ltens += [np.concatenate([next(lt) for lt in ltensiter], axis=0)]

    return mp.MPArray(ltens)

