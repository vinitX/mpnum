# encoding: utf-8
"""Auxiliary functions useful for writing tests"""


import itertools as it

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
from scipy import sparse


def assert_mpa_almost_equal(mpa1, mpa2, full=False, **kwargs):
    """Verify that two MPAs are almost equal
    """
    if full:
        assert_array_almost_equal(mpa1.to_array(), mpa2.to_array(), **kwargs)
    else:
        raise NotImplementedError("Coming soon...")


def assert_mpa_identical(mpa1, mpa2, decimal=np.infty):
    """Verify that two MPAs are complety identical
    """
    assert len(mpa1) == len(mpa2)
    assert mpa1.canonical_form == mpa2.canonical_form
    assert mpa1.dtype == mpa2.dtype

    for i, lten1, lten2 in zip(it.count(), mpa1.lt, mpa2.lt):
        if decimal is np.infty:
            assert_array_equal(lten1, lten2,
                               err_msg='mismatch in lten {}'.format(i))
        else:
            assert_array_almost_equal(lten1, lten2, decimal=decimal,
                                      err_msg='mismatch in lten {}'.format(i))
    # TODO: We should make a comprehensive comparison between `mpa1`
    # and `mpa2`.  Are we missing other things?


def _assert_lcanonical(ltens, msg=''):
    ltens = ltens.reshape((np.prod(ltens.shape[:-1]), ltens.shape[-1]))
    prod = ltens.conj().T.dot(ltens)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                              err_msg=msg)


def _assert_rcanonical(ltens, msg=''):
    ltens = ltens.reshape((ltens.shape[0], np.prod(ltens.shape[1:])))
    prod = ltens.dot(ltens.conj().T)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                              err_msg=msg)


def assert_correct_normalization(lt, lcanon_target=None, rcanon_target=None):
    """Verify that normalization info in `lt` is correct

    We check that `lt` is at least as normalized as specified by the
    information. `lt` being "more normalized" than the information
    specifies is admissible and not treated as an error.

    If `[lr]canon_target` are not None, verify that normalization
    info is exactly equal to the given values.

    """
    if hasattr(lt, 'lt'):
        lt = lt.lt  # We got an MPArray in lt, retrieve mpa.lt
    lnormal, rnormal = lt.canonical_form

    # Verify that normalization info is correct
    for n in range(lnormal):
        _assert_lcanonical(lt[n], msg="Wrong left canonical (n={}/{})"
                           .format(n, lnormal))
    for n in range(rnormal, len(lt)):
        _assert_rcanonical(lt[n], msg="Wrong right canonical (n={}/{})"
                           .format(n, rnormal))

    # If targets are given, verify that the information in
    # `lt.canonical_form` matches the targets.
    if lcanon_target is not None:
        assert_equal(lnormal, lcanon_target)
    if rcanon_target is not None:
        assert_equal(rnormal, rcanon_target)


def compression_svd(array, rank, direction='right', retproj=False):
    """Re-implement MPArray.compress('svd') but on the level of the dense
    array representation, i.e. it truncates the Schmidt-decompostion
    on each bipartition sequentially.

    :param mpa: Array to compress
    :param rank: Compress to this rank
    :param direction: 'right' means sweep from left to right, 'left' vice versa
    :param retproj: Besides the compressed array, also return the projectors
        on the appropriate eigenspaces
    :returns: Result as numpy.ndarray

    """
    def singlecut(array, nr_left, target_rank):
        array_shape = array.shape
        array = array.reshape((np.prod(array_shape[:nr_left]), -1))
        u, s, vt = np.linalg.svd(array, full_matrices=False)
        u = u[:, :target_rank]
        s = s[:target_rank]
        vt = vt[:target_rank, :]
        opt_compr = np.dot(u * s, vt)
        opt_compr = opt_compr.reshape(array_shape)

        if retproj:
            projector_l = np.dot(u, u.T.conj())
            projector_r = np.dot(vt.T.conj(), vt)
            return opt_compr, (projector_l, projector_r)
        else:
            return opt_compr, (None, None)

    nr_sites = array.ndim
    projectors = []
    if direction == 'right':
        nr_left_values = range(1, nr_sites)
    else:
        nr_left_values = range(nr_sites-1, 0, -1)

    for nr_left in nr_left_values:
        array, proj = singlecut(array, nr_left, rank)
        projectors.append(proj)

    if direction != 'right':
        projectors = projectors.reverse()

    return (array, projectors) if retproj else array



def random_lowrank(rows, cols, rank, randstate=np.random, dtype=np.float_):
    """Returns a random lowrank matrix of given shape and dtype"""
    if dtype == np.float_:
        A = randstate.randn(rows, rank)
        B = randstate.randn(cols, rank)
    elif dtype == np.complex_:
        A = randstate.randn(rows, rank) + 1.j * randstate.randn(rows, rank)
        B = randstate.randn(cols, rank) + 1.j * randstate.randn(cols, rank)
    else:
        raise ValueError("{} is not a valid dtype".format(dtype))

    C = A.dot(B.conj().T)
    return C / np.linalg.norm(C)


def random_fullrank(rows, cols, **kwargs):
    """Returns a random matrix of given shape and dtype. Should provide
    same interface as random_lowrank"""
    kwargs.pop('rank', None)
    return random_lowrank(rows, cols, min(rows, cols), **kwargs)
