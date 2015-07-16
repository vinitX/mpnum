#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import numpy as np
import pytest as pt
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mpnum.factory as factory
import mpnum.mparray as mp
from mpnum._tools import global_to_local, local_to_global

# List of test parameters (sites, local_dim, bond_dim)
MP_TEST_PARAMETERS = [(6, 2, 4), (4, 3, 5)]


# We choose to use a global reperentation of multipartite arrays throughout our
# tests to be consistent and a few operations (i.e. matrix multiplication) are
# easier to express
def mpo_to_global(mpo):
    return local_to_global(mpo.to_array(), len(mpo))


###############################################################################
#                         Basic creation & operations                         #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_from_full(nr_sites, local_dim, _):
    psi = factory.random_vec(nr_sites, local_dim)
    mps = mp.MPArray.from_array(psi, 1)
    assert_array_almost_equal(psi, mps.to_array())

    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    assert_array_almost_equal(op, mpo.to_array())


@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_conjugations(nr_sites, local_dim, _):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    assert_array_almost_equal(np.conj(op), mpo.C().to_array())


@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_transposition(nr_sites, local_dim, _):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(global_to_local(op, nr_sites), 2)

    opT = op.reshape((local_dim**nr_sites,) * 2).T \
        .reshape((local_dim,) * 2 * nr_sites)
    assert_array_almost_equal(opT, mpo_to_global(mpo.T()))


###############################################################################
#                            Algebraic operations                             #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_dot(nr_sites, local_dim, bond_dim):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op1 = mpo_to_global(mpo1)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op2 = mpo_to_global(mpo2)

    # Dotproduct of all 1st physical with 0th physical legs = np.dot
    dot_np = np.tensordot(op1.reshape((local_dim**nr_sites, ) * 2),
                          op2.reshape((local_dim**nr_sites, ) * 2),
                          axes=([1], [0]))
    dot_np = dot_np.reshape(op1.shape)
    dot_mp = mpo_to_global(mp.dot(mpo1, mpo2, axes=(1, 0)))
    assert_array_almost_equal(dot_np, dot_mp)
    # this should also be the default axes
    dot_mp = mpo_to_global(mp.dot(mpo1, mpo2))
    assert_array_almost_equal(dot_np, dot_mp)

    # Dotproduct of all 0th physical with 1st physical legs = np.dot
    dot_np = np.tensordot(op1.reshape((local_dim**nr_sites, ) * 2),
                          op2.reshape((local_dim**nr_sites, ) * 2),
                          axes=([0], [1]))
    dot_np = dot_np.reshape(op1.shape)
    dot_mp = mpo_to_global(mp.dot(mpo1, mpo2, axes=(0, 1)))
    assert_array_almost_equal(dot_np, dot_mp)
    # this should also be the default axes
    dot_mp = mpo_to_global(mp.dot(mpo1, mpo2, axes=(-2, -1)))
    assert_array_almost_equal(dot_np, dot_mp)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_add_and_subtr(nr_sites, local_dim, bond_dim):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op1 = mpo_to_global(mpo1)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op2 = mpo_to_global(mpo2)

    assert_array_almost_equal(op1 + op2, mpo_to_global(mpo1 + mpo2))
    assert_array_almost_equal(op1 - op2, mpo_to_global(mpo1 - mpo2))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mult_mpo_scalar(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)

    assert_array_almost_equal(2 * op, mpo_to_global(2 * mpo))


###############################################################################
#                         Normalization & Compression                         #
###############################################################################
def assert_lcannonical(ltens, msg=''):
    ltens = ltens.reshape((np.prod(ltens.shape[:-1]), ltens.shape[-1]))
    prod = ltens.conj().T.dot(ltens)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                                         err_msg=msg)


def assert_rcannonical(ltens, msg=''):
    ltens = ltens.reshape((ltens.shape[0], np.prod(ltens.shape[1:])))
    prod = ltens.dot(ltens.conj().T)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                                         err_msg=msg)


def assert_correct_normalzation(mpo, lnormal_target, rnormal_target):
    lnormal, rnormal = mpo.normal_form

    assert lnormal == lnormal_target, \
        "Wrong lnormal={} != {}".format(lnormal, lnormal_target)
    assert rnormal == rnormal_target, \
        "Wrong rnormal={} != {}".format(rnormal, rnormal_target)

    for n in xrange(lnormal):
        assert_lcannonical(mpo[n], msg="Failure left cannonical (n={}/{})"
                           .format(n, lnormal_target))
    for n in xrange(rnormal, len(mpo)):
        assert_rcannonical(mpo[n], msg="Failure right cannonical (n={}/{})"
                           .format(n, rnormal_target))


@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_from_full_normalization(nr_sites, local_dim, _):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    assert_correct_normalzation(mpo, nr_sites - 1, nr_sites)


# FIXME Add counter to normalization functions
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_incremental_normalization(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)
    assert_correct_normalzation(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    for site in xrange(1, nr_sites):
        mpo.normalize(left=site)
        assert_correct_normalzation(mpo, site, nr_sites)
        assert_array_almost_equal(op, mpo_to_global(mpo))

    for site in xrange(nr_sites - 1, 0, -1):
        mpo.normalize(right=site)
        assert_correct_normalzation(mpo, site - 1, site)
        assert_array_almost_equal(op, mpo_to_global(mpo))


# FIXME Add counter to normalization functions
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_jump_normalization(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)
    assert_correct_normalzation(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    center = nr_sites // 2
    mpo.normalize(left=center - 1, right=center)
    assert_correct_normalzation(mpo, center - 1, center)
    assert_array_almost_equal(op, mpo_to_global(mpo))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    zero = factory.zero_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    mpo_new = mpo + zero

    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    for bdims in zip(mpo.bdims, zero.bdims, mpo_new.bdims):
        assert bdims[0] + bdims[1] == bdims[2]

    # Right-compression
    mpo_new.compress(max_bdim=bond_dim, method='svd', direction='right')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    assert_correct_normalzation(mpo_new, nr_sites - 1, nr_sites)

    # Left-compression
    # mpo_new = mpo + zero
    mpo_new.compress(max_bdim=bond_dim, method='svd', direction='left')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    assert_correct_normalzation(mpo_new, 0, 1)
