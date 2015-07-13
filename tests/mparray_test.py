#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import pytest as pt
import numpy as np
import mptom.mparray as mp
import mptom.factory as factory
import mptom._qmtools as qm


# List of test parameters (sites, local_dim) that still allow for treatment of
# full representation
FULL_TEST_PARAMETERS = [(6, 2), (4, 3)]

# List of test parameters (sites, local_dim, bond_dim) for efficient matrix-
# product representation tests
MP_TEST_PARAMETERS = [(6, 2, 4), (4, 3, 5)]


###############################################################################
#                         Basic creation & operations                         #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim', FULL_TEST_PARAMETERS)
def test_from_full(nr_sites, local_dim):
    psi = factory.random_vec(nr_sites, local_dim)
    mps = mp.MPArray.from_array(psi, 1)
    np.testing.assert_array_almost_equal(psi, mps.to_array())

    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    np.testing.assert_array_almost_equal(op, mpo.to_array())


@pt.mark.parametrize('nr_sites, local_dim', FULL_TEST_PARAMETERS)
def test_conjugations(nr_sites, local_dim):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    np.testing.assert_array_almost_equal(np.conj(op), mpo.C().to_array())


@pt.mark.parametrize('nr_sites, local_dim', FULL_TEST_PARAMETERS)
def test_transposition(nr_sites, local_dim):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(qm.global_to_local(op, nr_sites), 2)

    opT = op.reshape((local_dim**nr_sites,) * 2).T \
        .reshape((local_dim,) * 2 * nr_sites)
    res = qm.local_to_global(mpo.T().to_array(), nr_sites)
    np.testing.assert_array_almost_equal(opT, res)


###############################################################################
#                            Algebraic operations                             #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_dot(nr_sites, local_dim, bond_dim):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op1 = qm.local_to_global(mpo1.to_array(), nr_sites)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op2 = qm.local_to_global(mpo2.to_array(), nr_sites)

    # Dotproduct of all 1st physical with 0th physical legs = np.dot
    dot_np = np.tensordot(op1.reshape((local_dim**nr_sites, ) * 2),
                          op2.reshape((local_dim**nr_sites, ) * 2),
                          axes=([1], [0]))
    dot_np = dot_np.reshape(op1.shape)
    dot_mp = mp.dot(mpo1, mpo2, axes=(1, 0)).to_array()
    dot_mp = qm.local_to_global(dot_mp, nr_sites)
    np.testing.assert_array_almost_equal(dot_np, dot_mp)
    # this should also be the default axes
    dot_mp = mp.dot(mpo1, mpo2).to_array()
    dot_mp = qm.local_to_global(dot_mp, nr_sites)
    np.testing.assert_array_almost_equal(dot_np, dot_mp)

    # Dotproduct of all 0th physical with 1st physical legs = np.dot
    dot_np = np.tensordot(op1.reshape((local_dim**nr_sites, ) * 2),
                          op2.reshape((local_dim**nr_sites, ) * 2),
                          axes=([0], [1]))
    dot_np = dot_np.reshape(op1.shape)
    dot_mp = mp.dot(mpo1, mpo2, axes=(0, 1)).to_array()
    dot_mp = qm.local_to_global(dot_mp, nr_sites)
    np.testing.assert_array_almost_equal(dot_np, dot_mp)
    # this should also be the default axes
    dot_mp = mp.dot(mpo1, mpo2, axes=(-2, -1)).to_array()
    dot_mp = qm.local_to_global(dot_mp, nr_sites)
    np.testing.assert_array_almost_equal(dot_np, dot_mp)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_add_and_subtr(nr_sites, local_dim, bond_dim):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op1 = qm.local_to_global(mpo1.to_array(), nr_sites)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op2 = qm.local_to_global(mpo2.to_array(), nr_sites)

    sum_mp = qm.local_to_global((mpo1 + mpo2).to_array(), nr_sites)
    np.testing.assert_array_almost_equal(op1 + op2, sum_mp)

    sum_mp = qm.local_to_global((mpo1 - mpo2).to_array(), nr_sites)
    np.testing.assert_array_almost_equal(op1 - op2, sum_mp)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mult_mpo_scalar(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = qm.local_to_global(mpo.to_array(), nr_sites)

    res = qm.local_to_global((2 * mpo).to_array(), nr_sites)
    np.testing.assert_array_almost_equal(2 * op, res)


###############################################################################
#                         Normalization & Compression                         #
###############################################################################
def assert_lcannonical(ltens):
    ltens = ltens.reshape((np.prod(ltens.shape[:-1]), ltens.shape[-1]))
    prod = ltens.conj().T.dot(ltens)
    np.testing.assert_array_almost_equal(prod, np.identity(prod.shape[0]))


def assert_rcannonical(ltens):
    ltens = ltens.reshape((ltens.shape[0], np.prod(ltens.shape[1:])))
    prod = ltens.dot(ltens.conj().T)
    np.testing.assert_array_almost_equal(prod, np.identity(prod.shape[0]))


def assert_correct_normalzation(mpo, lnormal_target, rnormal_target):
    lnormal, rnormal = mpo.normal_form

    assert lnormal == lnormal_target, \
        "Wrong lnormal={} != {}".format(lnormal, lnormal_target)
    assert rnormal == rnormal_target, \
        "Wrong rnormal={} != {}".format(rnormal, rnormal_target)

    for n, ltens in enumerate(mpo[:lnormal]):
        assert_lcannonical(ltens)
    for n, ltens in enumerate(mpo[rnormal:]):
        assert_rcannonical(ltens)


@pt.mark.parametrize('nr_sites, local_dim', FULL_TEST_PARAMETERS)
def test_from_full_normalization(nr_sites, local_dim):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    assert_correct_normalzation(mpo, nr_sites - 1, nr_sites)
