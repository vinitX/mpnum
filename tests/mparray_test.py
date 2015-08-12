#!/usr/bin/env python
# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.linalg import svd
from numpy.testing import assert_array_almost_equal, assert_array_equal, \
    assert_almost_equal, assert_equal
from six.moves import range # @UnresolvedImport

import mpnum.factory as factory
import mpnum.mparray as mp
from mpnum._tools import global_to_local, local_to_global
from mpnum import _tools


# nr_sites, local_dim, bond_dim
MP_TEST_PARAMETERS = [(6, 2, 4), (4, 3, 5), (5, 2, 1)]
# nr_sites, local_dim, bond_dim, sites_per_group
MP_TEST_PARAMETERS_GROUPS = [(6, 2, 4, 3), (6, 2, 4, 2), (4, 3, 5, 2)]


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


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_from_kron(nr_sites, local_dim, bond_dim):
    plegs = 2
    factors = tuple(factory._zrandn([nr_sites] + ([local_dim] * plegs)))
    op = _tools.mkron(*factors)
    op.shape = [local_dim] * (plegs * nr_sites)
    mpo = mp.MPArray.from_kron(factors)
    assert_array_almost_equal(op, mpo_to_global(mpo))


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
def test_inner_vec(nr_sites, local_dim, bond_dim):
    mp_psi1 = factory.random_mpa(nr_sites, local_dim, bond_dim)
    psi1 = mp_psi1.to_array().ravel()
    mp_psi2 = factory.random_mpa(nr_sites, local_dim, bond_dim)
    psi2 = mp_psi2.to_array().ravel()

    inner_np = np.vdot(psi1, psi2)
    inner_mp = mp.inner(mp_psi1, mp_psi2)
    assert_almost_equal(inner_mp, inner_np)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_inner_mat(nr_sites, local_dim, bond_dim):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op1 = mpo_to_global(mpo1).reshape((local_dim**nr_sites, ) * 2)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op2 = mpo_to_global(mpo2).reshape((local_dim**nr_sites, ) * 2)

    inner_np = np.trace(np.dot(op1.conj().transpose(), op2))
    inner_mp = mp.inner(mpo1, mpo2)
    assert_almost_equal(inner_mp, inner_np)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_norm(nr_sites, local_dim, bond_dim):
    mp_psi = factory.random_mpa(nr_sites, local_dim, bond_dim)
    psi = mp_psi.to_array()

    assert_almost_equal(mp.inner(mp_psi, mp_psi), mp.norm(mp_psi)**2)
    assert_almost_equal(np.sum(psi.conj() * psi), mp.norm(mp_psi)**2)


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
    scalar = np.random.randn()

    assert_array_almost_equal(scalar * op, mpo_to_global(scalar * mpo))

    mpo *= scalar
    assert_array_almost_equal(scalar * op, mpo_to_global(mpo))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_div_mpo_scalar(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)
    scalar = np.random.randn()

    assert_array_almost_equal(op / scalar, mpo_to_global(mpo / scalar))

    mpo /= scalar
    assert_array_almost_equal(op / scalar, mpo_to_global(mpo))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_outer(nr_sites, local_dim, bond_dim):
    # NOTE: Everything here is in local form!!!
    assert nr_sites > 1

    mpo = factory.random_mpa(nr_sites // 2, (local_dim, local_dim), bond_dim)
    op = mpo.to_array()

    # Test with 2-factors with full form
    mpo_double = mp.outer((mpo, mpo))
    op_double = np.tensordot(op, op, axes=(tuple(), ) * 2)
    assert len(mpo_double) == 2 * len(mpo)
    assert_array_almost_equal(op_double, mpo_double.to_array())
    assert_array_equal(mpo_double.bdims, mpo.bdims + (1,) + mpo.bdims)

    # Test 3-factors iteratively (since full form would be too large!!
    diff = mp.outer((mpo, mpo, mpo)) - mp.outer((mpo, mp.outer((mpo, mpo))))
    diff.normalize()
    assert len(diff) == 3 * len(mpo)
    assert mp.norm(diff) < 1e-6


###############################################################################
#                         Shape changes, conversions                          #
###############################################################################

@pt.mark.parametrize('nr_sites, local_dim, bond_dim, sites_per_group', MP_TEST_PARAMETERS_GROUPS)
def test_group_sites(nr_sites, local_dim, bond_dim, sites_per_group):
    assert (nr_sites % sites_per_group) == 0, \
        'nr_sites not a multiple of sites_per_group'
    mpa = factory.random_mpa(nr_sites, (local_dim,) * 2, bond_dim)
    grouped_mpa = mpa.group_sites(sites_per_group)
    op = mpa.to_array()
    grouped_op = grouped_mpa.to_array()
    assert_array_almost_equal(op, grouped_op)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, sites_per_group', MP_TEST_PARAMETERS_GROUPS)
def test_split_sites(nr_sites, local_dim, bond_dim, sites_per_group):
    assert (nr_sites % sites_per_group) == 0, \
        'nr_sites not a multiple of sites_per_group'
    mpa = factory.random_mpa(nr_sites // sites_per_group,
                             (local_dim,) * (2 * sites_per_group), bond_dim)
    split_mpa = mpa.split_sites(sites_per_group)
    op = mpa.to_array()
    split_op = split_mpa.to_array()
    assert_array_almost_equal(op, split_op)


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

    assert_equal(lnormal, lnormal_target)
    assert_equal(rnormal, rnormal_target)

    for n in range(lnormal):
        assert_lcannonical(mpo[n], msg="Failure left cannonical (n={}/{})"
                           .format(n, lnormal_target))
    for n in range(rnormal, len(mpo)):
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

    for site in range(1, nr_sites):
        mpo.normalize(left=site)
        assert_correct_normalzation(mpo, site, nr_sites)
        assert_array_almost_equal(op, mpo_to_global(mpo))

    for site in range(nr_sites - 1, 0, -1):
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
def test_full_normalization(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)
    assert_correct_normalzation(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    mpo.normalize(right=1)
    assert_correct_normalzation(mpo, 0, 1)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    ###########################################################################
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)
    assert_correct_normalzation(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    mpo.normalize(left=len(mpo) - 1)
    assert_correct_normalzation(mpo, len(mpo) - 1, len(mpo))
    assert_array_almost_equal(op, mpo_to_global(mpo))


def test_normalization_compression():
    """If the bond dimension is too large at the boundary, qr decompostion
    in normalization may yield smaller bond dimension"""
    mpo = factory.random_mpa(sites=2, ldim=2, bdim=20)
    mpo.normalize(right=1)
    assert_correct_normalzation(mpo, 0, 1)
    assert mpo.bdims[0] == 2

    mpo = factory.random_mpa(sites=2, ldim=2, bdim=20)
    mpo.normalize(left=1)
    assert_correct_normalzation(mpo, 1, 2)
    assert mpo.bdims[0] == 2


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mult_mpo_scalar_normalization(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)
    scalar = np.random.randn()

    center = nr_sites // 2
    mpo.normalize(left=center - 1, right=center)
    mpo_times_two = scalar * mpo

    assert_array_almost_equal(scalar * op, mpo_to_global(mpo_times_two))
    assert_correct_normalzation(mpo_times_two, center - 1, center)

    mpo *= scalar
    assert_array_almost_equal(scalar * op, mpo_to_global(mpo))
    assert_correct_normalzation(mpo, center - 1, center)


#####################
#  SVD compression  #
#####################
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd_trivial(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)

    mpo_new = mpo.copy()
    mpo_new.compress(bdim=10 * bond_dim, method='svd', direction='right')
    assert_array_equal(mpo.bdims, mpo_new.bdims)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))

    mpo_new = mpo.copy()
    mpo_new.compress(bdim=10 * bond_dim, method='svd', direction='left')
    assert_array_equal(mpo.bdims, mpo_new.bdims)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd_hard_cutoff(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    zero = factory.zero(nr_sites, (local_dim, local_dim), bond_dim)
    mpo_new = mpo + zero

    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    for bdims in zip(mpo.bdims, zero.bdims, mpo_new.bdims):
        assert_equal(bdims[0] + bdims[1], bdims[2])

    # Right-compression
    mpo_new = mpo + zero
    overlap = mpo_new.compress(bdim=bond_dim, method='svd', direction='right')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    assert_correct_normalzation(mpo_new, nr_sites - 1, nr_sites)
    # since no truncation error should occur
    assert_almost_equal(overlap, mp.norm(mpo)**2, decimal=5)

    # Left-compression
    mpo_new = mpo + zero
    overlap = mpo_new.compress(bdim=bond_dim, method='svd', direction='left')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    assert_correct_normalzation(mpo_new, 0, 1)
    # since no truncation error should occur
    assert_almost_equal(overlap, mp.norm(mpo)**2, decimal=5)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd_relerr(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    zero = factory.zero(nr_sites, (local_dim, local_dim), bond_dim)
    mpo_new = mpo + zero

    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    for bdims in zip(mpo.bdims, zero.bdims, mpo_new.bdims):
        assert_equal(bdims[0] + bdims[1], bdims[2])

    # Right-compression
    mpo_new = mpo + zero
    mpo_new.compress(relerr=1e-6, method='svd', direction='right')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    assert_correct_normalzation(mpo_new, nr_sites - 1, nr_sites)

    # Left-compression
    mpo_new = mpo + zero
    mpo_new.compress(relerr=1e-6, method='svd', direction='left')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    assert_correct_normalzation(mpo_new, 0, 1)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd_overlap(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    mpo_new = mpo.copy()

    # Catch superficious compression paramter
    max_bdim = max(bond_dim // 2, 1)

    overlap = mpo_new.compress(bdim=max_bdim, method='svd', direction='right')
    assert_almost_equal(overlap, mp.inner(mpo, mpo_new), decimal=5)
    assert all(bdim <= max_bdim for bdim in mpo_new.bdims)

    mpo_new = mpo.copy()
    overlap = mpo_new.compress(bdim=max_bdim, method='svd', direction='left')
    assert_almost_equal(overlap, mp.inner(mpo, mpo_new), decimal=5)
    assert all(bdim <= max_bdim for bdim in mpo_new.bdims)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd_compare(nr_sites, local_dim, bond_dim):
    randstate = np.random.RandomState(seed=46)
    mpa = factory.random_mpa(nr_sites, (local_dim,) * 2, bond_dim, randstate)
    target_bonddim = max(2 * bond_dim // 3, 1)
    directions = ('left', 'right')
    for direction in directions:
        target_array = svd_compression(mpa, direction, target_bonddim)
        mpa_compr = mpa.copy()
        mpa_compr.compress(method='svd', bdim=target_bonddim, direction=direction)
        array_compr = mpa_compr.to_array()
        assert_array_almost_equal(
            target_array, array_compr,
            err_msg='direction {0!r} failed'.format(direction))


def svd_compression(mpa, direction, target_bonddim):
    """Re-implement what SVD compression on MPAs does.

    Two implementations that produce the same data are not a guarantee
    for correctness, but a check for consistency is nice anyway.

    :param mpa: The MPA to compress
    :param direction: 'right' means sweep from left to right,
        'left' vice versa
    :param target_bonddim: Compress to this bond dimension
    :returns: Result as numpy.ndarray

    """
    array = mpa.to_array()
    plegs = mpa.plegs[0]
    nr_sites = len(mpa)
    if direction == 'right':
        nr_left_values = range(1, nr_sites)
    else:
        nr_left_values = range(nr_sites-1, 0, -1)
    for nr_left in nr_left_values:
        array = svd_compression_singlecut(array, nr_left, plegs, target_bonddim)
    return array


def svd_compression_singlecut(array, nr_left, plegs, target_bonddim):
    """
    SVD compression on a single left vs. right bipartition.

    :param array: The array to compress
    :param nr_left: Number of sites in the left part of the bipartition
    :param plegs: Number of physical legs per site
    :param target_bonddim: Compress to this bond dimension
    :returns: Result as numpy.ndarray (same shape as input)

    """
    array_shape = array.shape
    array = array.reshape((np.prod(array_shape[:nr_left * plegs]), -1))
    u, s, v = svd(array)
    u = u[:, :target_bonddim]
    s = s[:target_bonddim]
    v = v[:target_bonddim, :]
    opt_compr = np.dot(u * s, v)
    opt_compr = opt_compr.reshape(array_shape)
    return opt_compr


############################
# Variational compression  #
############################
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_var_trivial(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)

    # using internal initial vector
    mpo_new = mpo.copy()
    mpo_new.compress(method='var', bdim=10 * bond_dim)
    # since var. compression doesnt take into account the original bond dim
    assert all(d1 <= d2 for d1, d2 in zip(mpo.bdims, mpo_new.bdims))
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))

    # using an external initial vector
    mpo_new = mpo.copy()
    initmpa = factory.random_mpa(nr_sites, (local_dim, ) * 2, 10 * bond_dim)
    initmpa *= mp.norm(mpo) / mp.norm(initmpa)
    mpo_new.compress(method='var', initmpa=initmpa)
    assert all(d1 <= d2 for d1, d2 in zip(mpo.bdims, mpo_new.bdims))
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_var_hard_cutoff(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    zero = factory.zero(nr_sites, (local_dim, local_dim), bond_dim)
    mpo_new = mpo + zero

    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    for bdims in zip(mpo.bdims, zero.bdims, mpo_new.bdims):
        assert_equal(bdims[0] + bdims[1], bdims[2])

    mpo_new = mpo + zero
    initmpa = factory.random_mpa(nr_sites, (local_dim, ) * 2, bond_dim)
    overlap = mpo_new.compress(method='var', initmpa=initmpa)
    #  overlap = mpo_new.compress(bdim=bond_dim, method='var')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    # FIXME assert_correct_normalzation(mpo_new, nr_sites - 1, nr_sites)
    # since no truncation error should occur
    # FIXME assert_almost_equal(overlap, mp.norm(mpo)**2, decimal=5)

# FIXME
#  @pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
#  def test_compression_var_relerr(nr_sites, local_dim, bond_dim):
#      mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
#      zero = factory.zero(nr_sites, (local_dim, local_dim), bond_dim)
#      mpo_new = mpo + zero

#      assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
#      for bdims in zip(mpo.bdims, zero.bdims, mpo_new.bdims):
#          assert_equal(bdims[0] + bdims[1], bdims[2])

#      # Right-compression
#      mpo_new = mpo + zero
#      mpo_new.compress(relerr=1e-6, method='var', direction='right')
#      assert_array_equal(mpo_new.bdims, bond_dim)
#      assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
#      assert_correct_normalzation(mpo_new, nr_sites - 1, nr_sites)

#      # Left-compression
#      mpo_new = mpo + zero
#      mpo_new.compress(relerr=1e-6, method='var', direction='left')
#      assert_array_equal(mpo_new.bdims, bond_dim)
#      assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
#      assert_correct_normalzation(mpo_new, 0, 1)

# FIXME
#  @pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
#  def test_compression_var_overlap(nr_sites, local_dim, bond_dim):
#      mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
#      mpo_new = mpo.copy()

#      # Catch superficious compression paramter
#      max_bdim = max(bond_dim // 2, 1)

#      overlap = mpo_new.compress(max_bdim=max_bdim, method='var', direction='right')
#      assert_almost_equal(overlap, mp.inner(mpo, mpo_new), decimal=5)
#      assert all(bdim <= max_bdim for bdim in mpo_new.bdims)

#      mpo_new = mpo.copy()
#      overlap = mpo_new.compress(max_bdim=max_bdim, method='var', direction='left')
#      assert_almost_equal(overlap, mp.inner(mpo, mpo_new), decimal=5)
#      assert all(bdim <= max_bdim for bdim in mpo_new.bdims)

#  @pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
#  def test_variational_compression(nr_sites, local_dim, bond_dim):
#      randstate = np.random.RandomState(seed=42)
#      mpa = factory.random_mpa(nr_sites, (local_dim,) * 2, bond_dim, randstate)
#      mpa /= mp.norm(mpa)
#      array = mpa.to_array()
#      target_bonddim = max(2 * bond_dim // 3, 1)

#      right_svd_res = svd_compression(mpa, 'right', target_bonddim)
#      left_svd_res = svd_compression(mpa, 'left', target_bonddim)
#      right_svd_overlap = np.abs(np.dot(array.conj().flatten(), right_svd_res.flatten()))
#      left_svd_overlap = np.abs(np.dot(array.conj().flatten(), left_svd_res.flatten()))

#      # max_num_sweeps = 3 and 4 is sometimes not good enough.
#      mpa_compr = mpnum.linalg.variational_compression(
#          mpa, max_num_sweeps=5,
#          startvec_bonddim=target_bonddim, randstate=randstate)
#      mpa_compr_overlap = np.abs(np.dot(array.conj().flatten(),
#                                        mpa_compr.to_array().flatten()))

#      # The basic intuition is that variational compression, given
#      # enough sweeps, should be at least as good as left and right SVD
#      # compression because the SVD compression scheme has a strong
#      # interdependence between truncations at the individual sites,
#      # while variational compression does not have that. Therefore, we
#      # check exactly that.

#      overlap_rel_tol = 1e-6
#      assert mpa_compr_overlap >= right_svd_overlap * (1 - overlap_rel_tol)
#      assert mpa_compr_overlap >= left_svd_overlap * (1 - overlap_rel_tol)


#  @pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
#  def test_variational_compression_twosite(nr_sites, local_dim, bond_dim):
#      randstate = np.random.RandomState(seed=42)
#      mpa = factory.random_mpa(nr_sites, (local_dim,) * 2, bond_dim, randstate)
#      mpa /= mp.norm(mpa)
#      array = mpa.to_array()
#      target_bonddim = max(2 * bond_dim // 3, 1)

#      right_svd_res = svd_compression(mpa, 'right', target_bonddim)
#      left_svd_res = svd_compression(mpa, 'left', target_bonddim)
#      right_svd_overlap = np.abs(np.dot(array.conj().flatten(), right_svd_res.flatten()))
#      left_svd_overlap = np.abs(np.dot(array.conj().flatten(), left_svd_res.flatten()))

#      # With minimize_sites = 1, max_num_sweeps = 3 and 4 is sometimes
#      # not good enough. With minimiza_sites = 2, max_num_sweeps = 2 is
#      # fine.
#      mpa_compr = mpnum.linalg.variational_compression(
#          mpa, startvec_bonddim=target_bonddim, randstate=randstate,
#          max_num_sweeps=3, minimize_sites=2)
#      mpa_compr_overlap = np.abs(np.dot(array.conj().flatten(),
#                                        mpa_compr.to_array().flatten()))

#      # The basic intuition is that variational compression, given
#      # enough sweeps, should be at least as good as left and right SVD
#      # compression because the SVD compression scheme has a strong
#      # interdependence between truncations at the individual sites,
#      # while variational compression does not have that. Therefore, we
#      # check exactly that.

#      overlap_rel_tol = 1e-6
#      assert mpa_compr_overlap >= right_svd_overlap * (1 - overlap_rel_tol)
#      assert mpa_compr_overlap >= left_svd_overlap * (1 - overlap_rel_tol)

