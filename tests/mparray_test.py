#!/usr/bin/env python
# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import division, print_function

import numpy as np
import pytest as pt
from numpy.testing import assert_array_almost_equal, assert_array_equal, \
    assert_almost_equal, assert_equal
from six.moves import range # @UnresolvedImport

import mpnum.factory as factory
import mpnum.mparray as mp
from mpnum._tools import global_to_local, local_to_global
from mpnum import _tools


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


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, keep_width', [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_partial_trace_operator(nr_sites, local_dim, bond_dim, keep_width):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)

    startsites = range(nr_sites - keep_width + 1)
    for startsite, reduced_mpo in mp.partialtrace_operator(mpo, startsites, keep_width):
        traceout = tuple(range(startsite)) \
            + tuple(range(startsite + keep_width, nr_sites))
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(mpo_to_global(reduced_mpo), red_from_op,
                                  err_msg="not equal at startsite {}".format(startsite))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, keep_width', [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_partial_local_purification_mps(nr_sites, local_dim, bond_dim, keep_width):
    mps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mp.local_purification_mps_to_mpo(mps))

    startsites = range(nr_sites - keep_width + 1)
    for startsite, reduced_mps in mp.partialtrace_local_purification_mps(
            mps, startsites, keep_width):
        reduced_mpo = mp.local_purification_mps_to_mpo(reduced_mps)
        red = mpo_to_global(reduced_mpo).copy()
        traceout = tuple(range(startsite)) \
            + tuple(range(startsite + keep_width, nr_sites))
        red_from_op = _tools.partial_trace(op, traceout).copy()
        red.shape = (local_dim**keep_width,) * 2
        red_from_op.shape = (local_dim**keep_width,) * 2
        assert_array_almost_equal(red, red_from_op,
                                  err_msg="not equal at startsite {}".format(startsite))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_local_purification_mps_to_mpo(nr_sites, local_dim, bond_dim):
    assert (nr_sites % 2) == 0, 'this test can only be run for even numbers of sites'
    nr_sites = int(nr_sites / 2)
    mps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    mpo = mp.local_purification_mps_to_mpo(mps)
    # Local form is what we will use: One system site, one ancilla site, etc
    purification = mps.to_array()
    # Convert to a density matrix
    purification = np.outer(purification, purification.conj())
    purification.shape = (local_dim,) * (2 * 2 * nr_sites)
    # Trace out the ancilla sites
    traceout = tuple(range(1, 2 * nr_sites, 2))
    state = _tools.partial_trace(purification, traceout)
    # Here, we need global form
    state2 = mpo_to_global(mpo)
    assert_array_almost_equal(state, state2)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mps_as_mpo(nr_sites, local_dim, bond_dim):
    mps = factory.random_mpa(nr_sites, local_dim, bond_dim)
    # Instead of calling the two functions, we call mps_as_mpo(),
    # which does exactly that:
    #mps_as_puri = mp.mps_as_local_purification_mps(mps)
    #mpo = mp.local_purification_mps_to_mpo(mps_as_puri)
    mpo = mp.mps_as_mpo(mps)
    # This is also a test of mp.mps_as_local_purification_mps() in the
    # following sense: Local purifications are representations of
    # mixed states. Therefore, compare mps and mps_as_puri by
    # converting them to mixed states.
    state = mps.to_array()
    state = np.outer(state, state.conj())
    state.shape = (local_dim,) * (2 * nr_sites)
    state2 = mpo_to_global(mpo)
    assert_array_almost_equal(state, state2)


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


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd_errors(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)

    mpo_new = mpo.copy()
    mpo_new.compress(max_bdim=10 * bond_dim, method='svd', direction='right')
    assert_array_equal(mpo.bdims, mpo_new.bdims)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))

    mpo_new = mpo.copy()
    mpo_new.compress(max_bdim=10 * bond_dim, method='svd', direction='left')
    assert_array_equal(mpo.bdims, mpo_new.bdims)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd_hard_cutoff(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    zero = factory.zero_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    mpo_new = mpo + zero

    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    for bdims in zip(mpo.bdims, zero.bdims, mpo_new.bdims):
        assert_equal(bdims[0] + bdims[1], bdims[2])

    # Right-compression
    mpo_new = mpo + zero
    overlap = mpo_new.compress(max_bdim=bond_dim, method='svd', direction='right')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    assert_correct_normalzation(mpo_new, nr_sites - 1, nr_sites)
    # since no truncation error should occur
    assert_almost_equal(overlap, mp.norm(mpo)**2, decimal=5)

    # Left-compression
    mpo_new = mpo + zero
    overlap = mpo_new.compress(max_bdim=bond_dim, method='svd', direction='left')
    assert_array_equal(mpo_new.bdims, bond_dim)
    assert_array_almost_equal(mpo_to_global(mpo), mpo_to_global(mpo_new))
    assert_correct_normalzation(mpo_new, 0, 1)
    # since no truncation error should occur
    assert_almost_equal(overlap, mp.norm(mpo)**2, decimal=5)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_compression_svd_relerr(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    zero = factory.zero_mpa(nr_sites, (local_dim, local_dim), bond_dim)
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

    overlap = mpo_new.compress(max_bdim=bond_dim // 2, method='svd',
                               direction='right')
    assert_almost_equal(overlap, mp.inner(mpo, mpo_new), decimal=5)
    assert all(bdim_n < bdim_o for bdim_n, bdim_o in zip(mpo_new.bdims, mpo.bdims))

    mpo_new = mpo.copy()
    overlap = mpo_new.compress(max_bdim=bond_dim // 2, method='svd',
                               direction='left')
    assert_almost_equal(overlap, mp.inner(mpo, mpo_new), decimal=5)
    assert all(bdim_n < bdim_o for bdim_n, bdim_o in zip(mpo_new.bdims, mpo.bdims))
