# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.testing import assert_almost_equal

import mpnum.linalg
import mpnum.factory as factory

from mparray_test import mpo_to_global, MP_TEST_PARAMETERS


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mineig(nr_sites, local_dim, bond_dim):
    # With startvec_bonddim = 2 * bonddim and this seed, mineig() gets
    # stuck in a local minimum. With startvec_bonddim = 3 * bonddim,
    # it does not.
    randstate = np.random.RandomState(seed=46)
    mpo = factory.random_mpo(nr_sites, local_dim, bond_dim, randstate=randstate,
                             hermitian=True, normalized=True)
    mpo.normalize()
    op = mpo_to_global(mpo).reshape((local_dim**nr_sites,) * 2)
    eigvals, eigvec = np.linalg.eig(op)

    # Eigenvals should be real for a hermitian matrix
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig = eigvals[mineig_pos]
    mineig_eigvec = eigvec[:, mineig_pos]
    mineig_mp, mineig_eigvec_mp = mpnum.linalg.mineig(
        mpo, startvec_bonddim=3 * bond_dim, randstate=randstate)
    mineig_eigvec_mp = mineig_eigvec_mp.to_array().flatten()

    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec_mp)
    assert_almost_equal(mineig, mineig_mp)
    assert_almost_equal(1, abs(overlap))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mineig_minimize_sites(nr_sites, local_dim, bond_dim):
    # With startvec_bonddim = 2 * bonddim and minimize_sites=1,
    # mineig() gets stuck in a local minimum. With minimize_sites=2,
    # it does not.
    randstate = np.random.RandomState(seed=46)
    mpo = factory.random_mpo(nr_sites, local_dim, bond_dim, randstate=randstate,
                             hermitian=True, normalized=True)
    mpo.normalize()
    op = mpo_to_global(mpo).reshape((local_dim**nr_sites,) * 2)
    eigvals, eigvec = np.linalg.eig(op)

    # Eigenvals should be real for a hermitian matrix
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig, mineig_eigvec = eigvals[mineig_pos], eigvec[:, mineig_pos]
    mineig_mp, mineig_eigvec_mp = mpnum.linalg.mineig(
        mpo, startvec_bonddim=2 * bond_dim, randstate=randstate,
        minimize_sites=2)
    mineig_eigvec_mp = mineig_eigvec_mp.to_array().flatten()

    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec_mp)
    assert_almost_equal(mineig, mineig_mp)
    assert_almost_equal(1, abs(overlap))
