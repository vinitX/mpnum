# encoding: utf-8


from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.testing import assert_almost_equal

import mpnum as mp
import mpnum.linalg
import mpnum.factory as factory

from mparray_test import MP_TEST_PARAMETERS


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mineig(nr_sites, local_dim, bond_dim, rgen):
    # Need at least two sites
    if nr_sites < 2:
        return
    # With startvec_bonddim = 2 * bonddim and this seed, mineig() gets
    # stuck in a local minimum. With startvec_bonddim = 3 * bonddim,
    # it does not.
    mpo = factory.random_mpo(nr_sites, local_dim, bond_dim, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.normalize()
    op = mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)
    eigvals, eigvec = np.linalg.eig(op)

    # Eigenvals should be real for a hermitian matrix
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig = eigvals[mineig_pos]
    mineig_eigvec = eigvec[:, mineig_pos]
    mineig_mp, mineig_eigvec_mp = mpnum.linalg.mineig(
        mpo, startvec_bonddim=3 * bond_dim, randstate=rgen)
    mineig_eigvec_mp = mineig_eigvec_mp.to_array().flatten()

    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec_mp)
    assert_almost_equal(mineig, mineig_mp)
    assert_almost_equal(1, abs(overlap))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mineig_minimize_sites(nr_sites, local_dim, bond_dim, rgen):
    # Need at least three sites for minimize_sites = 2
    if nr_sites < 3:
        return
    # With startvec_bonddim = 2 * bonddim and minimize_sites=1,
    # mineig() gets stuck in a local minimum. With minimize_sites=2,
    # it does not.
    mpo = factory.random_mpo(nr_sites, local_dim, bond_dim, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.normalize()
    op = mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)
    eigvals, eigvec = np.linalg.eig(op)

    # Eigenvals should be real for a hermitian matrix
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig, mineig_eigvec = eigvals[mineig_pos], eigvec[:, mineig_pos]
    mineig_mp, mineig_eigvec_mp = mpnum.linalg.mineig(
        mpo, startvec_bonddim=2 * bond_dim, randstate=rgen,
        minimize_sites=2)
    mineig_eigvec_mp = mineig_eigvec_mp.to_array().flatten()

    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec_mp)
    assert_almost_equal(mineig_mp, mineig)
    assert_almost_equal(abs(overlap), 1)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mineig_eigs_opts(nr_sites, local_dim, bond_dim, rgen):
    """Verify correct operation if eigs_opts() is specified

    This test mainly verifies correct operation if the user specifies
    eigs_opts() but does not include which='SR' herself. It also tests
    minimization on another example MPO (a rank-1 MPO in this case).

    """
    # Need at least two sites
    if nr_sites < 2:
        return

    mps = factory.random_mps(nr_sites, local_dim, bond_dim, rgen)
    mpo = mp.mps_to_mpo(mps)
    # mineig does not support startvec_bonddim = 1
    bond_dim = 2 if bond_dim == 1 else bond_dim
    eigval, eigvec = mpnum.linalg.mineig(
        mpo, startvec_bonddim=bond_dim, randstate=rgen, max_num_sweeps=10,
        eigs_opts=dict(tol=1e-5), minimize_sites=1)
    # Check correct eigenvalue
    assert_almost_equal(eigval, 0)
    # Check for orthogonal eigenvectors and `eigvec` being in the
    # kernel of `mpo`
    assert_almost_equal(abs(mp.inner(eigvec, mps)), 0)
