# encoding: utf-8
# TODO Needs dtype dependent tests


from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.testing import assert_almost_equal

import mpnum as mp
import mpnum.linalg
import mpnum.factory as factory
from mpnum.utils import physics


@pt.mark.parametrize('nr_sites, local_dim, rank', pt.MP_TEST_PARAMETERS)
def test_mineig(nr_sites, local_dim, rank, rgen):
    # Need at least two sites
    if nr_sites < 2:
        return
    # With startvec_rank = 2 * rank and this seed, eig() gets
    # stuck in a local minimum. With startvec_rank = 3 * rank,
    # it does not.
    mpo = factory.random_mpo(nr_sites, local_dim, rank, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.canonicalize()
    op = mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)
    eigvals, eigvec = np.linalg.eig(op)

    # Eigenvals should be real for a hermitian matrix
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig = eigvals[mineig_pos]
    mineig_eigvec = eigvec[:, mineig_pos]
    mineig_mp, mineig_eigvec_mp = mpnum.linalg.eig(
        mpo, startvec_rank=3 * rank, randstate=rgen)
    mineig_eigvec_mp = mineig_eigvec_mp.to_array().flatten()

    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec_mp)
    assert_almost_equal(mineig, mineig_mp)
    assert_almost_equal(1, abs(overlap))


@pt.mark.parametrize('nr_sites, local_dim, rank', pt.MP_TEST_PARAMETERS)
def test_mineig_minimize_sites(nr_sites, local_dim, rank, rgen):
    # Need at least three sites for minimize_sites = 2
    if nr_sites < 3:
        return
    # With startvec_rank = 2 * rank and minimize_sites=1,
    # eig() gets stuck in a local minimum. With minimize_sites=2,
    # it does not.
    mpo = factory.random_mpo(nr_sites, local_dim, rank, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.canonicalize()
    op = mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)
    eigvals, eigvec = np.linalg.eig(op)
    eigs_opts = {'maxiter': 256}

    # Eigenvals should be real for a hermitian matrix
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig, mineig_eigvec = eigvals[mineig_pos], eigvec[:, mineig_pos]
    mineig_mp, mineig_eigvec_mp = mpnum.linalg.eig(
        mpo, startvec_rank=2 * rank, randstate=rgen,
        minimize_sites=2, eigs_opts=eigs_opts)
    mineig_eigvec_mp = mineig_eigvec_mp.to_array().flatten()

    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec_mp)
    assert_almost_equal(mineig_mp, mineig)
    assert_almost_equal(abs(overlap), 1)


@pt.mark.parametrize('nr_sites, local_dim, rank', pt.MP_TEST_PARAMETERS)
def test_mineig_sum_minimize_sites(nr_sites, local_dim, rank, rgen):
    # Need at least three sites for minimize_sites = 2
    if nr_sites < 3:
        return
    rank = max(1, rank // 2)
    mpo = factory.random_mpo(nr_sites, local_dim, rank, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.canonicalize()
    mps = factory.random_mpa(nr_sites, local_dim, rank, randstate=rgen,
                             dtype=np.complex_, normalized=True)
    mpas = [mpo, mps]

    vec = mps.to_array().ravel()
    op = mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)
    op += np.outer(vec, vec.conj())
    eigvals, eigvec = np.linalg.eig(op)

    # Eigenvals should be real for a hermitian matrix
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig, mineig_eigvec = eigvals[mineig_pos], eigvec[:, mineig_pos]
    mineig_mp, mineig_eigvec_mp = mpnum.linalg.eig_sum(
        mpas, startvec_rank=5 * rank, randstate=rgen,
        minimize_sites=2)
    mineig_eigvec_mp = mineig_eigvec_mp.to_array().flatten()

    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec_mp)
    assert_almost_equal(mineig_mp, mineig)
    assert_almost_equal(abs(overlap), 1)


@pt.mark.parametrize('nr_sites, gamma, rank, tol', [
    (10, 0.61, 6, 1e-3),
    pt.mark.long((50, 0.95, 16, 1e-12)),
    pt.mark.long((130, 0.9, 2, 1e-3)),
])
def test_mineig_cXY(nr_sites, gamma, rank, tol, rgen, local_dim=2):
    # Verify that linalg.eig() finds the correct ground state energy
    # of the cyclic XY model
    E0 = physics.cXY_E0(nr_sites, gamma)
    mpo = physics.mpo_cH(physics.cXY_local_terms(nr_sites, gamma))
    E0_mp, mineig_eigvec_mp = mpnum.linalg.eig(
        mpo, startvec_rank=rank, randstate=rgen, minimize_sites=2, max_num_sweeps=3)
    print(abs(E0_mp - E0))
    assert abs(E0_mp - E0) <= tol


BENCHMARK_MINEIG_PARAMS = [(20, 2, 12, 12)]


@pt.mark.benchmark(group='eig_sum', min_rounds=2)
@pt.mark.parametrize(
    'nr_sites, local_dim, rank, ev_rank', BENCHMARK_MINEIG_PARAMS)
def test_mineig_benchmark(
        nr_sites, local_dim, rank, ev_rank, rgen, benchmark):
    mpo = factory.random_mpo(nr_sites, local_dim, rank, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.canonicalize()
    mps = factory.random_mpa(nr_sites, local_dim, rank, randstate=rgen,
                             dtype=np.complex_, normalized=True)
    mpo = mpo + mp.mps_to_mpo(mps)

    benchmark(
        mpnum.linalg.eig,
        mpo, startvec_rank=ev_rank, randstate=rgen,
        minimize_sites=1, max_num_sweeps=1,
    )


@pt.mark.benchmark(group='eig_sum', min_rounds=2)
@pt.mark.parametrize(
    'nr_sites, local_dim, rank, ev_rank', BENCHMARK_MINEIG_PARAMS)
def test_mineig_sum_benchmark(
        nr_sites, local_dim, rank, ev_rank, rgen, benchmark):
    mpo = factory.random_mpo(nr_sites, local_dim, rank, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.canonicalize()
    mps = factory.random_mpa(nr_sites, local_dim, rank, randstate=rgen,
                             dtype=np.complex_, normalized=True)

    benchmark(
        mpnum.linalg.eig_sum,
        [mpo, mps], startvec_rank=ev_rank, randstate=rgen,
        minimize_sites=1, max_num_sweeps=1,
    )


@pt.mark.parametrize('nr_sites, local_dim, rank', pt.MP_TEST_PARAMETERS)
def test_mineig_eigs_opts(nr_sites, local_dim, rank, rgen):
    """Verify correct operation if eigs_opts() is specified

    This test mainly verifies correct operation if the user specifies
    eigs_opts() but does not include which='SR' herself. It also tests
    minimization on another example MPO (a rank-1 MPO in this case).

    """
    # Need at least two sites
    if nr_sites < 2:
        return

    mps = factory.random_mps(nr_sites, local_dim, rank, rgen)
    mpo = mp.mps_to_mpo(mps)
    # eig does not support startvec_rank = 1
    rank = 2 if rank == 1 else rank
    eigval, eigvec = mpnum.linalg.eig(
        mpo, startvec_rank=rank, randstate=rgen, max_num_sweeps=10,
        eigs_opts=dict(tol=1e-5), minimize_sites=1)
    # Check correct eigenvalue
    assert_almost_equal(eigval, 0)
    # Check for orthogonal eigenvectors and `eigvec` being in the
    # kernel of `mpo`
    assert_almost_equal(abs(mp.inner(eigvec, mps)), 0)
