# encoding: utf-8
# TODO Needs dtype dependent tests


from __future__ import absolute_import, division, print_function

import functools as ft
import numpy as np
import pytest as pt
from _pytest.mark import matchmark
from numpy.testing import assert_almost_equal
from scipy.sparse.linalg import eigsh

import mpnum as mp
import mpnum.linalg
import mpnum.factory as factory
from mpnum.utils import physics


def _pytest_want_long(request):
    # Is there a better way to find out whether items marked with
    # `long` should be run or not?
    class dummy:
        keywords = {'long': pt.mark.long}
    return matchmark(dummy, request.config.option.markexpr)


@pt.mark.parametrize('which', ['LM', 'LA', 'SA'])
@pt.mark.parametrize('var_sites', [1, 2])
@pt.mark.parametrize('nr_sites, local_dim, rank', pt.MP_TEST_PARAMETERS)
def test_eig(nr_sites, local_dim, rank, which, var_sites, rgen, request):
    if nr_sites <= var_sites:
        pt.skip("Nothing to test")
        return  # No local optimization can be defined
    if not (_pytest_want_long(request) or
            (nr_sites, local_dim, rank, var_sites, which) in {
                (3, 2, 4, 1, 'SA'), (4, 3, 5, 1, 'LM'), (5, 2, 1, 2, 'LA'),
                (6, 2, 4, 2, 'SA'),
            }):
        pt.skip("Should only be run in long tests")
    # With startvec_rank = 2 * rank and this seed, eig() gets
    # stuck in a local minimum. With startvec_rank = 3 * rank,
    # it does not.
    mpo = factory.random_mpo(nr_sites, local_dim, rank, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.canonicalize()
    op = mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)
    v0 = factory._zrandn([local_dim**nr_sites], rgen)
    eigval, eigvec = eigsh(op, k=1, which=which, v0=v0)
    eigval, eigvec = eigval[0], eigvec[:, 0]

    eig_rank = (4 - var_sites) * rank
    eigval_mp, eigvec_mp = mp.eig(
        mpo, num_sweeps=5, var_sites=1, startvec_rank=eig_rank, randstate=rgen,
        eigs=ft.partial(eigsh, k=1, which=which, tol=1e-6, maxiter=250))
    eigvec_mp = eigvec_mp.to_array().flatten()

    overlap = np.inner(eigvec.conj(), eigvec_mp)
    assert_almost_equal(eigval, eigval_mp, decimal=14)
    assert_almost_equal(1, abs(overlap), decimal=14)


@pt.mark.parametrize('nr_sites, local_dim, rank', pt.MP_TEST_PARAMETERS)
def test_eig_sum(nr_sites, local_dim, rank, rgen):
    # Need at least three sites for var_sites = 2
    if nr_sites < 3:
        pt.skip("Nothing to test")
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
    eigvals, eigvec = np.linalg.eigh(op)

    # Eigenvals should be real for a hermitian matrix
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig, mineig_eigvec = eigvals[mineig_pos], eigvec[:, mineig_pos]
    mineig_mp, mineig_eigvec_mp = mp.eig_sum(
        mpas, num_sweeps=5, startvec_rank=5 * rank, randstate=rgen,
        eigs=ft.partial(eigsh, k=1, which='SA', tol=1e-6),
        var_sites=2)
    mineig_eigvec_mp = mineig_eigvec_mp.to_array().flatten()

    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec_mp)
    assert_almost_equal(mineig_mp, mineig)
    assert_almost_equal(abs(overlap), 1)


@pt.mark.parametrize('nr_sites, gamma, rank, tol', [
    (10, 0.61, 6, 1e-3),
    pt.mark.verylong((50, 0.95, 16, 1e-12)),
    pt.mark.long((130, 0.9, 2, 1e-3)),
])
def test_eig_cXY_groundstate(nr_sites, gamma, rank, tol, rgen, local_dim=2):
    # Verify that linalg.eig() finds the correct ground state energy
    # of the cyclic XY model
    E0 = physics.cXY_E0(nr_sites, gamma)
    mpo = physics.mpo_cH(physics.cXY_local_terms(nr_sites, gamma))
    eigs = ft.partial(eigsh, k=1, which='SA', tol=1e-6)
    E0_mp, mineig_eigvec_mp = mpnum.linalg.eig(
        mpo, startvec_rank=rank, randstate=rgen, var_sites=2, num_sweeps=3,
        eigs=eigs)
    print(abs(E0_mp - E0))
    assert abs(E0_mp - E0) <= tol


BENCHMARK_MINEIG_PARAMS = [(20, 2, 12, 12)]


@pt.mark.benchmark(group='eig_sum', min_rounds=2)
@pt.mark.parametrize(
    'nr_sites, local_dim, rank, ev_rank', BENCHMARK_MINEIG_PARAMS)
def test_eig_benchmark(
        nr_sites, local_dim, rank, ev_rank, rgen, benchmark):
    mpo = factory.random_mpo(nr_sites, local_dim, rank, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.canonicalize()
    mps = factory.random_mpa(nr_sites, local_dim, rank, randstate=rgen,
                             dtype=np.complex_, normalized=True)
    mpo = mpo + mp.mps_to_mpo(mps)

    benchmark(
        mp.eig,
        mpo, startvec_rank=ev_rank, randstate=rgen,
        var_sites=1, num_sweeps=1,
    )


@pt.mark.benchmark(group='eig_sum', min_rounds=2)
@pt.mark.parametrize(
    'nr_sites, local_dim, rank, ev_rank', BENCHMARK_MINEIG_PARAMS)
def test_eig_sum_benchmark(
        nr_sites, local_dim, rank, ev_rank, rgen, benchmark):
    mpo = factory.random_mpo(nr_sites, local_dim, rank, randstate=rgen,
                             hermitian=True, normalized=True)
    mpo.canonicalize()
    mps = factory.random_mpa(nr_sites, local_dim, rank, randstate=rgen,
                             dtype=np.complex_, normalized=True)

    benchmark(
        mp.eig_sum,
        [mpo, mps], startvec_rank=ev_rank, randstate=rgen,
        var_sites=1, num_sweeps=1,
    )
