#!/usr/bin/env python
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
import mpnum.mparray as mp

from mparray_test import mpo_to_global, svd_compression, MP_TEST_PARAMETERS


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mineig(nr_sites, local_dim, bond_dim):
    # With startvec_bonddim = 2 * bonddim and this seed, mineig() gets
    # stuck in a local minimum. With startvec_bonddim = 3 * bonddim,
    # it does not.
    randstate = np.random.RandomState(seed=46)
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim, randstate)
    # make mpa Herimitan in place, without increasing bond dimension:
    for lten in mpo:
        lten += lten.swapaxes(1, 2).conj()
    mpo.normalize()
    mpo /= mp.norm(mpo)
    op = mpo_to_global(mpo).reshape((local_dim**nr_sites,) * 2)
    eigvals, eigvec = np.linalg.eig(op)
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig = eigvals[mineig_pos]
    mineig_eigvec = eigvec[:, mineig_pos]
    mineig2, mineig_eigvec2 = mpnum.linalg.mineig(
        mpo, startvec_bonddim=3 * bond_dim, startvec_randstate=randstate)
    mineig_eigvec2 = mineig_eigvec2.to_array().flatten()
    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec2)
    assert_almost_equal(mineig, mineig2)
    assert_almost_equal(1, abs(overlap))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mineig_minimize_sites(nr_sites, local_dim, bond_dim):
    # With startvec_bonddim = 2 * bonddim and minimize_sites=1,
    # mineig() gets stuck in a local minimum. With minimize_sites=2,
    # it does not.
    randstate = np.random.RandomState(seed=46)
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim, randstate)
    # make mpa Herimitan in place, without increasing bond dimension:
    for lten in mpo:
        lten += lten.swapaxes(1, 2).conj()
    mpo.normalize()
    mpo /= mp.norm(mpo)
    op = mpo_to_global(mpo).reshape((local_dim**nr_sites,) * 2)
    eigvals, eigvec = np.linalg.eig(op)
    assert (np.abs(eigvals.imag) < 1e-10).all(), str(eigvals.imag)
    mineig_pos = eigvals.argmin()
    mineig = eigvals[mineig_pos]
    mineig_eigvec = eigvec[:, mineig_pos]
    mineig2, mineig_eigvec2 = mpnum.linalg.mineig(
        mpo, startvec_bonddim=2 * bond_dim, startvec_randstate=randstate,
        minimize_sites=2)
    mineig_eigvec2 = mineig_eigvec2.to_array().flatten()
    overlap = np.inner(mineig_eigvec.conj(), mineig_eigvec2)
    assert_almost_equal(mineig, mineig2)
    assert_almost_equal(1, abs(overlap))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_variational_compression(nr_sites, local_dim, bond_dim):
    overlap_rel_tol = 1e-6
    plegs = 2
    randstate = np.random.RandomState(seed=42)
    mpa = factory.random_mpa(nr_sites, (local_dim,) * plegs, bond_dim, randstate)
    mpa /= mp.norm(mpa)
    array = mpa.to_array()
    target_bonddim = max(2 * bond_dim // 3, 1)

    right_svd_res = svd_compression(mpa, 'right', target_bonddim)
    left_svd_res = svd_compression(mpa, 'left', target_bonddim)
    right_svd_overlap = np.abs(np.dot(array.conj().flatten(), right_svd_res.flatten()))
    left_svd_overlap = np.abs(np.dot(array.conj().flatten(), left_svd_res.flatten()))

    # max_num_sweeps = 3 and 4 is sometimes not good enough.
    mpa_compr = mpnum.linalg.variational_compression(
        mpa, max_num_sweeps=5,
        startvec_bonddim=target_bonddim, startvec_randstate=randstate)
    mpa_compr_overlap = np.abs(np.dot(array.conj().flatten(),
                                      mpa_compr.to_array().flatten()))

    # The basic intuition is that variational compression, given
    # enough sweeps, should be at least as good as left and right SVD
    # compression because the SVD compression scheme has a strong
    # interdependence between truncations at the individual sites,
    # while variational compression does not have that. Therefore, we
    # check exactly that.

    assert mpa_compr_overlap >= right_svd_overlap * (1 - overlap_rel_tol)
    assert mpa_compr_overlap >= left_svd_overlap * (1 - overlap_rel_tol)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_variational_compression_twosite(nr_sites, local_dim, bond_dim):
    overlap_rel_tol = 1e-6
    plegs = 2
    randstate = np.random.RandomState(seed=42)
    mpa = factory.random_mpa(nr_sites, (local_dim,) * plegs, bond_dim, randstate)
    mpa /= mp.norm(mpa)
    array = mpa.to_array()
    target_bonddim = max(2 * bond_dim // 3, 1)

    right_svd_res = svd_compression(mpa, 'right', target_bonddim)
    left_svd_res = svd_compression(mpa, 'left', target_bonddim)
    right_svd_overlap = np.abs(np.dot(array.conj().flatten(), right_svd_res.flatten()))
    left_svd_overlap = np.abs(np.dot(array.conj().flatten(), left_svd_res.flatten()))

    # With minimize_sites = 1, max_num_sweeps = 3 and 4 is sometimes
    # not good enough. With minimiza_sites = 2, max_num_sweeps = 2 is
    # fine.
    mpa_compr = mpnum.linalg.variational_compression(
        mpa, startvec_bonddim=target_bonddim, startvec_randstate=randstate,
        max_num_sweeps=3, minimize_sites=2)
    mpa_compr_overlap = np.abs(np.dot(array.conj().flatten(),
                                      mpa_compr.to_array().flatten()))

    # The basic intuition is that variational compression, given
    # enough sweeps, should be at least as good as left and right SVD
    # compression because the SVD compression scheme has a strong
    # interdependence between truncations at the individual sites,
    # while variational compression does not have that. Therefore, we
    # check exactly that.

    assert mpa_compr_overlap >= right_svd_overlap * (1 - overlap_rel_tol)
    assert mpa_compr_overlap >= left_svd_overlap * (1 - overlap_rel_tol)
