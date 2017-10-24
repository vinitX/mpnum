# encoding: utf-8

from __future__ import division, print_function

import pytest as pt
import sys
from scipy.sparse.linalg import eigsh

from mpnum.utils import physics


@pt.mark.parametrize('nr_sites', [2, 6, pt.mark.long(10)])
@pt.mark.parametrize('gamma', [-0.5, pt.mark.long(0.1), pt.mark.long(1.0)])
def test_cXY_E0(nr_sites, gamma, rgen, ldim=2):
    if sys.version_info[:2] == (3, 3) and gamma == -0.5:
        # Skip this test on Python 3.3 because it fails on Travis (but
        # only for Python 3.3). eigsh() fails with:
        # scipy.sparse.linalg.eigen.arpack.arpack.ArpackNoConvergence:
        # ARPACK error -1: No convergence (xxx iterations, 0/1
        # eigenvectors converged) [ARPACK error -14: ZNAUPD did not
        # find any eigenvalues to sufficient accuracy.]
        pt.skip("Test fails on Travis for unknown reason")
        return

    # Verify that the analytical solution of the ground state energy
    # matches the numerical value from eigsh()
    E0 = physics.cXY_E0(nr_sites, gamma)
    H = physics.sparse_cH(physics.cXY_local_terms(nr_sites, gamma))
    # Fix start vector for eigsh()
    v0 = rgen.randn(ldim**nr_sites) + 1j * rgen.randn(ldim**nr_sites)
    ev = eigsh(H, k=1, which='SR', v0=v0, return_eigenvectors=False).min()
    assert abs(E0 - ev) <= 1e-13
