#!/usr/bin/env python
# encoding: utf-8
"""Module to create random test instances of matrix product arrays"""

from __future__ import division, print_function
import numpy as np
import mparray as mp


def random_vec(sites, ldim):
    """Returns a random complex vector (normalized to ||x||_2 = 1) of shape
    (ldim,) * sites, i.e. a pure state with local dimension `ldim` living on
    `sites` sites.

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :returns: numpy.ndarray of shape (ldim,) * sites

    >>> psi = random_vec(5, 2); psi.shape
    (2, 2, 2, 2, 2)
    >>> np.abs(np.vdot(psi, psi) - 1) < 1e-6
    True
    """
    shape = (ldim, ) * sites
    psi = np.random.randn(*shape) + 1.j * np.random.randn(*shape)
    psi /= np.sqrt(np.vdot(psi, psi))
    return psi


def random_op(sites, ldim):
    """Returns a random operator  of shape (ldim,ldim) * sites with local
    dimension `ldim` living on `sites` sites.

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :returns: numpy.ndarray of shape (ldim,ldim) * sites

    >>> A = random_op(3, 2); A.shape
    (2, 2, 2, 2, 2, 2)
    """
    shape = (ldim, ldim) * sites
    return np.random.randn(*shape) + 1.j * np.random.randn(*shape)


def random_state(sites, ldim):
    """Returns a random positive semidefinite operator of shape (ldim, ldim) *
    sites normalized to Tr rho = 1, i.e. a mixed state with local dimension
    `ldim` living on `sites` sites. Note that the returned state is positive
    semidefinite only when interpreted in global form (see
    :func:`_qmtools.global_to_local`)

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :returns: numpy.ndarray of shape (ldim, ldim) * sites

    >>> from numpy.linalg import eigvalsh
    >>> rho = random_state(3, 2).reshape((2**3, 2**3))
    >>> all(eigvalsh(rho) >= 0)
    True
    >>> np.abs(np.trace(rho) - 1) < 1e-6
    True
    """
    shape = (ldim**sites, ldim**sites)
    mat = np.random.randn(*shape) + 1.j * np.random.randn(*shape)
    rho = np.conj(mat.T).dot(mat)
    rho /= np.trace(rho)
    return rho.reshape((ldim,) * 2 * sites)
