#!/usr/bin/env python
# encoding: utf-8
"""Module to create random test instances of matrix product arrays"""

from __future__ import division, print_function
import numpy as np
from six.moves import range

import mpnum.mparray as mp


def _zrandn(shape):
    """Shortcut for np.random.randn(*shape) + 1.j * np.random.randn(*shape)
    """
    return np.random.randn(*shape) + 1.j * np.random.randn(*shape)


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
    psi = _zrandn(shape)
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
    return _zrandn(shape)


def random_state(sites, ldim):
    """Returns a random positive semidefinite operator of shape (ldim, ldim) *
    sites normalized to Tr rho = 1, i.e. a mixed state with local dimension
    `ldim` living on `sites` sites. Note that the returned state is positive
    semidefinite only when interpreted in global form (see
    :func:`_tools.global_to_local`)

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
    mat = _zrandn(shape)
    rho = np.conj(mat.T).dot(mat)
    rho /= np.trace(rho)
    return rho.reshape((ldim,) * 2 * sites)


def _generate(sites, ldim, bdim, func):
    """Returns a matrix product operator with identical number and dimensions
    of the physical legs. The local tensors are generated using `func`

    :param sites: Number of sites
    :param ldim: Tuple of int-like of local dimensions
    :param bdim: Bond dimension
    :param func: Generator function for local tensors, should accept shape as
        tuple in first argument and should return numpy.ndarray of given shape
    :returns: randomly choosen matrix product array

    """
    assert sites > 1, "Cannot generate MPA with sites {} < 2".format(sites)
    # if ldim is passed as scalar, make it 1-element tuple
    ldim = ldim if hasattr(ldim, '__iter__') else (ldim, )
    ltens_l = func((1, ) + ldim + (bdim, ))
    ltenss = [func((bdim, ) + ldim + (bdim, ))
              for _ in range(sites - 2)]
    ltens_r = func((bdim, ) + ldim + (bdim, ))
    return mp.MPArray([ltens_l] + ltenss + [ltens_r])


def random_mpa(sites, ldim, bdim):
    """Returns a MPA with randomly choosen local tensors

    :param sites: Number of sites
    :param ldim: Tuple of int-like of local dimensions
    :param bdim: Bond dimension
    :returns: randomly choosen matrix product array

    """
    return _generate(sites, ldim, bdim, _zrandn)


def zero_mpa(sites, ldim, bdim):
    """Returns a MPA with localtensors beeing zero (but of given shape)

    :param sites: Number of sites
    :param ldim: Tuple of int-like of local dimensions
    :param bdim: Bond dimension
    :returns: Representation of the zero-array as MPA

    """
    return _generate(sites, ldim, bdim, np.zeros)
