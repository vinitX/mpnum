# encoding: utf-8


from __future__ import division, print_function

import numpy as np
import pytest as pt

import mpnum.factory as factory
from numpy.testing import assert_array_almost_equal

from mparray_test import MP_TEST_PARAMETERS, MP_TEST_DTYPES
from mpnum._testing import assert_correct_normalization


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', [(2, 3, 3), (3, 2, 4),
                                                       (6, 2, 4), (4, 3, 5),
                                                       (5, 2, 1)])
def test_mpdo_positivity(nr_sites, local_dim, bond_dim, rgen):
    rho = factory.random_mpdo(nr_sites, local_dim, bond_dim, rgen)
    rho_dense = rho.to_array_global().reshape((local_dim**nr_sites,) * 2)

    np.testing.assert_array_almost_equal(rho_dense, rho_dense.conj().T)
    lambda_min = min(np.real(np.linalg.eigvals(rho_dense)))
    assert lambda_min > -1e-14, "{} < -1e-14".format(lambda_min)


#  @pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_diagonal_mpa(nr_sites, local_dim, _, rgen, dtype=np.float_):
    randfunc = factory._randfuncs[dtype]
    entries = randfunc((local_dim,), randstate=rgen)

    mpa_mp = factory.diagonal_mpa(entries, nr_sites)
    if nr_sites > 1:
        mpa_np = np.zeros((local_dim,) * nr_sites)
        np.fill_diagonal(mpa_np, entries)
    else:
        mpa_np = entries

    assert len(mpa_mp) == nr_sites
    assert_array_almost_equal(mpa_mp.to_array(), mpa_np)
    assert mpa_mp.dtype is dtype
    assert_correct_normalization(mpa_mp, nr_sites - 1, nr_sites)

    if nr_sites > 1:
        assert mpa_mp.bdim == local_dim
