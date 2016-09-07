# encoding: utf-8
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_equal)

import mpnum.factory as factory
import mpnum.mparray as mp
import mpnum.special as mpsp
from mpnum import _tools
from mpnum._testing import (assert_correct_normalization,
                            assert_mpa_almost_equal, assert_mpa_identical,
                            mpo_to_global)
from mpnum._tools import global_to_local
from mparray_test import MP_TEST_DTYPES, MP_TEST_PARAMETERS

MP_INNER_PARAMETERS = [(10, 10, 5), (20, 2, 10)]
MP_SUMUP_PARAMETERS = [(6, 2, 1000, 100, 200), (10, 2, 500, 5, 20)]


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_inner_prod_mps(nr_sites, local_dim, bond_dim, dtype, rgen):
    mpa1 = factory.random_mpa(nr_sites, local_dim, 1, dtype=dtype,
                              randstate=rgen, normalized=True)
    mpa2 = factory.random_mpa(nr_sites, local_dim, bond_dim, dtype=dtype,
                              randstate=rgen, normalized=True)

    res_slow = mp.inner(mpa1, mpa2)
    res_fast = mpsp.inner_prod_mps(mpa1, mpa2)
    assert_almost_equal(res_slow, res_fast)

    try:
        mpsp.inner_prod_mps(mpa2, mpa1)
    except AssertionError:
        pass
    else:
        if bond_dim > 1:
            raise AssertionError("inner_prod_mps should only accept bdim=1 in first argument")

    mpa1 = factory.random_mpo(nr_sites, local_dim, 1)
    try:
        mpsp.inner_prod_mps(mpa1, mpa1)
    except AssertionError:
        pass
    else:
        raise AssertionError("inner_prod_mps should only accept plegs=1")


@pt.mark.benchmark(group="inner")
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_INNER_PARAMETERS)
def test_inner_fast(nr_sites, local_dim, bond_dim, benchmark, rgen):
    mpa1 = factory.random_mpa(nr_sites, local_dim, 1, dtype=np.float_,
                              randstate=rgen, normalized=True)
    mpa2 = factory.random_mpa(nr_sites, local_dim, bond_dim, dtype=np.float_,
                              randstate=rgen, normalized=True)

    benchmark(mpsp.inner_prod_mps, mpa1, mpa2)


@pt.mark.benchmark(group="inner")
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_INNER_PARAMETERS)
def test_inner_slow(nr_sites, local_dim, bond_dim, benchmark, rgen):
    mpa1 = factory.random_mpa(nr_sites, local_dim, 1, dtype=np.float_,
                              randstate=rgen)
    mpa2 = factory.random_mpa(nr_sites, local_dim, bond_dim, dtype=np.float_,
                              randstate=rgen)

    benchmark(mp.inner, mpa1, mpa2)
