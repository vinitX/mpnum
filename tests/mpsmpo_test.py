#!/usr/bin/env python
# encoding: utf-8
#
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt

from numpy.testing import assert_array_almost_equal

import mpnum.factory as factory
import mpnum.mpsmpo as mm
import mpnum._tools as _tools
from mparray_test import MP_TEST_PARAMETERS, mpo_to_global


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, keep_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_mpo(nr_sites, local_dim, bond_dim, keep_width):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mpo)

    startsites = range(nr_sites - keep_width + 1)
    for site, reduced_mpo in mm.reductions_mpo(mpo, startsites, keep_width):
        traceout = tuple(range(site)) \
            + tuple(range(site + keep_width, nr_sites))
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(mpo_to_global(reduced_mpo), red_from_op,
                                  err_msg="not equal at site {}".format(site))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, keep_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_pmps(nr_sites, local_dim, bond_dim, keep_width):
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo_to_global(mm.pmps_to_mpo(pmps))

    startsites = range(nr_sites - keep_width + 1)
    for site, reduced_mps in mm.reductions_pmps(pmps, startsites, keep_width):
        reduced_mpo = mm.pmps_to_mpo(reduced_mps)
        red = mpo_to_global(reduced_mpo)
        traceout = tuple(range(site)) + tuple(range(site + keep_width, nr_sites))
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(red, red_from_op,
                                  err_msg="not equal at site {}".format(site))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_pmps_to_mpo(nr_sites, local_dim, bond_dim):
    if (nr_sites % 2) != 0:
        return
    nr_sites = nr_sites // 2
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    rho_mp = mpo_to_global(mm.pmps_to_mpo(pmps))

    # Local form is what we will use: One system site, one ancilla site, etc
    purification = pmps.to_array()
    # Convert to a density matrix
    purification = np.outer(purification, purification.conj())
    purification = purification.reshape((local_dim,) * (2 * 2 * nr_sites))
    # Trace out the ancilla sites
    traceout = tuple(range(1, 2 * nr_sites, 2))
    rho_np = _tools.partial_trace(purification, traceout)

    # Here, we need global form
    assert_array_almost_equal(rho_mp, rho_np)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mps_as_mpo(nr_sites, local_dim, bond_dim):
    mps = factory.random_mpa(nr_sites, local_dim, bond_dim)
    # Instead of calling the two functions, we call mps_as_mpo(),
    # which does exactly that:
    #   mps_as_puri = mp.mps_as_local_purification_mps(mps)
    #   mpo = mp.pmps_to_mpo(mps_as_puri)
    mpo = mm.mps_as_mpo(mps)
    # This is also a test of mp.mps_as_local_purification_mps() in the
    # following sense: Local purifications are representations of
    # mixed states. Therefore, compare mps and mps_as_puri by
    # converting them to mixed states.
    state = mps.to_array()
    state = np.outer(state, state.conj())
    state.shape = (local_dim,) * (2 * nr_sites)
    state2 = mpo_to_global(mpo)
    assert_array_almost_equal(state, state2)
