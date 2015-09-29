# encoding: utf-8
#
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt

from numpy.testing import assert_array_almost_equal

import mpnum.factory as factory
import mpnum.mpsmpo as mm
import mpnum._tools as _tools
from mparray_test import MP_TEST_PARAMETERS


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, max_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_mpo(nr_sites, local_dim, bond_dim, max_width):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global()

    param = tuple(
        (start, start + width)
        for width in range(1, max_width)
        for start in range(nr_sites - width + 1)
    )
    start = tuple(start for start, _ in param)
    stop = tuple(stop for _, stop in param)
    red = mm.reductions_mpo(mpo, startsites=start, stopsites=stop)
    for start, stop, reduced_mpo in zip(start, stop, red):
        traceout = tuple(range(start)) + tuple(range(stop, nr_sites))
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(
            reduced_mpo.to_array_global(), red_from_op,
            err_msg="not equal at {}:{}".format(start, stop))

    # check default argument for startsite
    assert len(list(mm.reductions_mpo(mpo, max_width))) == nr_sites - max_width + 1


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, keep_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_pmps(nr_sites, local_dim, bond_dim, keep_width):
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mm.pmps_to_mpo(pmps).to_array_global()

    startsites = range(nr_sites - keep_width + 1)
    for site, reduced_mps in zip(startsites,
                                 mm.reductions_pmps(pmps, keep_width, startsites)):
        reduced_mpo = mm.pmps_to_mpo(reduced_mps)
        red = reduced_mpo.to_array_global()
        traceout = tuple(range(site)) + tuple(range(site + keep_width, nr_sites))
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(red, red_from_op,
                                  err_msg="not equal at site {}".format(site))

    # check default argument for startsite
    assert len(list(mm.reductions_pmps(pmps, keep_width))) == nr_sites - keep_width + 1


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_pmps_to_mpo(nr_sites, local_dim, bond_dim):
    if (nr_sites % 2) != 0:
        return
    nr_sites = nr_sites // 2
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    rho_mp = mm.pmps_to_mpo(pmps).to_array_global()

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
def test_mps_to_mpo(nr_sites, local_dim, bond_dim):
    mps = factory.random_mpa(nr_sites, local_dim, bond_dim)
    # Instead of calling the two functions, we call mps_to_mpo(),
    # which does exactly that:
    #   mps_as_puri = mp.mps_as_local_purification_mps(mps)
    #   mpo = mp.pmps_to_mpo(mps_as_puri)
    mpo = mm.mps_to_mpo(mps)
    # This is also a test of mp.mps_as_local_purification_mps() in the
    # following sense: Local purifications are representations of
    # mixed states. Therefore, compare mps and mps_as_puri by
    # converting them to mixed states.
    state = mps.to_array()
    state = np.outer(state, state.conj())
    state.shape = (local_dim,) * (2 * nr_sites)
    state2 = mpo.to_array_global()
    assert_array_almost_equal(state, state2)
