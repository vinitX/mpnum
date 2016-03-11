# encoding: utf-8
#
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt

from numpy.testing import assert_array_almost_equal

import mpnum.mparray as mp
import mpnum.factory as factory
import mpnum.mpsmpo as mm
import mpnum._tools as _tools
from mparray_test import MP_TEST_PARAMETERS


def _get_reductions(red_fun, mpa, max_red_width):
    startstop = tuple(
        (start, start + width)
        for width in range(1, max_red_width)
        for start in range(len(mpa) - max_red_width + 1)
    )
    start = tuple(start for start, _ in startstop)
    stop = tuple(stop for _, stop in startstop)
    return start, stop, red_fun(mpa, startsites=start, stopsites=stop)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, max_red_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_mpo(nr_sites, local_dim, bond_dim, max_red_width, rgen):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen)
    op = mpo.to_array_global()

    start, stop, red = _get_reductions(mm.reductions_mpo, mpo, max_red_width)
    for start, stop, reduced_mpo in zip(start, stop, red):
        # Check that startsites/stopsites and width produce the same result:
        reduced_mpo2 = tuple(mm.reductions_mpo(mpo, stop - start))[start]
        assert_array_almost_equal(reduced_mpo.to_array(),
                                  reduced_mpo2.to_array())
        traceout = tuple(range(start)) + tuple(range(stop, nr_sites))
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(
            reduced_mpo.to_array_global(), red_from_op,
            err_msg="not equal at {}:{}".format(start, stop))

    # check default argument for startsite
    assert len(list(mm.reductions_mpo(mpo, max_red_width))) \
        == nr_sites - max_red_width + 1


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, max_red_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_pmps(nr_sites, local_dim, bond_dim, max_red_width, rgen):
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen)
    op = mm.pmps_to_mpo(pmps).to_array_global()

    start, stop, red = _get_reductions(mm.reductions_pmps, pmps, max_red_width)
    for start, stop, reduced_pmps in zip(start, stop, red):
        # Check that startsites/stopsites and width produce the same result:
        reduced_pmps2 = tuple(mm.reductions_pmps(pmps, stop - start))[start]
        # NB: reduced_pmps and reduced_pmps2 are in general not equal,
        # but red and red2 are.
        red = mm.pmps_to_mpo(reduced_pmps).to_array_global()
        red2 = mm.pmps_to_mpo(reduced_pmps2).to_array_global()
        assert_array_almost_equal(red, red2)
        traceout = tuple(range(start)) + tuple(range(stop, nr_sites))
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(
            red, red_from_op,
            err_msg="not equal at {}:{}".format(start, stop))

    # check default argument for startsite
    assert len(list(mm.reductions_pmps(pmps, max_red_width))) \
        == nr_sites - max_red_width + 1


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_mps(nr_sites, local_dim, bond_dim, width, rgen):
    mps = factory.random_mpa(nr_sites, (local_dim,), bond_dim,
                             randstate=rgen)
    mpo = mp.louter(mps, mps.conj())

    pmps_reds = mm.reductions_mps_as_mpo(mps, width)
    mpo_reds = mm.reductions_mpo(mpo, width)

    for red1, red2 in zip(pmps_reds, mpo_reds):
        assert_array_almost_equal(red1.to_array(), red2.to_array())


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_pmps_to_mpo(nr_sites, local_dim, bond_dim, rgen):
    if (nr_sites % 2) != 0:
        return
    nr_sites = nr_sites // 2
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen)
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
def test_mps_to_mpo(nr_sites, local_dim, bond_dim, rgen):
    mps = factory.random_mpa(nr_sites, local_dim, bond_dim, randstate=rgen)
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
