#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import pytest as pt
import numpy as np
import mptom.mparray as mp
import mptom.factory as factory
import mptom._qmtools as qm


# List of test parameters (sites, local_dim that still allow for treatment of
# full representation
FULL_TEST_PARAMETERS = [(6, 2), (4, 3)]


###############################################################################
#                         Basic creation & operations                         #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim', FULL_TEST_PARAMETERS)
def test_mps_from_full(nr_sites, local_dim):
    psi = factory.random_vec(nr_sites, local_dim)
    mps = mp.MPArray.from_array(psi, 1)
    np.testing.assert_array_almost_equal(psi, mps.to_array())


@pt.mark.parametrize('nr_sites, local_dim', FULL_TEST_PARAMETERS)
def test_mpo_from_full(nr_sites, local_dim):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    np.testing.assert_array_almost_equal(op, mpo.to_array())


@pt.mark.parametrize('nr_sites, local_dim', FULL_TEST_PARAMETERS)
def test_mpo_conjugations(nr_sites, local_dim):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    np.testing.assert_array_almost_equal(np.conj(op), mpo.C().to_array())


@pt.mark.parametrize('nr_sites, local_dim', FULL_TEST_PARAMETERS)
def test_mpo_transposition(nr_sites, local_dim):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(qm.global_to_local(op, nr_sites), 2)

    opT = op.reshape((local_dim**nr_sites,) * 2).T \
        .reshape((local_dim,) * 2 * nr_sites)
    res = qm.local_to_global(mpo.T().to_array(), nr_sites)
    np.testing.assert_array_almost_equal(opT, res)
