#!/usr/bin/env python
# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import division, print_function

import itertools

import numpy as np
import pytest as pt
from numpy.testing import assert_array_almost_equal

import mpnum.mparray as mp
from mpnum import factory, povm


POVM_TEST_PARAMETERS_MPA = [(6, 2, 7), (3, 3, 3), (3, 6, 3), (3, 7, 4)]


@pt.mark.parametrize('d', [(2), (3), (6), (7)])
def test_povm_normalization_ic(d):
    for povm_cls in povm.all_povms:
        # Check that the POVM is normalized: elements must sum to the identity
        p = povm_cls(opts={'d': d})
        element_sum = p.elements.sum(axis=0)
        element_sum.shape = (d, d)
        assert_array_almost_equal(element_sum, np.eye(d))
        # Check that the attribute that says whether the POVM is IC is correct.
        linear_inversion_reconstruction = np.dot(p.linear_inversion_map, p.probability_map)
        if p.informationally_complete:
            assert_array_almost_equal(linear_inversion_reconstruction, np.eye(d**2), err_msg='POVM {} is supposed to be IC but is not'.format(povm_cls))
        else:
            assert np.abs(linear_inversion_reconstruction-np.eye(d**2)).max() > 0.1, 'POVM {} is not supposed to be IC but it is'.format(povm_cls)

@pt.mark.parametrize('nr_sites, d, bond_dim', POVM_TEST_PARAMETERS_MPA)
def test_povm_ic_mpa(nr_sites, d, bond_dim):
    # Check that the tensor product of the PauliGen POVM is IC.
    p = povm.PauliGen(opts={'d': d})
    probab_map = p.get_probability_map_mpa(nr_sites)
    inv_map = p.get_linear_inversion_map_mpo(nr_sites)
    reconstruction_map = mp.dot(inv_map, probab_map)
    eye = mp.MPArray.from_kron(itertools.repeat(np.eye(d**2), nr_sites))
    diff = reconstruction_map - eye
    diff_norm = mp.norm(diff)
    assert diff_norm < 1e-6
    # Check linear inversion for a particular example MPA.
    # Linear inversion works for arbitrary matrices, not only for states,
    # so we test it for an arbitrary MPA.
    mpa = factory.random_mpa(nr_sites, d**2, bond_dim)
    # Normalize somehow, otherwise the absolute error check below will not work.
    mpa /= mp.norm(mpa)
    probab = mp.dot(probab_map, mpa)
    rec = mp.dot(inv_map, probab)
    diff = mpa - rec
    diff_norm = mp.norm(diff)
    assert diff_norm < 1e-6

@pt.mark.parametrize('nr_sites, d, bond_dim', POVM_TEST_PARAMETERS_MPA)
def test_maxlik_R(nr_sites, d, bond_dim):
    # Check that the tensor product of the PauliGen POVM is IC.
    p = povm.PauliGen(opts={'d': d})
    mpa = factory.random_mpa(nr_sites, d**2, bond_dim)
    # Normalize somehow
    mpa /= mp.norm(mpa)
    probab = mp.dot(p.get_probability_map_mpa(nr_sites), mpa)
    probab = probab.to_array()
    mpa = factory.random_mpa(nr_sites, d**2, bond_dim)
    # Normalize somehow
    mpa /= mp.norm(mpa)
    R = p.maxlik_R(mpa, probab)
    # UNTESTED -- so far, we the only test is that the code returns something without error
    assert True
