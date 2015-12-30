# encoding: utf-8


from __future__ import division, print_function

import itertools as it
from inspect import isfunction

import numpy as np
import pytest as pt
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mpnum.mparray as mp
import mpnum.povm as povm
import mpnum.povm.mppovm as mppovm
import mpnum.factory as factory
import mpnum.mpsmpo as mpsmpo

ALL_POVMS = {name: constructor for name, constructor in povm.__dict__.items()
             if name.endswith('_povm') and isfunction(constructor)}


def mp_from_array_repeat(array, nr_sites):
    """Generate a MPA representation of the `nr_sites`-fold tensor product of
    array.
    """
    mpa = mp.MPArray.from_array(array)
    return mp.outer(it.repeat(mpa, nr_sites))


@pt.mark.parametrize('dim', [(2), (3), (6), (7)])
def test_povm_normalization_ic(dim):
    for name, constructor in ALL_POVMS.items():
        # Check that the POVM is normalized: elements must sum to the identity
        current_povm = constructor(dim)
        element_sum = sum(iter(current_povm))
        assert_array_almost_equal(element_sum, np.eye(dim))

        # Check that the attribute that says whether the POVM is IC is correct.
        linear_inversion_recons = np.dot(current_povm.linear_inversion_map,
                                         current_povm.probability_map)
        if current_povm.informationally_complete:
            assert_array_almost_equal(
                linear_inversion_recons, np.eye(dim**2),
                err_msg='POVM {} is not informationally complete'.format(name))
        else:
            assert np.abs(linear_inversion_recons - np.eye(dim**2)).max() > 0.1, \
                'POVM {} is informationally complete'.format(name)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim',
                     [(6, 2, 7), (3, 3, 3), (3, 6, 3), (3, 7, 4)])
def test_povm_ic_mpa(nr_sites, local_dim, bond_dim):
    # Check that the tensor product of the PauliGen POVM is IC.
    paulis = povm.pauli_povm(local_dim)
    inv_map = mp_from_array_repeat(paulis.linear_inversion_map, nr_sites)
    probab_map = mp_from_array_repeat(paulis.probability_map, nr_sites)
    reconstruction_map = mp.dot(inv_map, probab_map)

    eye = factory.eye(nr_sites, local_dim**2)
    assert mp.norm(reconstruction_map - eye) < 1e-5

    # Check linear inversion for a particular example MPA.
    # Linear inversion works for arbitrary matrices, not only for states,
    # so we test it for an arbitrary MPA.
    mpa = factory.random_mpa(nr_sites, local_dim**2, bond_dim)
    # Normalize, otherwise the absolute error check below will not work.
    mpa /= mp.norm(mpa)
    probabs = mp.dot(probab_map, mpa)
    recons = mp.dot(inv_map, probabs)
    assert mp.norm(recons - mpa) < 1e-6


@pt.mark.parametrize('nr_sites, width, local_dim, bond_dim',
                     [(6, 3, 2, 5), (4, 2, 3, 4)])
def test_mppovm_expectation(nr_sites, width, local_dim, bond_dim):
    paulis = povm.pauli_povm(local_dim)
    mppaulis = mppovm.MPPovm.from_local_povm(paulis, width)
    rho = factory.random_mpdo(nr_sites, local_dim, bond_dim)
    pmap = paulis.probability_map
    expectations = list(mppaulis.expectations(rho))
    reductions = mpsmpo.reductions_mpo(rho, width)

    assert len(expectations) == nr_sites - width + 1
    for ssite, evals_mp in enumerate(expectations):
        evals_np = next(reductions)
        evals_np = evals_np.ravel().to_array()

        for _ in range(width):
            evals_np = np.tensordot(evals_np, pmap, axes=(0, 1))

        assert_array_equal(evals_np.shape, (len(paulis),) * width)
        assert_array_almost_equal(evals_mp.to_array(), evals_np)
