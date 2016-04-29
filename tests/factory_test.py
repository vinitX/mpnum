# encoding: utf-8


from __future__ import division, print_function

import numpy as np
import pytest as pt

import mpnum.factory as factory
import numpy as np


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', [(2, 3, 3), (3, 2, 4),
                                                       (6, 2, 4), (4, 3, 5),
                                                       (5, 2, 1)])
def test_mpdo_positivity(nr_sites, local_dim, bond_dim, rgen):
    rho = factory.random_mpdo(nr_sites, local_dim, bond_dim, rgen)
    rho_dense = rho.to_array_global().reshape((local_dim**nr_sites,) * 2)

    np.testing.assert_array_almost_equal(rho_dense, rho_dense.conj().T)
    lambda_min = min(np.real(np.linalg.eigvals(rho_dense)))
    assert lambda_min > -1e-14, "{} < -1e-14".format(lambda_min)
