# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal)

from mpnum import factory
from mpnum.mpstruct import LocalTensors
from mpnum._testing import assert_correct_normalization
from six.moves import range, zip


def test_iter_readonly():
    mpa = factory.random_mpa(4, 2, 1)
    ltens = next(iter(mpa.lt))

    try:
        ltens[0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("Iterator over ltens should be read only")


UPDATE_N_SITES = 4
@pt.mark.parametrize(
    'mpa_norm',
    [(lnorm, rnorm)
     for rnorm in range(UPDATE_N_SITES) for lnorm in range(rnorm)])
@pt.mark.parametrize('upd_pos', range(UPDATE_N_SITES))
@pt.mark.parametrize('upd_norm', [None, 'left', 'right'])
def test_update_normalization(mpa_norm, upd_pos, upd_norm, rgen,
                              n_sites=UPDATE_N_SITES):
    """Verify normalization after local tensor update

    We test two things:
    1. The normalization info after update is what we expect 
       (in some special cases, see `norm_expected`)
    2. The normalization info is actually correct (in all cases)

    """
    n_sites = UPDATE_N_SITES
    ldim = 4
    bdim = 3
    mpa = factory.random_mpa(n_sites, ldim, bdim, rgen)
    assert_correct_normalization(mpa, 0, n_sites)

    mpa.normalize(*mpa_norm)
    assert_correct_normalization(mpa, *mpa_norm)

    dims = mpa.dims[upd_pos]
    tensor = factory._zrandn(dims, rgen)
    if upd_norm == 'left':
        tensor = tensor.reshape((-1, dims[-1]))
        tensor, _ = np.linalg.qr(tensor)
        tensor = tensor.reshape(dims)
    elif upd_norm == 'right':
        tensor = tensor.reshape((dims[0], -1)).T
        tensor, _ = np.linalg.qr(tensor)
        tensor = tensor.T.reshape(dims)

    norm_expected = {
        # Replacing in unnormalized tensor
        (0, n_sites, 0, None): (0, 4),
        (0, n_sites, 3, None): (0, 4),
        # Replacing in left-normalized part with unnormalized tensor
        (3, n_sites, 3, 'left'): (3, 4),
        (3, n_sites, 0, 'left'): (0, 4),
        # Replacing in right-normalized part with unnormalized tensor
        (0, 1, 0, None): (0, 1),
        (0, 1, 3, None): (0, 4),
        # Replacing in left-normalized part with normalized tensor
        (3, 4, 2, 'left'): (3, 4),
        (3, 4, 2, 'right'): (2, 4),
        # Replacing in right-normalized part with normalized tensor
        (0, 1, 2, 'right'): (0, 1),
        (0, 1, 2, 'left'): (0, 3),
    }
    expected = norm_expected.get((mpa_norm[0], mpa_norm[1], upd_pos, upd_norm),
                                 ())

    mpa.lt.update(upd_pos, tensor, upd_norm)
    assert_correct_normalization(mpa, *expected)


def test_getitem():
    ltens = factory.random_mpa(10, 2, 1).lt

    assert isinstance(ltens[0], np.ndarray)
    try:
        ltens[0][0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("Getitem should be read only")

    rest = list(ltens[1:])
    assert len(rest) == 9
    for n, lt in enumerate(rest):
        assert_array_equal(lt, ltens[1 + n])

        try:
            lt[0] = 0
        except ValueError:
            pass
        else:
            raise AssertionError("Getitem slice over ltens should be read only")
