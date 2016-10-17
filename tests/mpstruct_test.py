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


def test_update_normalization(rgen):
    """Verify normalization after local tensor update

    We test two things:
    1. The normalization info after update is what we expect
    2. The normalization info is actually correct

    """
    ldim = 4
    bdim = 3
    mpa = factory.random_mpa(4, ldim, bdim, rgen)
    assert_correct_normalization(mpa, 0, 4)
    tensor = np.zeros((1, ldim, bdim))

    # Replacing in unnormalized tensor
    mpa.lt.update(0, tensor)
    assert_correct_normalization(mpa, 0, 4)
    mpa.lt.update(3, tensor.T)
    assert_correct_normalization(mpa, 0, 4)

    # Replacing in left-normalized part with unnormalized tensor
    mpa.normalize(3, 4)
    assert_correct_normalization(mpa, 3, 4)
    mpa.lt.update(3, tensor.T)
    assert_correct_normalization(mpa, 3, 4)
    mpa.lt.update(0, tensor)
    assert_correct_normalization(mpa, 0, 4)

    # Replacing in right-normalized part with unnormalized tensor
    mpa.normalize(0, 1)
    assert_correct_normalization(mpa, 0, 1)
    mpa.lt.update(0, tensor)
    assert_correct_normalization(mpa, 0, 1)
    mpa.lt.update(3, tensor.T)
    assert_correct_normalization(mpa, 0, 4)

    # Replacing in left-normalized part with normalized tensor
    mpa.normalize(3, 4)
    assert_correct_normalization(mpa, 3, 4)
    tensor = factory._zrandn((bdim * ldim, bdim), rgen)
    tensor, _ = np.linalg.qr(tensor)
    tensor = tensor.reshape((bdim, ldim, bdim))
    mpa.lt.update(2, tensor, normalization='left')
    assert_correct_normalization(mpa, 3, 4)
    mpa.lt.update(2, tensor, normalization='right')
    assert_correct_normalization(mpa, 2, 4)

    # Replacing in right-normalized part with normalized tensor
    mpa.normalize(0, 1)
    assert_correct_normalization(mpa, 0, 1)
    tensor = factory._zrandn((bdim, ldim * bdim), rgen).T
    tensor, _ = np.linalg.qr(tensor)
    tensor = tensor.T.reshape((bdim, ldim, bdim))
    mpa.lt.update(2, tensor, normalization='right')
    assert_correct_normalization(mpa, 0, 1)
    mpa.lt.update(2, tensor, normalization='left')
    assert_correct_normalization(mpa, 0, 3)


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
