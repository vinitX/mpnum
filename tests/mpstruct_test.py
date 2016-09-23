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


def test_update_normalization():
    ltens = factory.random_mpa(4, 2, 1).lt
    tensor = np.array([0, 0])[None, :, None]

    # Replacing in unnormalized tensor
    ltens.update(0, tensor)
    assert ltens.normal_form == (0, 4)
    ltens.update(3, tensor)
    assert ltens.normal_form == (0, 4)

    # Replacing in left-normalized part with unnormalized tensor
    ltens._lnormalized, ltens._rnormalized = (3, 4)
    ltens.update(3, tensor)
    assert ltens.normal_form == (3, 4)
    ltens.update(0, tensor)
    assert ltens.normal_form == (0, 4)

    # Replacing in right-normalized part with unnormalized tensor
    ltens._lnormalized, ltens._rnormalized = (0, 1)
    ltens.update(0, tensor)
    assert ltens.normal_form == (0, 1)
    ltens.update(3, tensor)
    assert ltens.normal_form == (0, 4)

    # Replacing in left-normalized part with normalized tensor
    ltens._lnormalized, ltens._rnormalized = (3, 4)
    ltens.update(2, tensor, normalization='left')
    assert ltens.normal_form == (3, 4)
    ltens.update(2, tensor, normalization='right')
    assert ltens.normal_form == (2, 4)

    # Replacing in right-normalized part with normalized tensor
    ltens._lnormalized, ltens._rnormalized = (0, 1)
    ltens.update(2, tensor, normalization='right')
    assert ltens.normal_form == (0, 1)
    ltens.update(2, tensor, normalization='left')
    assert ltens.normal_form == (0, 3)


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
