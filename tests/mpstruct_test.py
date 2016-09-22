# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import absolute_import, division, print_function

import functools as ft
import itertools as it

import h5py as h5
import numpy as np
import pytest as pt
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal)

from mpnum import factory
from mpnum.mpstruct import LocalTensors
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



