#!/usr/bin/env python
# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import division, print_function

import numpy as np
import pytest as pt
from numpy.testing import assert_array_almost_equal, assert_array_equal, \
    assert_almost_equal, assert_equal
from six.moves import range # @UnresolvedImport

import mpnum.factory as factory
import mpnum.mparray as mp
from mpnum._tools import global_to_local, local_to_global
from testconf import POVM_TEST_PARAMETERS # @UnresolvedImport
from mpnum import _tools, povm

@pt.mark.parametrize('d', POVM_TEST_PARAMETERS)
def test_partial_trace(d):
    for povm_cls in povm.all_povms:
        p = povm_cls(opts={'d': d})
        element_sum = p.elements.sum(axis=0)
        element_sum.shape = (d, d)
        assert_array_almost_equal(element_sum, np.eye(d))
        linear_inversion_reconstruction = np.dot(p.linear_inversion_map, p.probability_map)
        if p.informationally_complete:
            assert_array_almost_equal(linear_inversion_reconstruction, np.eye(d**2), err_msg='POVM {} is supposed to be IC but is not'.format(povm_cls))
        else:
            assert np.abs(linear_inversion_reconstruction-np.eye(d**2)).max() > 0.1, 'POVM {} is not supposed to be IC but it is'.format(povm_cls)
