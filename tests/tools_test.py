# encoding: utf-8

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_array_almost_equal
from mpnum import factory, _tools
from scipy.linalg import block_diag
from six.moves import range


def test_block_diag_simple(rgen):
    rows = (4, 7)
    cols = (3, 6)
    summands = [factory._zrandn((rows[i], cols[i]), randstate=rgen)
                for i in range(len(rows))]
    blockdiag_sum = _tools.block_diag(summands)
    blockdiag_sum_scipy = block_diag(*summands)
    assert_array_almost_equal(blockdiag_sum, blockdiag_sum_scipy)


def test_block_diag(rgen):
    leftmargin = 3
    rightmargin = 5
    rows = (4, 7)
    cols = (3, 6)
    nr_blocks = len(rows)
    nr_summands = 3
    leftvecs = factory._zrandn((nr_blocks, nr_summands, leftmargin),
                               randstate=rgen)
    middlematrices = [factory._zrandn((nr_summands, rows[i], cols[i]), randstate=rgen)
                      for i in range(nr_blocks)]
    rightvecs = factory._zrandn((nr_blocks, nr_summands, rightmargin),
                                randstate=rgen)
    blockdiag_summands = []
    for i in range(nr_blocks):
        summand = np.zeros((leftmargin, rows[i], cols[i], rightmargin), dtype=complex)
        for j in range(nr_summands):
            summand += np.outer(np.outer(leftvecs[i, j, :], middlematrices[i][j, :, :]),
                                rightvecs[i, j, :]).reshape(summand.shape)
        blockdiag_summands.append(summand)
    blockdiag_sum = _tools.block_diag(blockdiag_summands, axes=(1, 2))
    blockdiag_sum_explicit = np.zeros((leftmargin, sum(rows), sum(cols), rightmargin),
                                      dtype=complex)

    for i in range(nr_blocks):
        for j in range(nr_summands):
            summands = [middlematrices[i2][j]
                        if i2 == i else np.zeros_like(middlematrices[i2][j])
                        for i2 in range(nr_blocks)]
            middle = block_diag(*summands)
            blockdiag_sum_explicit += \
                np.outer(np.outer(leftvecs[i][j], middle), rightvecs[i][j]) \
                .reshape(blockdiag_sum_explicit.shape)

    assert_array_almost_equal(blockdiag_sum, blockdiag_sum_explicit)
