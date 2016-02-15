# encoding: utf-8


"""Auxiliary functions useful for writing tests"""


import itertools as it

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


def assert_mpa_almost_equal(mpa1, mpa2, full=False, **kwargs):
    """Verify that two MPAs are almost equal
    """
    if full:
        assert_array_almost_equal(mpa1.to_array(), mpa2.to_array(), **kwargs)
    else:
        raise NotImplementedError("Coming soon...")


def assert_mpa_identical(mpa1, mpa2, decimal=np.infty):
    """Verify that two MPAs are complety identical
    """
    assert len(mpa1) == len(mpa2)
    assert mpa1.normal_form == mpa2.normal_form

    for i, lten1, lten2 in zip(it.count(), mpa1, mpa2):
        if decimal is np.infty:
            assert_array_equal(lten1, lten2,
                            err_msg='mismatch in lten {}'.format(i))
        else:
            assert_array_almost_equal(lten1, lten2, decimal=decimal,
                                      err_msg='mismatch in lten {}'.format(i))
    # TODO: We should make a comprehensive comparison between `mpa1`
    # and `mpa2`.  Are we missing other things?


# FIXME If we have the method, we dont need this function
def mpo_to_global(mpo):
    """Convert mpo to dense global array

    .. todo:: Use `mpa.to_array_global()` instead.

    """
    return mpo.to_array_global()
