# encoding: utf-8


"""Auxiliary functions useful for writing tests"""


import itertools as it
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ._tools import local_to_global


def assert_mpa_almost_equal(mpa1, mpa2, full=False, **kwargs):
    """Verify that two MPAs are almost equal
    """
    if full:
        assert_array_almost_equal(mpa1.to_array(), mpa2.to_array(), **kwargs)
        return
    raise ValueError('not implemented yet')


def assert_mpa_identical(mpa1, mpa2):
    """Verify that two MPAs are complety identical
    """
    assert len(mpa1) == len(mpa2)
    for i, lten1, lten2 in zip(it.count(), mpa1, mpa2):
        assert_array_equal(lten1, lten2,
                           err_msg='mismatch in lten {}'.format(i))
    assert mpa1.normal_form == mpa2.normal_form
    # TODO: We should make a comprehensive comparison between `mpa1`
    # and `mpa2`.  Are we missing other things?


def mpo_to_global(mpo):
    """Convert mpo to dense global array

    .. todo:: Use `mpa.to_array_global()` instead.

    """
    return mpo.to_array_global()
