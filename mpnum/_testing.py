# encoding: utf-8


"""Auxiliary functions useful for writing tests"""


import itertools as it

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)


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

    for i, lten1, lten2 in zip(it.count(), mpa1.lt, mpa2.lt):
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


def _assert_lcanonical(ltens, msg=''):
    ltens = ltens.reshape((np.prod(ltens.shape[:-1]), ltens.shape[-1]))
    prod = ltens.conj().T.dot(ltens)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                              err_msg=msg)


def _assert_rcanonical(ltens, msg=''):
    ltens = ltens.reshape((ltens.shape[0], np.prod(ltens.shape[1:])))
    prod = ltens.dot(ltens.conj().T)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                              err_msg=msg)


def assert_correct_normalization(lt, lnormal_target=None, rnormal_target=None):
    """Verify that normalization info in `lt` is correct

    We check that `lt` is at least as normalized as specified by the
    information. `lt` being "more normalized" than the information
    specifies is admissible and not treated as an error.

    If `[lr]normal_target` are not None, verify that normalization
    info is exactly equal to the given values.

    """
    if hasattr(lt, 'lt'):
        lt = lt.lt  # We got an MPArray in lt, retrieve mpa.lt
    lnormal, rnormal = lt.normal_form

    # Verify that normalization info is correct
    for n in range(lnormal):
        _assert_lcanonical(lt[n], msg="Wrong left canonical (n={}/{})"
                          .format(n, lnormal))
    for n in range(rnormal, len(lt)):
        _assert_rcanonical(lt[n], msg="Wrong right canonical (n={}/{})"
                          .format(n, rnormal))

    # If targets are given, verify that the information in
    # `lt.normal_form` matches the targets.
    if lnormal_target is not None:
        assert_equal(lnormal, lnormal_target)
    if rnormal_target is not None:
        assert_equal(rnormal, rnormal_target)
