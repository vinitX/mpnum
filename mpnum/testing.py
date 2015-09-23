# encoding: utf-8


"""Auxiliary functions useful for writing tests"""


import itertools as it
from numpy.testing import assert_array_equal, assert_array_almost_equal


def params_product(*iterables):
    """Return all combinations of parameter lists

    :func:`tuplize` may be useful for lists that contain only a single
    parameter:

    >>> params_product(((1, 11), (2, 22)), tuplize(('a', 'b')))
    ((1, 11, 'a'), (1, 11, 'b'), (2, 22, 'a'), (2, 22, 'b'))

    """
    return tuple(tuple(it.chain(*setting))
                 for setting in it.product(*iterables))


def tuplize(iterable):
    """tuple((x,) for x in iterable) says it all

    >>> tuplize([1, 2, 3])
    ((1,), (2,), (3,))
    >>> tuplize(range(3))
    ((0,), (1,), (2,))

    """
    return tuple((x,) for x in iterable)


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
