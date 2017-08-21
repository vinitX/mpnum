# encoding: utf-8
"""Auxiliary functions useful for writing tests"""


def project_nonneg(values, imag_eps=1e-10, real_eps=1e-10, real_trunc=0.0):
    """Check that values are real and non-negative

    :param np.ndarray values: An ndarray of complex or real values (or
        a single value). `values` is modified in-place unless `values`
        is complex. A single value is also accepted.

    :param float imag_eps: Raise an error if imaginary parts with
        modulus larger than `imag_eps` are present.

    :param float real_eps: Raise an error if real parts smaller than
        `-real_eps` are present. Replace all remaining negative values
        by zero.

    :param float real_trunc: Replace positive real values smaller than
        or equal to `real_trunc` by zero.

    :returns: An ndarray of real values (or a single real value).

    If `values` is an array with complex type, a new array is
    returned. If `values` is an array with real type, it is modified
    in-place and returned.

    """
    if values.dtype.kind == 'c':
        assert (abs(values.imag) <= imag_eps).all()
        values = values.real.copy()
    if getattr(values, 'ndim', 0) == 0:
        assert values >= -real_eps
        return 0.0 if values <= real_trunc else values
    assert (values >= -real_eps).all()
    values[values <= real_trunc] = 0.0
    return values


def project_pmf(values, imag_eps=1e-10, real_eps=1e-10, real_trunc=0.0):
    """Check that values are real probabilities

    See :func:`check_nonneg_trunc` for parameters and return value. In
    addition, we check that `abs(values.sum() - 1.0)` is smaller than
    or equal to `real_eps` and divide `values` by `values.sum()`
    afterwards.

    """
    values = project_nonneg(values, imag_eps, real_eps, real_trunc)
    s = values.sum()
    assert abs(s - 1.0) <= real_eps
    values /= s
    return values
