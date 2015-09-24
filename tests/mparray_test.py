# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.linalg import svd
from numpy.testing import assert_array_almost_equal, assert_array_equal, \
    assert_almost_equal, assert_equal
from six.moves import range, zip

import mpnum.factory as factory
import mpnum.mparray as mp
from mpnum import _tools
from mpnum._tools import global_to_local
from mpnum.testing import params_product, tuplize, assert_mpa_almost_equal, \
    assert_mpa_identical

try:
    from collections import ChainMap
except ImportError:
    # Python 2 :(
    # Let's make up a cheap replacement:
    def ChainMap(*dicts):
        res = {}
        for d in reversed(dicts):
            res.update(d)
        return res


# nr_sites, local_dim, bond_dim
MP_TEST_PARAMETERS = [(1, 7, np.nan), (2, 3, 3), (3, 2, 4), (6, 2, 4),
                      (4, 3, 5), (5, 2, 1)]
# nr_sites, local_dim, bond_dim, sites_per_group
MP_TEST_PARAMETERS_GROUPS = [(6, 2, 4, 3), (6, 2, 4, 2), (4, 3, 5, 2)]


# We choose to use mostly a global reperentation of multipartite
# arrays throughout our tests. A few operations (i.e. matrix
# multiplication) are easier to express. Sometimes, local
# representation will be used as well.


###############################################################################
#                         Basic creation & operations                         #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_from_full(nr_sites, local_dim, _):
    psi = factory.random_vec(nr_sites, local_dim)
    mps = mp.MPArray.from_array(psi, 1)
    assert_array_almost_equal(psi, mps.to_array())

    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    assert_array_almost_equal(op, mpo.to_array())


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_from_kron(nr_sites, local_dim, bond_dim):
    plegs = 2
    factors = tuple(factory._zrandn([nr_sites] + ([local_dim] * plegs)))
    op = _tools.mkron(*factors)
    op.shape = [local_dim] * (plegs * nr_sites)
    mpo = mp.MPArray.from_kron(factors)
    assert_array_almost_equal(op, mpo.to_array_global())


@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_conjugations(nr_sites, local_dim, _):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    assert_array_almost_equal(np.conj(op), mpo.conj().to_array())


@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_transposition(nr_sites, local_dim, _):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(global_to_local(op, nr_sites), 2)

    opT = op.reshape((local_dim**nr_sites,) * 2).T \
        .reshape((local_dim,) * 2 * nr_sites)
    assert_array_almost_equal(opT, mpo.T().to_array_global())


###############################################################################
#                            Algebraic operations                             #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_dot(nr_sites, local_dim, bond_dim):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op1 = mpo1.to_array_global()
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op2 = mpo2.to_array_global()

    # Dotproduct of all 1st physical with 0th physical legs = np.dot
    dot_np = np.tensordot(op1.reshape((local_dim**nr_sites, ) * 2),
                          op2.reshape((local_dim**nr_sites, ) * 2),
                          axes=([1], [0]))
    dot_np = dot_np.reshape(op1.shape)
    dot_mp = mp.dot(mpo1, mpo2, axes=(1, 0)).to_array_global()
    assert_array_almost_equal(dot_np, dot_mp)
    # this should also be the default axes
    dot_mp = mp.dot(mpo1, mpo2).to_array_global()
    assert_array_almost_equal(dot_np, dot_mp)

    # Dotproduct of all 0th physical with 1st physical legs = np.dot
    dot_np = np.tensordot(op1.reshape((local_dim**nr_sites, ) * 2),
                          op2.reshape((local_dim**nr_sites, ) * 2),
                          axes=([0], [1]))
    dot_np = dot_np.reshape(op1.shape)
    dot_mp = mp.dot(mpo1, mpo2, axes=(0, 1)).to_array_global()
    assert_array_almost_equal(dot_np, dot_mp)
    # this should also be the default axes
    dot_mp = mp.dot(mpo1, mpo2, axes=(-2, -1)).to_array_global()
    assert_array_almost_equal(dot_np, dot_mp)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_partialdot(nr_sites, local_dim, bond_dim):
    # Only for at least two sites, we can apply an operator to a part
    # of a chain.
    if nr_sites < 2:
        return
    part_sites = nr_sites // 2
    start_at = min(2, nr_sites // 2)

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)
    mpo_part = factory.random_mpa(part_sites, (local_dim, local_dim), bond_dim)
    op_part = mpo_part.to_array_global().reshape((local_dim**part_sites,) * 2)
    op_part_embedded = np.kron(
        np.kron(np.eye(local_dim**start_at), op_part),
        np.eye(local_dim**(nr_sites - part_sites - start_at)))

    prod1 = np.dot(op, op_part_embedded)
    prod2 = np.dot(op_part_embedded, op)
    prod1_mpo = mp.partialdot(mpo, mpo_part, start_at=start_at)
    prod2_mpo = mp.partialdot(mpo_part, mpo, start_at=start_at)
    prod1_mpo = prod1_mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)
    prod2_mpo = prod2_mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)

    assert_array_almost_equal(prod1, prod1_mpo)
    assert_array_almost_equal(prod2, prod2_mpo)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_inner_vec(nr_sites, local_dim, bond_dim):
    mp_psi1 = factory.random_mpa(nr_sites, local_dim, bond_dim)
    psi1 = mp_psi1.to_array().ravel()
    mp_psi2 = factory.random_mpa(nr_sites, local_dim, bond_dim)
    psi2 = mp_psi2.to_array().ravel()

    inner_np = np.vdot(psi1, psi2)
    inner_mp = mp.inner(mp_psi1, mp_psi2)
    assert_almost_equal(inner_mp, inner_np)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_inner_mat(nr_sites, local_dim, bond_dim):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op1 = mpo1.to_array_global().reshape((local_dim**nr_sites, ) * 2)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op2 = mpo2.to_array_global().reshape((local_dim**nr_sites, ) * 2)

    inner_np = np.trace(np.dot(op1.conj().transpose(), op2))
    inner_mp = mp.inner(mpo1, mpo2)
    assert_almost_equal(inner_mp, inner_np)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_norm(nr_sites, local_dim, bond_dim):
    mp_psi = factory.random_mpa(nr_sites, local_dim, bond_dim)
    psi = mp_psi.to_array()

    assert_almost_equal(mp.inner(mp_psi, mp_psi), mp.norm(mp_psi)**2)
    assert_almost_equal(np.sum(psi.conj() * psi), mp.norm(mp_psi)**2)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normdist(nr_sites, local_dim, bond_dim):
    psi1 = factory.random_mpa(nr_sites, local_dim, bond_dim)
    psi2 = factory.random_mpa(nr_sites, local_dim, bond_dim)

    assert_almost_equal(mp.normdist(psi1, psi2), mp.norm(psi1 - psi2))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, keep_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_partialtrace(nr_sites, local_dim, bond_dim, keep_width):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global()

    for site in range(nr_sites - keep_width + 1):
        traceout = tuple(range(site)) \
            + tuple(range(site + keep_width, nr_sites))
        axes = [(0, 1) if site in traceout else None for site in range(nr_sites)]
        red_mpo = mp.partialtrace(mpo, axes=axes)
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(red_mpo.to_array_global(), red_from_op,
                                  err_msg="not equal at site {}".format(site))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_trace(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global().reshape((local_dim**nr_sites,) * 2)

    assert_almost_equal(np.trace(op), mp.trace(mpo))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_add_and_subtr(nr_sites, local_dim, bond_dim):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op1 = mpo1.to_array_global()
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op2 = mpo2.to_array_global()

    assert_array_almost_equal(op1 + op2, (mpo1 + mpo2).to_array_global())
    assert_array_almost_equal(op1 - op2, (mpo1 - mpo2).to_array_global())

    mpo1 += mpo2
    assert_array_almost_equal(op1 + op2, mpo1.to_array_global())


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', [(3, 2, 2)])
def test_operations_typesafety(nr_sites, local_dim, bond_dim):
    # create a real MPA
    mpo1 = factory._generate(nr_sites, (local_dim, local_dim), bond_dim,
                             func=lambda shape: np.random.randn(*shape))
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)

    assert mpo1[0].dtype == float
    assert mpo2[0].dtype == complex

    assert (mpo1 + mpo1)[0].dtype == float
    assert (mpo1 + mpo2)[0].dtype == complex
    assert (mpo2 + mpo1)[0].dtype == complex

    assert (mpo1 - mpo1)[0].dtype == float
    assert (mpo1 - mpo2)[0].dtype == complex
    assert (mpo2 - mpo1)[0].dtype == complex

    mpo1 += mpo2
    assert mpo1[0].dtype == complex


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mult_mpo_scalar(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    # For nr_sites == 1, changing `mpo` below will change `op` as
    # well, unless we call .copy().
    op = mpo.to_array_global().copy()
    scalar = np.random.randn()

    assert_array_almost_equal(scalar * op, (scalar * mpo).to_array_global())

    mpo *= scalar
    assert_array_almost_equal(scalar * op, mpo.to_array_global())


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_div_mpo_scalar(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    # For nr_sites == 1, changing `mpo` below will change `op` as
    # well, unless we call .copy().
    op = mpo.to_array_global().copy()
    scalar = np.random.randn()

    assert_array_almost_equal(op / scalar, (mpo / scalar).to_array_global())

    mpo /= scalar
    assert_array_almost_equal(op / scalar, mpo.to_array_global())


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_outer(nr_sites, local_dim, bond_dim):
    # This test produces at most `nr_sites` by tensoring two
    # MPOs. This doesn't work for :code:`nr_sites = 1`.
    if nr_sites < 2:
        return

    # NOTE: Everything here is in local form!!!
    mpo = factory.random_mpa(nr_sites // 2, (local_dim, local_dim), bond_dim)
    op = mpo.to_array()

    # Test with 2-factors with full form
    mpo_double = mp.outer((mpo, mpo))
    op_double = np.tensordot(op, op, axes=(tuple(), ) * 2)
    assert len(mpo_double) == 2 * len(mpo)
    assert_array_almost_equal(op_double, mpo_double.to_array())
    assert_array_equal(mpo_double.bdims, mpo.bdims + (1,) + mpo.bdims)

    # Test 3-factors iteratively (since full form would be too large!!
    diff = mp.outer((mpo, mpo, mpo)) - mp.outer((mpo, mp.outer((mpo, mpo))))
    diff.normalize(allbutone=True)
    assert len(diff) == 3 * len(mpo)
    assert mp.norm(diff) < 1e-6


@pt.mark.parametrize('_, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_inject(_, local_dim, bond_dim):
    # bond_dim is np.nan for nr_sites = 1 (first argument,
    # ignored). We require a value for bond_dim.
    if np.isnan(bond_dim):
        return

    # plegs = 3 is hardcoded below (argument to .transpose()).
    # Uniform local dimension is also hardcoded below (arguments to
    # .reshape()).
    plegs = 3
    local_dim = (local_dim,) * plegs

    a, b, c = factory._zrandn((3, 2) + local_dim)
    # We don't use b[1, :]
    b = b[0, :]
    # Here, only global order (as given by np.kron()).
    abbc0 = _tools.mkron(a[0, :], b, b, c[0, :])
    abbc1 = _tools.mkron(a[1, :], b, b, c[1, :])
    abbc = (abbc0 + abbc1).reshape(4 * local_dim)
    ac0 = np.kron(a[0, :], c[0, :])
    ac1 = np.kron(a[1, :], c[1, :])
    ac = (ac0 + ac1).reshape(2 * local_dim)
    ac_mpo = mp.MPArray.from_array(global_to_local(ac, sites=2), plegs)
    abbc_mpo = mp.inject(ac_mpo, pos=1, num=2, inject_ten=b)
    abbc_from_mpo = abbc_mpo.to_array_global()
    assert_array_almost_equal(abbc, abbc_from_mpo)

    # Here, only local order.
    ac = factory._zrandn(local_dim * 2)
    b = factory._zrandn(local_dim)
    acb = np.tensordot(ac, b, axes=((), ()))
    abc = acb.transpose((0, 1, 2, 6, 7, 8, 3, 4, 5))
    ac_mpo = mp.MPArray.from_array(ac, plegs)
    abc_mpo = mp.inject(ac_mpo, pos=1, num=1, inject_ten=b)
    # Keep local order
    abc_from_mpo = abc_mpo.to_array()
    assert_array_almost_equal(abc, abc_from_mpo)

    # plegs = 2 is hardcoded below (argument to .transpose()).
    # Uniform local dimension is also hardcoded below (arguments to
    # .reshape()).
    plegs = 2
    local_dim = (local_dim[0],) * plegs

    a, c = factory._zrandn((2, 2) + local_dim)
    b = np.eye(local_dim[0])
    # Here, only global order (as given by np.kron()).
    abbc0 = _tools.mkron(a[0, :], b, b, c[0, :])
    abbc1 = _tools.mkron(a[1, :], b, b, c[1, :])
    abbc = (abbc0 + abbc1).reshape(4 * local_dim)
    ac0 = np.kron(a[0, :], c[0, :])
    ac1 = np.kron(a[1, :], c[1, :])
    ac = (ac0 + ac1).reshape(2 * local_dim)
    ac_mpo = mp.MPArray.from_array(global_to_local(ac, sites=2), plegs)
    abbc_mpo = mp.inject(ac_mpo, pos=1, num=2, inject_ten=None)
    abbc_from_mpo = abbc_mpo.to_array_global()
    assert_array_almost_equal(abbc, abbc_from_mpo)

    # Here, only local order.
    ac = factory._zrandn(local_dim * 2)
    b = np.eye(local_dim[0])
    acb = np.tensordot(ac, b, axes=((), ()))
    abc = acb.transpose((0, 1, 4, 5, 2, 3))
    ac_mpo = mp.MPArray.from_array(ac, plegs)
    abc_mpo = mp.inject(ac_mpo, pos=1, num=1, inject_ten=None)
    # Keep local order
    abc_from_mpo = abc_mpo.to_array()
    assert_array_almost_equal(abc, abc_from_mpo)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, local_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_local_sum(nr_sites, local_dim, bond_dim, local_width):
    eye_mpa = factory.eye(1, local_dim)

    def embed_mpa(mpa, startpos):
        mpas = [eye_mpa] * startpos + [mpa] + \
               [eye_mpa] * (nr_sites - startpos - local_width)
        res = mp.outer(mpas)
        return res

    rs = np.random.RandomState(seed=0)
    nr_startpos = nr_sites - local_width + 1
    mpas = [factory.random_mpa(local_width, (local_dim,) * 2, bond_dim,
                               randstate=rs)
            for i in range(nr_startpos)]

    # Embed with mp.outer() and calculate naive MPA sum:
    mpas_embedded = [embed_mpa(mpa, i) for i, mpa in enumerate(mpas)]
    mpa_sum = mpas_embedded[0]
    for mpa in mpas_embedded[1:]:
        mpa_sum += mpa

    # Compare with local_sum: Same result, smaller bond
    # dimension.
    mpa_local_sum = mp.local_sum(mpas)

    assert all(d1 <= d2 for d1, d2 in zip(mpa_local_sum.bdims, mpa_sum.bdims))
    assert_array_almost_equal(mpa_local_sum.to_array(), mpa_sum.to_array())


###############################################################################
#                         Shape changes, conversions                          #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim, bond_dim, sites_per_group',
                     MP_TEST_PARAMETERS_GROUPS)
def test_group_sites(nr_sites, local_dim, bond_dim, sites_per_group):
    assert (nr_sites % sites_per_group) == 0, \
        'nr_sites not a multiple of sites_per_group'
    mpa = factory.random_mpa(nr_sites, (local_dim,) * 2, bond_dim)
    grouped_mpa = mpa.group_sites(sites_per_group)
    op = mpa.to_array()
    grouped_op = grouped_mpa.to_array()
    assert_array_almost_equal(op, grouped_op)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, sites_per_group',
                     MP_TEST_PARAMETERS_GROUPS)
def test_split_sites(nr_sites, local_dim, bond_dim, sites_per_group):
    assert (nr_sites % sites_per_group) == 0, \
        'nr_sites not a multiple of sites_per_group'
    mpa = factory.random_mpa(nr_sites // sites_per_group,
                             (local_dim,) * (2 * sites_per_group), bond_dim)
    split_mpa = mpa.split_sites(sites_per_group)
    op = mpa.to_array()
    split_op = split_mpa.to_array()
    assert_array_almost_equal(op, split_op)


###############################################################################
#                         Normalization & Compression                         #
###############################################################################
def assert_lcanonical(ltens, msg=''):
    ltens = ltens.reshape((np.prod(ltens.shape[:-1]), ltens.shape[-1]))
    prod = ltens.conj().T.dot(ltens)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                              err_msg=msg)


def assert_rcanonical(ltens, msg=''):
    ltens = ltens.reshape((ltens.shape[0], np.prod(ltens.shape[1:])))
    prod = ltens.dot(ltens.conj().T)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                              err_msg=msg)


def assert_correct_normalization(mpo, lnormal_target=None, rnormal_target=None):
    lnormal, rnormal = mpo.normal_form

    # If no targets are given, verify that the data matches the
    # information in `mpo.normal_form`.
    lnormal_target = lnormal_target or lnormal
    rnormal_target = rnormal_target or rnormal

    # If targets are given, verify that the information in
    # `mpo.normal_form` matches the targets.
    assert_equal(lnormal, lnormal_target)
    assert_equal(rnormal, rnormal_target)

    for n in range(lnormal):
        assert_lcanonical(mpo[n], msg="Failure left canonical (n={}/{})"
                          .format(n, lnormal_target))
    for n in range(rnormal, len(mpo)):
        assert_rcanonical(mpo[n], msg="Failure right canonical (n={}/{})"
                          .format(n, rnormal_target))


@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_normalization_from_full(nr_sites, local_dim, _):
    op = factory.random_op(nr_sites, local_dim)
    mpo = mp.MPArray.from_array(op, 2)
    assert_correct_normalization(mpo, nr_sites - 1, nr_sites)


# FIXME Add counter to normalization functions
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normalization_incremental(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global()
    assert_correct_normalization(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo.to_array_global())

    for site in range(1, nr_sites):
        mpo.normalize(left=site)
        assert_correct_normalization(mpo, site, nr_sites)
        assert_array_almost_equal(op, mpo.to_array_global())

    for site in range(nr_sites - 1, 0, -1):
        mpo.normalize(right=site)
        assert_correct_normalization(mpo, site - 1, site)
        assert_array_almost_equal(op, mpo.to_array_global())


# FIXME Add counter to normalization functions
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normalization_jump(nr_sites, local_dim, bond_dim):
    # This test assumes at least two sites.
    if nr_sites == 1:
        return

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global()
    assert_correct_normalization(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo.to_array_global())

    center = nr_sites // 2
    mpo.normalize(left=center - 1, right=center)
    assert_correct_normalization(mpo, center - 1, center)
    assert_array_almost_equal(op, mpo.to_array_global())


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normalization_full(nr_sites, local_dim, bond_dim):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global()
    assert_correct_normalization(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo.to_array_global())

    mpo.normalize(right=1)
    assert_correct_normalization(mpo, 0, 1)
    assert_array_almost_equal(op, mpo.to_array_global())

    ###########################################################################
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global()
    assert_correct_normalization(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo.to_array_global())

    mpo.normalize(left=len(mpo) - 1)
    assert_correct_normalization(mpo, len(mpo) - 1, len(mpo))
    assert_array_almost_equal(op, mpo.to_array_global())


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normalization_default_args(nr_sites, local_dim, bond_dim):
    # The following normalizations assume at least two sites.
    if nr_sites == 1:
        return

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    assert_correct_normalization(mpo, 0, nr_sites)

    mpo.normalize(left=1)
    mpo.normalize(allbutone=True)
    assert_correct_normalization(mpo, nr_sites - 1, nr_sites)

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    assert_correct_normalization(mpo, 0, nr_sites)

    # The following normalization assumes at least three sites.
    if nr_sites == 2:
        return

    mpo.normalize(left=1)
    mpo.normalize(right=nr_sites - 2)
    mpo.normalize(allbutone=True)
    assert_correct_normalization(mpo, 0, 1)


def test_normalization_compression():
    """If the bond dimension is too large at the boundary, qr decompostion
    in normalization may yield smaller bond dimension"""
    mpo = factory.random_mpa(sites=2, ldim=2, bdim=20)
    mpo.normalize(right=1)
    assert_correct_normalization(mpo, 0, 1)
    assert mpo.bdims[0] == 2

    mpo = factory.random_mpa(sites=2, ldim=2, bdim=20)
    mpo.normalize(left=1)
    assert_correct_normalization(mpo, 1, 2)
    assert mpo.bdims[0] == 2


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mult_mpo_scalar_normalization(nr_sites, local_dim, bond_dim):
    if nr_sites < 2:
        # Re-normalization has no effect for nr_sites == 1. There is
        # nothing more to test than :func:`test_mult_mpo_scalar`.
        return

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim)
    op = mpo.to_array_global()
    scalar = np.random.randn()

    center = nr_sites // 2
    mpo.normalize(left=center - 1, right=center)
    mpo_times_two = scalar * mpo

    assert_array_almost_equal(scalar * op, mpo_times_two.to_array_global())
    assert_correct_normalization(mpo_times_two, center - 1, center)

    mpo *= scalar
    assert_array_almost_equal(scalar * op, mpo.to_array_global())
    assert_correct_normalization(mpo, center - 1, center)


#####################################
#  SVD and variational compression  #
#####################################

# nr_sites, local_dims, bond_dim
COMPR_SIZES = (
    (2, (2, 3), 5),
    (5, 3, 4),
    # TODO Some of the following settings take very long. Read the
    # pytest documentation and find a way to conveniently run tests
    # with larger parameter sets.
    # (4, (2, 3), 5),
    # (6, 2, 3),
    # (5, (2, 2, 2), 20),  # about  2 minutes (Core i5-3380M)
    # (16, 2, 10),         # about  2 minutes
    # (16, 2, 30),         # about 10 minutes
)

COMPR_SETTINGS = tuplize((
    dict(method='svd', direction='left'),
    dict(method='svd', direction='right'),
    dict(method='svd', direction='left', relerr=1e-6),
    dict(method='svd', direction='right', relerr=1e-6),
    dict(method='var', num_sweeps=1, var_sites=1),
    dict(method='var', num_sweeps=2, var_sites=1),
    dict(method='var', num_sweeps=3, var_sites=1),
    dict(method='var', num_sweeps=1, var_sites=2),
    dict(method='var', num_sweeps=2, var_sites=2),
    dict(method='var', num_sweeps=3, var_sites=2),
    dict(method='var', num_sweeps=2, var_sites=2, startmpa='fillbelow'),
))

COMPR_NORM = tuplize((
    None, dict(allbutone=True),
    dict(left=1), dict(left=-1), dict(right=1), dict(right=-1),
    dict(left=1, right=2), dict(left=-2, right=-1), dict(left=1, right=-1),
))

COMPR_SETTINGS = params_product(COMPR_SIZES, COMPR_NORM, COMPR_SETTINGS)


def normalize_if_applicable(mpa, nmz):
    """Check whether the given normalization can be applied.

    :param mp.MPArray mpa: Will call `mpa.normalize()`
    :param nmz: Keyword arguments for `mpa.normalize()` or `None`

    :returns: True if the normalization has been applied.

    `nmz=None` means not to call `mpa.normalize()` at all.

    The test whether the normalization can be applied is not
    comprehensive.

    """
    if nmz is not None:
        if 'allbutone' not in nmz and len(mpa) == 1:
            return False
        if nmz.get('left') == 1 and nmz.get('right') == -1 and len(mpa) == 2:
            return False
        mpa.normalize(**nmz)
    return True


def call_compression(mpa, comparg, bonddim, call_compress=False):
    """Add 'bdim' if relerr is not given. Add 'startmpa' if requested.

    """
    if 'relerr' in comparg:
        pass
    elif comparg.get('startmpa') == 'fillbelow':
        startmpa = factory.random_mpa(len(mpa), mpa.pdims[0], bonddim, norm1=True)
        comparg = ChainMap({'startmpa': startmpa}, comparg)
    else:
        comparg = ChainMap({'bdim': bonddim}, comparg)
    if call_compress:
        return mpa.compress(**comparg)
    else:
        return mpa.compression(**comparg)


@pt.mark.parametrize(
    'nr_sites, local_dims, bond_dim, normalize, comparg', COMPR_SETTINGS)
def test_compression_and_compress(nr_sites, local_dims, bond_dim, normalize, comparg):
    """Test that .compression() and .compress() produce identical results.

    """
    mpa = factory.random_mpa(nr_sites, local_dims, bond_dim * 2, norm1=True)
    if not normalize_if_applicable(mpa, normalize):
        return

    comparg = comparg.copy()
    if comparg['method'] == 'var':
        # Exact equality between `compr` and `compr2` below requires a
        # fixed start vector.
        comparg['startmpa'] = factory.random_mpa(nr_sites, local_dims, bond_dim)

    # The results from .compression() and .compress() must match
    # exactly. No numerical difference is allowed.
    compr2 = mpa.copy()
    overlap2 = call_compression(compr2, comparg, bond_dim, call_compress=True)
    compr, overlap = call_compression(mpa, comparg, bond_dim)
    assert overlap == overlap2
    assert_mpa_identical(compr, compr2)


@pt.mark.parametrize(
    'nr_sites, local_dims, bond_dim, normalize, comparg', COMPR_SETTINGS)
def test_compression_result_properties(nr_sites, local_dims, bond_dim,
                                        normalize, comparg):
    """Test general properties of the MPA coming from a compression.

    TODO: The worst case for compression is that all singular values
    have the same size.  This gives a fidelity lower bound for the
    compression result.  Check that lower bound.

    FIXME: Make this test a wrapper around MPArray.compression() to
    reduce code duplication.  This wrapper would replace
    call_compression().  This would also apply more tests
    .compress(). At the moment, we mostly test .compression().

    """
    st = None
    if comparg['method'] == 'var' and comparg['num_sweeps'] == 3:
        # Do a large number of sweeps and use a fixed seed to compare
        # with SVD compression below.
        comparg = ChainMap({'num_sweeps': 20 // comparg['var_sites']}, comparg)
        st = np.random.RandomState(seed=42)

    mpa = factory.random_mpa(nr_sites, local_dims, bond_dim * 2, st, norm1=True)
    if not normalize_if_applicable(mpa, normalize):
        return
    compr, overlap = call_compression(mpa.copy(), comparg, bond_dim)

    # 'relerr' is currently 1e-6 and no bond_dim is provided, so no
    # compression will occur.
    if 'relerr' not in comparg:
        # Check that the bond dimension has changed.
        assert compr.bdim < mpa.bdim
        # Check that the target bond dimension is satisfied
        assert compr.bdim <= bond_dim

    # Check that the overlap is correct.
    assert_almost_equal(overlap, mp.inner(mpa, compr))

    # SVD: Check that .normal_form is as expected.
    if comparg['method'] == 'svd':
        normtarget = {'left': (0, 1), 'right': (len(compr) - 1, len(compr))}
        assert compr.normal_form == normtarget[comparg['direction']]

    # Check the content of .normal_form is correct.
    assert_correct_normalization(compr)

    # SVD: compare with alternative implementation
    if comparg['method'] == 'svd' and 'relerr' not in comparg:
        alt_compr = _svd_compression_full(mpa, comparg['direction'], bond_dim)
        compr = compr.to_array()
        assert_array_almost_equal(alt_compr, compr)

    if comparg['method'] == 'var' and comparg['num_sweeps'] > 5:
        right_svd_res = _svd_compression_full(mpa, 'right', bond_dim)
        left_svd_res = _svd_compression_full(mpa, 'left', bond_dim)
        array = mpa.to_array()
        right_svd_overlap = np.abs(np.vdot(array, right_svd_res))
        left_svd_overlap = np.abs(np.vdot(array, left_svd_res))
        overlap_rel_tol = 1e-6
        assert abs(overlap) >= right_svd_overlap * (1 - overlap_rel_tol)
        assert abs(overlap) >= left_svd_overlap * (1 - overlap_rel_tol)


@pt.mark.parametrize(
    'nr_sites, local_dims, bond_dim, normalize, comparg', COMPR_SETTINGS)
def test_compression_bonddim_noincrease(nr_sites, local_dims, bond_dim,
                                         normalize, comparg):
    """Compression to larger bond dimension doesn't increase bond
    dimension.

    """
    if 'relerr' in COMPR_SETTINGS:
        return  # Test does not apply
    mpa = factory.random_mpa(nr_sites, local_dims, bond_dim, norm1=True)
    if not normalize_if_applicable(mpa, normalize):
        return

    for factor in (1, 2):
        compr, overlap = call_compression(mpa, comparg, bond_dim * factor)
        assert_almost_equal(overlap, 1)
        assert_mpa_almost_equal(compr, mpa, full=True)
        assert (np.array(compr.bdims) <= np.array(mpa.bdims)).all()


@pt.mark.parametrize(
    'nr_sites, local_dims, bond_dim, normalize, comparg, add',
    params_product(COMPR_SETTINGS, (('zero',), ('self',))))
def test_compression_trivialsum(nr_sites, local_dims, bond_dim, normalize, comparg, add):
    """`a + b` compresses exactly to a multiple of `a` if `b` is `0`, `a`
    or `-2*a`

    """
    mpa = factory.random_mpa(nr_sites, local_dims, bond_dim, norm1=True)
    if not normalize_if_applicable(mpa, normalize):
        return
    zero = factory.zero(nr_sites, local_dims, bond_dim)
    choices = {'zero': (zero, 1), 'self': (mpa, 2), 'self2': (-2*mpa, -1)}
    add, factor = choices[add]

    msum = mpa + add
    assert_mpa_almost_equal(msum, factor * mpa, full=True)

    # Check that bond dimension has increased (they exactly add)
    for dim1, dim2, sum_dim in zip(mpa.bdims, add.bdims, msum.bdims):
        assert dim1 + dim2 == sum_dim

    compr, overlap = call_compression(msum, comparg, bond_dim)
    assert_almost_equal(overlap, factor**2)
    assert_mpa_almost_equal(compr, factor * mpa, full=True)
    assert (np.array(compr.bdims) <= np.array(mpa.bdims)).all()


#######################################
#  Compression test helper functions  #
#######################################
def _svd_compression_full(mpa, direction, target_bonddim):
    """Re-implement MPArray.compress('svd') but on the level of the full
    matrix representation, i.e. it truncates the Schmidt-decompostion
    on each bipartition sequentially.

    Two implementations that produce the same data are not a guarantee
    for correctness, but a check for consistency is nice anyway.

    :param mpa: The MPA to compress
    :param direction: 'right' means sweep from left to right,
        'left' vice versa
    :param target_bonddim: Compress to this bond dimension
    :returns: Result as numpy.ndarray

    """
    def singlecut(array, nr_left, plegs, target_bonddim):
        array_shape = array.shape
        array = array.reshape((np.prod(array_shape[:nr_left * plegs]), -1))
        u, s, v = svd(array, full_matrices=False)
        u = u[:, :target_bonddim]
        s = s[:target_bonddim]
        v = v[:target_bonddim, :]
        opt_compr = np.dot(u * s, v)
        opt_compr = opt_compr.reshape(array_shape)
        return opt_compr

    array = mpa.to_array()
    plegs = mpa.plegs[0]
    nr_sites = len(mpa)
    if direction == 'right':
        nr_left_values = range(1, nr_sites)
    else:
        nr_left_values = range(nr_sites-1, 0, -1)
    for nr_left in nr_left_values:
        array = singlecut(array, nr_left, plegs, target_bonddim)
    return array
