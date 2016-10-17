# encoding: utf-8

"""Linear algebra with matrix product arrays

Currently, we support computing ground states (i.e. minimal eigenvalue
and eigenvector).

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse.linalg import eigs

from six.moves import range

from . import mparray as mp
from . import _tools
from ._named_ndarray import named_ndarray
from .factory import random_mpa

__all__ = ['mineig']


def _mineig_leftvec_add(leftvec, mpo_lten, mps_lten):
    """Add one column to the left vector.

    :param leftvec: existing left vector
        It has three indices: mps bond, mpo bond, complex conjugate mps bond
    :param op_lten: Local tensor of the MPO
    :param mps_lten: Local tensor of the current MPS eigenstate

    leftvecs[i] is L_{i-1}, See [Sch11_, arXiv version, Fig. 39 ond
    p. 63 and Fig. 38 and Eq. (191) on p. 62].  Regarding Fig. 39,
    things are as follows:

    Figure:

    Upper row: MPS matrices
    Lower row: Complex Conjugate MPS matrices
    Middle row: MPO matrices with row (column) indices to bottom (top)

    Figure, left part:

    a_{i-1} (left): 'mps_bond' of leftvec
    a_{i-1} (right): 'left_mps_bond' of mps_lten
    b_{i-1} (left): 'mpo_bond' of leftvec
    b_{i-1} (right): 'left_mpo_bond' of mpo_lten
    a'_{i-1} (left): 'cc_mps_bond' of leftvec
    a'_{i+1} (left): 'left_mps_bond' of mps_lten.conj()
    a_i: 'right_mps_bond' of mps_lten
    b_i: 'right_mpo_bond' of mpo_lten
    a'_i: 'right_mps_bond' of mps_lten.conj()

    """
    leftvec_names = ('mps_bond', 'mpo_bond', 'cc_mps_bond')
    mpo_names = ('left_mpo_bond', 'phys_row', 'phys_col', 'right_mpo_bond')
    mps_names = ('left_mps_bond', 'phys', 'right_mps_bond')
    leftvec = named_ndarray(leftvec, leftvec_names)
    mpo_lten = named_ndarray(mpo_lten, mpo_names)
    mps_lten = named_ndarray(mps_lten, mps_names)

    contract_mps = (('mps_bond', 'left_mps_bond'),)
    leftvec = leftvec.tensordot(mps_lten, contract_mps)
    rename_mps = (('right_mps_bond', 'mps_bond'),)
    leftvec = leftvec.rename(rename_mps)

    contract_mpo = (
        ('mpo_bond', 'left_mpo_bond'),
        ('phys', 'phys_col'))
    leftvec = leftvec.tensordot(mpo_lten, contract_mpo)
    contract_cc_mps = (
        ('cc_mps_bond', 'left_mps_bond'),
        ('phys_row', 'phys'))
    leftvec = leftvec.tensordot(mps_lten.conj(), contract_cc_mps)
    rename_mps_mpo = (
        ('right_mpo_bond', 'mpo_bond'),
        ('right_mps_bond', 'cc_mps_bond'))
    leftvec = leftvec.rename(rename_mps_mpo)

    leftvec = leftvec.to_array(leftvec_names)
    return leftvec


def _mineig_rightvec_add(rightvec, mpo_lten, mps_lten):
    """Add one column to the right vector.

    :param rightvec: existing right vector
        It has three indices: mps bond, mpo bond, complex conjugate mps bond
    :param op_lten: Local tensor of the MPO
    :param mps_lten: Local tensor of the current MPS eigenstate

    This does the same thing as _mineig_leftvec_add(), except that
    'left' and 'right' are exchanged in the contractions (but not in
    the axis names of the input tensors).

    """
    rightvec_names = ('mps_bond', 'mpo_bond', 'cc_mps_bond')
    mpo_names = ('left_mpo_bond', 'phys_row', 'phys_col', 'right_mpo_bond')
    mps_names = ('left_mps_bond', 'phys', 'right_mps_bond')
    rightvec = named_ndarray(rightvec, rightvec_names)
    mpo_lten = named_ndarray(mpo_lten, mpo_names)
    mps_lten = named_ndarray(mps_lten, mps_names)

    contract_mps = (('mps_bond', 'right_mps_bond'),)
    rightvec = rightvec.tensordot(mps_lten, contract_mps)
    rename_mps = (('left_mps_bond', 'mps_bond'),)
    rightvec = rightvec.rename(rename_mps)

    contract_mpo = (
        ('mpo_bond', 'right_mpo_bond'),
        ('phys', 'phys_col'))
    rightvec = rightvec.tensordot(mpo_lten, contract_mpo)
    contract_cc_mps = (
        ('cc_mps_bond', 'right_mps_bond'),
        ('phys_row', 'phys'))
    rightvec = rightvec.tensordot(mps_lten.conj(), contract_cc_mps)
    rename_mps_mpo = (
        ('left_mpo_bond', 'mpo_bond'),
        ('left_mps_bond', 'cc_mps_bond'))
    rightvec = rightvec.rename(rename_mps_mpo)

    rightvec = rightvec.to_array(rightvec_names)
    return rightvec


def _mineig_local_op(leftvec, mpo_ltens, rightvec):
    """Create the operator for local eigenvalue minimization on one site.

    :param leftvec: Left vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param mpo_ltens: List of local tensors of the MPO
    :param rightvec: Right vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond

    See [Sch11_, arXiv version, Fig. 38 on p. 62].  If len(mpo_ltens)
    == 1, this method implements the contractions across the dashed
    lines in the figure. For let(mpo_ltens) > 1, we return the
    operator for what is probably called "multi-site DMRG".

    Indices and axis names map as follows:

    Upper row: MPS matrices
    Lower row: Complex Conjugate MPS matrices
    Middle row: MPO matrices with row (column) indices to bottom (top)

    a_{i-1}: 'mps_bond' of leftvec
    a'_{i-1}: 'cc_mps_bond' of leftvec
    a_i: 'mps_bond' of rightvec
    a'_i: 'mps_bond' of rightvec
    sigma_i: 'phys_col' of mpo_lten
    sigma'_i: 'phys_row' of mpo_lten

    """
    # Produce one MPO local tensor supported on len(mpo_ltens) sites.
    nr_sites = len(mpo_ltens)
    mpo_lten = mpo_ltens[0]
    for lten in mpo_ltens[1:]:
        mpo_lten = _tools.matdot(mpo_lten, lten)
    mpo_lten = _tools.local_to_global(mpo_lten, nr_sites,
                                      left_skip=1, right_skip=1)
    s = mpo_lten.shape
    mpo_lten = mpo_lten.reshape(
        (s[0], np.prod(s[1:1 + nr_sites]), np.prod(s[1 + nr_sites:-1]), s[-1]))

    # Do the contraction mentioned above.
    leftvec_names = ('left_mps_bond', 'left_mpo_bond', 'left_cc_mps_bond')
    mpo_names = ('left_mpo_bond', 'phys_row', 'phys_col', 'right_mpo_bond')
    rightvec_names = ('right_mps_bond', 'right_mpo_bond', 'right_cc_mps_bond')
    leftvec = named_ndarray(leftvec, leftvec_names)
    mpo_lten = named_ndarray(mpo_lten, mpo_names)
    rightvec = named_ndarray(rightvec, rightvec_names)

    contract = (('left_mpo_bond', 'left_mpo_bond'),)
    op = leftvec.tensordot(mpo_lten, contract)
    contract = (('right_mpo_bond', 'right_mpo_bond'),)
    op = op.tensordot(rightvec, contract)

    op_names = (
        'left_cc_mps_bond', 'phys_row', 'right_cc_mps_bond',
        'left_mps_bond', 'phys_col', 'right_mps_bond',
    )
    op = op.to_array(op_names)
    op = op.reshape((np.prod(op.shape[0:3]), -1))
    return op


def _mineig_minimize_locally(leftvec, mpo_ltens, rightvec, eigvec_ltens,
                             user_eigs_opts=None):
    """Perform the local eigenvalue minimization on one site on one site.

    Return a new (expectedly smaller) eigenvalue and a new local
    tensor for the MPS eigenvector.

    :param leftvec: Left vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param mpo_ltens: List of local tensors of the MPO
    :param rightvec: Right vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param eigvec_ltens: List of local tensors of the MPS eigenvector
    :returns: mineigval, mineigval_eigvec_lten

    See [Sch11_, arXiv version, Fig. 42 on p. 67].  This method
    computes the operator ('op'), defined by everything except the
    circle of the first term in the figure. It then obtains the
    minimal eigenvalue (lambda in the figure) and eigenvector (circled
    part / single matrix in the figure).

    We use the figure as follows:

    Upper row: MPS matrices
    Lower row: Complex Conjugate MPS matrices
    Middle row: MPO matrices with row (column) indices to bottom (top)

    """
    # FIXME Do we really need this conversion?
    mpo_ltens = list(mpo_ltens)
    eigvec_ltens = list(eigvec_ltens)

    eigs_opts = {'k': 1, 'which': 'SR', 'tol': 1e-6}
    if user_eigs_opts is not None:
        eigs_opts.update(user_eigs_opts)
    if eigs_opts['k'] != 1:
        raise ValueError('Supplying k != 1 in requires changes in the code, '
                         'k={} was requested'.format(user_eigs_opts['k']))
    op = _mineig_local_op(leftvec, mpo_ltens, rightvec)
    eigvec_bonddim = max(lten.shape[0] for lten in eigvec_ltens)
    eigvec_lten = eigvec_ltens[0]
    for lten in eigvec_ltens[1:]:
        eigvec_lten = _tools.matdot(eigvec_lten, lten)
    eigvals, eigvecs = eigs(op, v0=eigvec_lten.flatten(), **eigs_opts)
    eigval_pos = eigvals.real.argmin()
    eigval = eigvals[eigval_pos]
    eigvec_lten = eigvecs[:, eigval_pos].reshape(eigvec_lten.shape)
    if len(eigvec_ltens) == 1:
        eigvec_lten = (eigvec_lten,)
    else:
        # If we minimize on multiple sites, we must compress to the
        # desired bond dimension.
        #
        # TODO: Return the truncation error.
        #
        # "the truncation error of conventional DMRG [...] has emerged
        # as a highly reliable tool for gauging the quality of
        # results" [Sch11_, Sec. 6.4, p. 74]
        eigvec_lten = mp.MPArray.from_array(eigvec_lten, 1, has_bond=True)
        eigvec_lten.compress(method='svd', bdim=eigvec_bonddim)
        eigvec_lten = eigvec_lten.lt
    return eigval, eigvec_lten


def mineig(mpo,
           startvec=None, startvec_bonddim=None, randstate=None,
           max_num_sweeps=5, eigs_opts=None, minimize_sites=1):
    """Iterative search for smallest eigenvalue and eigenvector of an MPO.

    Algorithm: [Sch11_, Sec. 6.3]

    :param MPArray mpo: A matrix product operator (MPA with two physical legs)
    :param startvec: initial guess for eigenvector (default random MPS with
        bond `startvec_bonddim`)
    :param startvec_bonddim: Bond dimension of random start vector if
        no start vector is given. (default: Use the bond dimension of `mpo`)
    :param randstate: numpy.random.RandomState instance or None
    :param max_num_sweeps: Maximum number of sweeps to do (default 5)
    :param eigs_opts: kwargs for `scipy.sparse.linalg.eigs()`. If you
        supple `which`, you will probably not obtain the minimal
        eigenvalue. `k` different from one is not supported at the moment.
    :param int minimize_sites: Number of connected sites minimization should
        be performed on (default 1)
    :returns: mineigval, mineigval_eigvec_mpa

    We minimize the eigenvalue by obtaining the minimal eigenvalue of
    an operator supported on 'minimize_sites' many sites. For
    minimize_sites=1, this is called "variational MPS ground state
    search" or "single-site DMRG" [Sch11_, Sec. 6.3, p. 69]. For
    minimize_sites>1, this is called "multi-site DMRG".

    Comments on the implementation, for minimize_sites=1:

    References are to the arXiv version of [Sch11_] assuming we replace
    zero-based with one-based indices there.

    leftvecs[i] is L_{i-1}  \
    rightvecs[i] is R_{i}   |  See Fig. 38 and Eq. (191) on p. 62.
    mpo[i] is W_{i}         /
    eigvec[i] is M_{i}         This is just the MPS matrix.

    Psi^A_{i-1} and Psi^B_{i} are identity matrices because of
    normalization. (See Fig. 42 on p. 67 and the text; see also
    Figs. 14 and 15 and pages 28 and 29.)

    """
    # Possible TODOs:
    #  - Can we refactor this function into several shorter functions?
    #  - compute the overlap between 'eigvec' from successive iterations
    #    to check whether we have converged
    #  - compute var(H) = <psi| H^2 |psi> - (<psi| H |psi>)^2 every n-th
    #    iteration to check whether we have converged (this criterion is
    #    better but more expensive to compute)
    #  - increase the bond dimension of 'eigvec' if var(H) remains above
    #    a given threshold
    #  - for multi-site updates, track the error in the SVD truncation
    #    (see comment there why)
    #  - return these details for tracking errors in larger computations

    nr_sites = len(mpo)
    assert nr_sites - minimize_sites > 0, (
        'Require ({} =) nr_sites > minimize_sites (= {})'
        .format(nr_sites, minimize_sites))

    if startvec is None:
        pdims = max(dim[0] for dim in mpo.pdims)
        if startvec_bonddim is None:
            startvec_bonddim = max(mpo.bdims)
        if startvec_bonddim == 1:
            raise ValueError('startvec_bonddim must be at least 2')

        startvec = random_mpa(nr_sites, pdims, startvec_bonddim,
                              randstate=randstate)
        startvec /= mp.norm(startvec)
    # Can we avoid this overly complex check by improving
    # _mineig_minimize_locally()? eigs() will fail under the excluded
    # conditions because of too small matrices.
    assert not any(bdim12 == (1, 1) for bdim12 in
                   zip((1,) + startvec.bdims, startvec.bdims + (1,))), \
        'startvec must not contain two consecutive bonds of dimension 1, ' \
        'bdims including dummy bonds = (1,) + {!r} + (1,)' \
            .format(startvec.bdims)
    # For
    #
    #   pos in range(nr_sites - minimize_sites),
    #
    # we find the ground state of an operator supported on
    #
    #   range(pos, pos_end),  pos_end = pos + minimize_sites
    #
    # leftvecs[pos] and rightvecs[pos] contain the vectors needed to
    # construct that operator for that. Therefore, leftvecs[pos] is
    # constructed from matrices on
    #
    #   range(0, pos - 1)
    #
    # and rightvecs[pos] is constructed from matrices on
    #
    #   range(pos_end, nr_sites),  pos_end = pos + minimize_sites
    eigvec = startvec
    eigvec.normalize(right=1)
    leftvecs = [np.array(1, ndmin=3)] + [None] * (nr_sites - minimize_sites)
    rightvecs = [None] * (nr_sites - minimize_sites) + [np.array(1, ndmin=3)]
    for pos in reversed(range(nr_sites - minimize_sites)):
        rightvecs[pos] = _mineig_rightvec_add(rightvecs[pos + 1],
                                                 mpo.lt[pos + minimize_sites],
                                                 eigvec.lt[pos + minimize_sites])

    # The iteration pattern is very similar to
    # :func:`mpnum.mparray.MPArray._adapt_to()`. See there for more
    # comments.
    for num_sweep in range(max_num_sweeps):
        # Sweep from left to right
        for pos in range(nr_sites - minimize_sites + 1):
            if pos == 0 and num_sweep > 0:
                # Don't do first site again if we are not in the first sweep.
                continue

            if pos > 0:
                eigvec.normalize(left=pos)
                rightvecs[pos - 1] = None
                leftvecs[pos] = _mineig_leftvec_add(
                    leftvecs[pos - 1], mpo.lt[pos - 1], eigvec.lt[pos - 1])
            pos_end = pos + minimize_sites
            eigval, eigvec_lten = _mineig_minimize_locally(
                leftvecs[pos], mpo.lt[pos:pos_end], rightvecs[pos],
                eigvec.lt[pos:pos_end], eigs_opts)
            eigvec.lt[pos:pos_end] = eigvec_lten

        # Sweep from right to left (don't do last site again)
        for pos in reversed(range(nr_sites - minimize_sites)):
            pos_end = pos + minimize_sites
            if pos < nr_sites - minimize_sites:
                # We always do this, because we don't do the last site again.
                eigvec.normalize(right=pos_end)
                leftvecs[pos + 1] = None
                rightvecs[pos] = _mineig_rightvec_add(
                    rightvecs[pos + 1], mpo.lt[pos_end], eigvec.lt[pos_end])
            eigval, eigvec_lten = _mineig_minimize_locally(
                leftvecs[pos], mpo.lt[pos:pos_end], rightvecs[pos],
                eigvec.lt[pos:pos_end], eigs_opts)
            eigvec.lt[pos:pos_end] = eigvec_lten

    return eigval, eigvec

