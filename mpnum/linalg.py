# encoding: utf-8


from __future__ import absolute_import, division, print_function
from six.moves import range

import numpy as np

from scipy.sparse.linalg import eigs

import mpnum
import mpnum.factory
import mpnum.mparray as mp
import mpnum._tools as _tools


def _variational_compression_leftvec_add(leftvec, compr_lten, tgt_lten):
    """Add one column to the left vector.

    :param leftvec: existing left vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param compr_lten: Local tensor of the compressed MPS
    :param tgt_lten: Local tensor of the target MPS

    Construct L from [Sch11, Fig. 27, p. 48]. We have compr_lten in
    the top row of the figure without complex conjugation and tgt_lten
    in the bottom row with complex conjugation.

    """
    leftvec_names = ('compr_bond', 'tgt_bond')
    compr_names = ('compr_left_bond', 'compr_phys', 'compr_right_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    leftvec = mpnum.named_ndarray(leftvec, leftvec_names)
    compr_lten = mpnum.named_ndarray(compr_lten, compr_names)
    tgt_lten = mpnum.named_ndarray(tgt_lten, tgt_names)

    contract_compr_mps = (('compr_bond', 'compr_left_bond'),)
    leftvec = leftvec.tensordot(compr_lten, contract_compr_mps)

    contract_tgt_mps = (
        ('compr_phys', 'tgt_phys'),
        ('tgt_bond', 'tgt_left_bond'))
    leftvec = leftvec.tensordot(tgt_lten.conj(), contract_tgt_mps)
    rename_mps_mpo = (
        ('compr_right_bond', 'compr_bond'),
        ('tgt_right_bond', 'tgt_bond'))
    leftvec = leftvec.rename(rename_mps_mpo)

    leftvec = leftvec.to_array(leftvec_names)
    return leftvec


def _variational_compression_rightvec_add(rightvec, compr_lten, tgt_lten):
    """Add one column to the right vector.

    :param rightvec: existing right vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param compr_lten: Local tensor of the compressed MPS
    :param tgt_lten: Local tensor of the target MPS

    Construct R from [Sch11, Fig. 27, p. 48]. See comments in
    _variational_compression_leftvec_add() for further details.

    """
    rightvec_names = ('compr_bond', 'tgt_bond')
    compr_names = ('compr_left_bond', 'compr_phys', 'compr_right_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    rightvec = mpnum.named_ndarray(rightvec, rightvec_names)
    compr_lten = mpnum.named_ndarray(compr_lten, compr_names)
    tgt_lten = mpnum.named_ndarray(tgt_lten, tgt_names)

    contract_compr_mps = (('compr_bond', 'compr_right_bond'),)
    rightvec = rightvec.tensordot(compr_lten, contract_compr_mps)

    contract_tgt_mps = (
        ('compr_phys', 'tgt_phys'),
        ('tgt_bond', 'tgt_right_bond'))
    rightvec = rightvec.tensordot(tgt_lten.conj(), contract_tgt_mps)
    rename = (
        ('compr_left_bond', 'compr_bond'),
        ('tgt_left_bond', 'tgt_bond'))
    rightvec = rightvec.rename(rename)

    rightvec = rightvec.to_array(rightvec_names)
    return rightvec


def _variational_compression_new_lten(leftvec, tgt_ltens, rightvec, max_bonddim):
    """Create new local tensors for the compressed MPS.

    :param leftvec: Left vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param tgt_ltens: List of local tensor of the target MPS
    :param rightvec: Right vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param int max_bonddim: Maximal bond dimension of the result

    Compute the right-hand side of [Sch11, Fig. 27, p. 48]. We have
    compr_lten in the top row of the figure without complex
    conjugation and tgt_lten in the bottom row with complex
    conjugation.

    For len(tgt_ltens) > 1, compute the right-hand side of [Sch11,
    Fig. 29, p. 49].

    """
    # Produce one MPS local tensor supported on len(tgt_ltens) sites.
    tgt_lten = tgt_ltens[0]
    for lten in tgt_ltens[1:]:
        tgt_lten = _tools.matdot(tgt_lten, lten)
    tgt_lten_shape = tgt_lten.shape
    tgt_lten = tgt_lten.reshape((tgt_lten_shape[0], -1, tgt_lten_shape[-1]))

    # Contract the middle part with the left and right parts.
    leftvec_names = ('compr_left_bond', 'tgt_left_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    rightvec_names = ('compr_right_bond', 'tgt_right_bond')
    leftvec = mpnum.named_ndarray(leftvec, leftvec_names)
    tgt_lten = mpnum.named_ndarray(tgt_lten, tgt_names)
    rightvec = mpnum.named_ndarray(rightvec, rightvec_names)

    contract = (('tgt_left_bond', 'tgt_left_bond'),)
    compr_lten = leftvec.tensordot(tgt_lten.conj(), contract)
    contract = (('tgt_right_bond', 'tgt_right_bond'),)
    compr_lten = compr_lten.tensordot(rightvec, contract)

    compr_lten_names = (
        'compr_left_bond', 'tgt_phys', 'compr_right_bond'
    )
    compr_lten = compr_lten.to_array(compr_lten_names).conj()
    s = compr_lten.shape
    compr_lten = compr_lten.reshape((s[0],) + tgt_lten_shape[1:-1] + (s[-1],))

    if len(tgt_ltens) == 1:
        compr_ltens = (compr_lten,)
    else:
        # [Sch11, p. 49] says that we can go with QR instead of SVD
        # here. However, will generally increase the bond dimension of
        # our compressed MPS, which we do not want.
        compr_ltens = mp.MPArray.from_array(compr_lten, plegs=1, has_bond=True)
        compr_ltens.compress(method='svd', max_bd=max_bonddim)
    return compr_ltens


def variational_compression(
        mpa,
        startvec=None, startvec_bonddim=None, startvec_randstate=None,
        max_num_sweeps=5, minimize_sites=1):
    """Iterative compression of an MPA.

    Algorithm: [Sch11, Sec. 4.5.2]. All references refer to the arXiv
    version of [Sch11].

    Possible TODOs:

    - implement calculating the overlap between 'compr' and 'mpa' from
      the norm of 'compr', given that 'mpa' is normalized

    - track overlap between 'compr' and 'mpa' and stop sweeping if it
      is small

    - maybe increase bond dimension of given error cannot be reached

    - Shall we track the error in the SVD truncation for multi-site
      updates? [Sch11] says it turns out to be useful in actual DMRG.

    - return these details for tracking errors in larger computations

    :param MPArray mpa: The matrix product array to be compressed

    :param startvec_bonddim: Bond dimension of random start vector if
        no start vector is given. Use the bond dimension of the MPA if
        None.

    :param startvec: Start vector; generate a random start vector if
        None.

    :param startvec_randstate: numpy.random.RandomState instance or None

    :param max_num_sweeps: Maximum number of sweeps to do. Currently,
        always do that many sweeps.

    :param int minimize_sites: Minimize distance by changing that many
        sites simultaneously.

    :returns: compressed_mpa

    """
    nr_sites = len(mpa)
    mpa_old_shape = None
    if max(mpa.plegs) > 1:
        mpa_old_shape = mpa.pdims
        mpa = mpa.pleg_reshape((-1,))
    compr = startvec
    if compr is None:
        if startvec_randstate is None:
            startvec_randstate = np.random
        pdims = max(dim[0] for dim in mpa.pdims)
        if startvec_bonddim is None:
            startvec_bonddim = max(mpa.bdims)
        compr = mpnum.factory.random_mpa(nr_sites, pdims, startvec_bonddim,
                                         randstate=startvec_randstate)
        compr /= mp.norm(compr)
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
    leftvecs = [np.array(1, ndmin=2)] + [None] * (nr_sites - minimize_sites)
    rightvecs = [None] * (nr_sites - minimize_sites) + [np.array(1, ndmin=2)]
    compr.normalize(right=1)
    for pos in range(nr_sites - minimize_sites - 1, -1, -1):
        rightvecs[pos] = _variational_compression_rightvec_add(
            rightvecs[pos + 1], compr[pos + minimize_sites], mpa[pos + minimize_sites])
    max_bonddim = max(compr.bdims)

    for num_sweep in range(max_num_sweeps):

        # Sweep from left to right
        for pos in range(nr_sites - minimize_sites + 1):
            if pos == 0 and num_sweep > 0:
                # Don't do first site again if we are not in the first
                # sweep.
                continue
            if pos > 0:
                compr.normalize(left=pos)
                rightvecs[pos - 1] = None
                leftvecs[pos] = _variational_compression_leftvec_add(
                    leftvecs[pos - 1], compr[pos - 1], mpa[pos - 1])
            pos_end = pos + minimize_sites
            compr_lten = _variational_compression_new_lten(
                leftvecs[pos], mpa[pos:pos_end], rightvecs[pos], max_bonddim)
            compr[pos:pos_end] = compr_lten

        # Sweep from right to left (don't do last site again)
        for pos in range(nr_sites - minimize_sites - 1, -1, -1):
            pos_end = pos + minimize_sites
            if pos < nr_sites - minimize_sites:
                # We always do this, because we don't do the last site again.
                compr.normalize(right=pos + minimize_sites)
                leftvecs[pos + 1] = None
                rightvecs[pos] = _variational_compression_rightvec_add(
                    rightvecs[pos + 1], compr[pos_end], mpa[pos_end])
            compr_lten = _variational_compression_new_lten(
                leftvecs[pos], mpa[pos:pos_end], rightvecs[pos], max_bonddim)
            compr[pos:pos_end] = compr_lten

    if mpa_old_shape is not None:
        compr = compr.pleg_reshape(mpa_old_shape)
    return compr


def _mineig_leftvec_add(leftvec, mpo_lten, mps_lten):
    """Add one column to the left vector.

    :param leftvec: existing left vector
        It has three indices: mps bond, mpo bond, complex conjugate mps bond
    :param op_lten: Local tensor of the MPO
    :param mps_lten: Local tensor of the current MPS eigenstate

    leftvecs[i] is L_{i-1}, See [Sch11, arXiv version, Fig. 39 ond
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
    leftvec = mpnum.named_ndarray(leftvec, leftvec_names)
    mpo_lten = mpnum.named_ndarray(mpo_lten, mpo_names)
    mps_lten = mpnum.named_ndarray(mps_lten, mps_names)

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
    rightvec = mpnum.named_ndarray(rightvec, rightvec_names)
    mpo_lten = mpnum.named_ndarray(mpo_lten, mpo_names)
    mps_lten = mpnum.named_ndarray(mps_lten, mps_names)

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

    See [Sch11, arXiv version, Fig. 38 on p. 62].  If len(mpo_ltens)
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
    leftvec = mpnum.named_ndarray(leftvec, leftvec_names)
    mpo_lten = mpnum.named_ndarray(mpo_lten, mpo_names)
    rightvec = mpnum.named_ndarray(rightvec, rightvec_names)

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
                             eigs_opts=None):
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

    See [Sch11, arXiv version, Fig. 42 on p. 67].  This method
    computes the operator ('op'), defined by everything except the
    circle of the first term in the figure. It then obtains the
    minimal eigenvalue (lambda in the figure) and eigenvector (circled
    part / single matrix in the figure).

    We use the figure as follows:

    Upper row: MPS matrices
    Lower row: Complex Conjugate MPS matrices
    Middle row: MPO matrices with row (column) indices to bottom (top)

    """
    if eigs_opts is None:
        eigs_opts = {'k': 1, 'which': 'SR', 'tol': 1e-6}
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
        # results" [Sch11, Sec. 6.4, p. 74]
        eigvec_lten = mp.MPArray.from_array(eigvec_lten, 1, has_bond=True)
        eigvec_lten.compress(method='svd', max_bdim=eigvec_bonddim)
    return eigval, eigvec_lten


def mineig(mpo,
           startvec=None, startvec_bonddim=None, startvec_randstate=None,
           max_num_sweeps=5, eigs_opts=None, minimize_sites=1):
    """Iterative search for smallest eigenvalue and eigenvector of an MPO.

    Algorithm: [Sch11, Sec. 6.3]

    Possible TODOs:

    - compute the overlap between 'eigvec' from successive iterations
      to check whether we have converged

    - compute var(H) = <psi| H^2 |psi> - (<psi| H |psi>)^2 every n-th
      iteration to check whether we have converged (this criterion is
      better but more expensive to compute)

    - increase the bond dimension of 'eigvec' if var(H) remains above
      a given threshold

    - for multi-site updates, track the error in the SVD truncation
      (see comment there why)

    - return these details for tracking errors in larger computations

    :param MPArray mpo: A matrix product operator (MPA with two physical legs)

    :param startvec_bonddim: Bond dimension of random start vector if
        no start vector is given. Use the bond dimension of the MPA if
        None.

    :param startvec: Start vector; generate a random start vector if
        None.

    :param startvec_randstate: numpy.random.RandomState instance or None

    :param max_num_sweeps: Maximum number of sweeps to do. Currently,
        always do that many sweeps.

    :param eigs_opts: kwargs for scipy.sparse.linalg.eigs(). You
        should always set k=1.

    :param int minimize_sites: Minimize eigenvalue on that many sites.

    :returns: mineigval, mineigval_eigvec_mpa

    We minimize the eigenvalue by obtaining the minimal eigenvalue of
    an operator supported on 'minimize_sites' many sites. For
    minimize_sites=1, this is called "variational MPS ground state
    search" or "single-site DMRG" [Sch11, Sec. 6.3, p. 69]. For
    minimize_sites>1, this is called "multi-site DMRG".

    Comments on the implementation, for minimize_sites=1:

    References are to the arXiv version of [Sch11] assuming we replace
    zero-based with one-based indices there.

    leftvecs[i] is L_{i-1}  \
    rightvecs[i] is R_{i}   |  See Fig. 38 and Eq. (191) on p. 62.
    mpo[i] is W_{i}         /
    eigvec[i] is M_{i}         This is just the MPS matrix.

    Psi^A_{i-1} and Psi^B_{i} are identity matrices because of
    normalization. (See Fig. 42 on p. 67 and the text; see also
    Figs. 14 and 15 and pages 28 and 29.)

    """
    nr_sites = len(mpo)
    eigvec = startvec
    if eigvec is None:
        if startvec_randstate is None:
            startvec_randstate = np.random
        pdims = max(dim[0] for dim in mpo.pdims)
        if startvec_bonddim is None:
            startvec_bonddim = max(mpo.bdims)
        eigvec = mpnum.factory.random_mpa(nr_sites, pdims, startvec_bonddim,
                                          randstate=startvec_randstate)
        eigvec /= mp.norm(eigvec)
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
    leftvecs = [np.array(1, ndmin=3)] + [None] * (nr_sites - minimize_sites)
    rightvecs = [None] * (nr_sites - minimize_sites) + [np.array(1, ndmin=3)]
    eigvec.normalize(right=1)
    for pos in range(nr_sites - minimize_sites - 1, -1, -1):
        rightvecs[pos] = _mineig_rightvec_add(
            rightvecs[pos + 1], mpo[pos + minimize_sites], eigvec[pos + minimize_sites])

    for num_sweep in range(max_num_sweeps):

        # Sweep from left to right
        for pos in range(nr_sites - minimize_sites + 1):
            if pos == 0 and num_sweep > 0:
                # Don't do first site again if we are not in the first
                # sweep.
                continue
            if pos > 0:
                eigvec.normalize(left=pos)
                rightvecs[pos - 1] = None
                leftvecs[pos] = _mineig_leftvec_add(
                    leftvecs[pos - 1], mpo[pos - 1], eigvec[pos - 1])
            pos_end = pos + minimize_sites
            eigval, eigvec_lten = _mineig_minimize_locally(
                leftvecs[pos], mpo[pos:pos_end], rightvecs[pos],
                eigvec[pos:pos_end], eigs_opts)
            eigvec[pos:pos_end] = eigvec_lten

        # Sweep from right to left (don't do last site again)
        for pos in range(nr_sites - minimize_sites - 1, -1, -1):
            pos_end = pos + minimize_sites
            if pos < nr_sites - minimize_sites:
                # We always do this, because we don't do the last site again.
                eigvec.normalize(right=pos + minimize_sites)
                leftvecs[pos + 1] = None
                rightvecs[pos] = _mineig_rightvec_add(
                    rightvecs[pos + 1], mpo[pos_end], eigvec[pos_end])
            eigval, eigvec_lten = _mineig_minimize_locally(
                leftvecs[pos], mpo[pos:pos_end], rightvecs[pos],
                eigvec[pos:pos_end], eigs_opts)
            eigvec[pos:pos_end] = eigvec_lten

    return eigval, eigvec
