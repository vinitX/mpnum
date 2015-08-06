# encoding: utf-8


import numpy as np

from scipy.sparse.linalg import eigs

import mpnum
import mpnum.factory
import mpnum.mparray as mp


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


def _mineig_local_op(leftvec, mpo_lten, rightvec):
    """Create the operator for local eigenvalue minimization on one site.

    :param leftvec: Left vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param mpo_lten: Local tensor of the MPO
    :param rightvec: Right vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond

    See [Sch11, arXiv version, Fig. 38 on p. 62].  This method
    implements the contractions across the dashed lines in the figure.

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


def _mineig_minimize_locally(leftvec, mpo_lten, rightvec, eigvec_lten):
    """Perform the local eigenvalue minimization on one site on one site.

    Return a new (expectedly smaller) eigenvalue and a new local
    tensor for the MPS eigenvector.

    :param leftvec: Left vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param mpo_lten: Local tensor of the MPO
    :param rightvec: Right vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param eigvec_lten: Local tensor of the MPS eigenvector
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
    eigs_opts = {'k': 1, 'which': 'SR', 'tol': 1e-6}
    op = _mineig_local_op(leftvec, mpo_lten, rightvec)
    eigvals, eigvecs = eigs(op, v0=eigvec_lten.flatten(), **eigs_opts)
    eigval = eigvals[0]
    eigvec_lten = eigvecs[:, 0].reshape(eigvec_lten.shape)
    return eigval, eigvec_lten


def mineig(mpo, startvec=None, startvec_bonddim=None):
    """Iterative search for smallest eigenvalue and eigenvector of an MPO.

    Algorithm: [Sch11, Sec. 6.3]

    :param MPArray mpo: A matrix product operator (MPA with two physical legs)

    :param startvec_bonddim: Bond dimension of random start vector if
        no start vector is given. Use the bond dimension of the MPA if
        None.

    :param startvec: Start vector; generate a random start vector if
        None.

    :returns: mineigval, mineigval_eigvec_mpa

    Comments on the implementation: 

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
        pdims = max(dim[0] for dim in mpo.pdims)
        if startvec_bonddim is None:
            startvec_bonddim = max(mpo.bdims)
        eigvec = mpnum.factory.random_mpa(nr_sites, pdims, startvec_bonddim)
        eigvec /= mp.norm(eigvec)
    eigvec.normalize(right=1)
    leftvecs = [np.array(1, ndmin=3)] + [None] * (nr_sites - 1)
    rightvecs = [None] * (nr_sites - 1) + [np.array(1, ndmin=3)]
    for pos in range(nr_sites - 2, -1, -1):
        rightvecs[pos] = _mineig_rightvec_add(
            rightvecs[pos + 1], mpo[pos + 1], eigvec[pos + 1])

    num_sweeps = 5
    for num_sweep in range(num_sweeps):
        
        # Sweep from left to right
        for pos in range(nr_sites):
            if pos == 0 and num_sweep > 0:
                # Don't do first site again if we are not in the first
                # sweep.
                continue
            if pos > 0:
                eigvec.normalize(left=pos)
                rightvecs[pos - 1] = None
                leftvecs[pos] = _mineig_leftvec_add(
                    leftvecs[pos - 1], mpo[pos - 1], eigvec[pos - 1])
            eigval, eigvec_lten = _mineig_minimize_locally(
                leftvecs[pos], mpo[pos], rightvecs[pos], eigvec[pos])
            eigvec[pos] = eigvec_lten

        # Sweep from right to left (don't do last site again)
        for pos in range(nr_sites - 2, -1, -1):
            if pos < nr_sites - 1:
                eigvec.normalize(right=pos + 1)
                leftvecs[pos + 1] = None
                rightvecs[pos] = _mineig_rightvec_add(
                    rightvecs[pos + 1], mpo[pos + 1], eigvec[pos + 1])
            eigval, eigvec_lten = _mineig_minimize_locally(
                leftvecs[pos], mpo[pos], rightvecs[pos], eigvec[pos])
            eigvec[pos] = eigvec_lten

    return eigval, eigvec

