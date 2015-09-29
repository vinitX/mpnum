# encoding: utf-8


r"""Matrix Product State (MPS) and Operator (MPO) functions

Definitions
-----------

We consider a linear chain of :math:`n` sites with associated Hilbert
spaces \mathcal H_k = \C^{d_k}, :math:`d_k`, :math:`k \in [1..n] :=
\{1, 2, \ldots, n\}`. The set of linear operators :math:`\mathcal H_k
\to \mathcal H_k` is denoted by :math:`\mathcal B_k`. We write
:math:`\mathcal H = \mathcal H_1 \otimes \cdots \otimes \mathcal H_n`
and the same for :math:`\mathcal B`.

We use the following three representations:

* Matrix product state (MPS): Vector :math:`\lvert \psi \rangle \in
  \mathcal H`

* Matrix product operator (MPO): Operator :math:`M \in \mathcal B`

* Locally purified matrix product state (PMPS): Positive semidefinite
  operator :math:`\rho \in \mathcal B`

All objects are represented by :math:`n` local tensors.

Matrix product state (MPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent a vector :math:`\lvert \psi \rangle \in \mathcal H` as

.. math::

   \langle i_1 \ldots i_n \vert \psi \rangle 
   = A^{(1)}_{i_1} \cdots A^{(n)}_{i_n}, 
   \quad A^{(k)}_{i_k} \in \mathbb C^{D_{k-1} \times D_k}, 
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,r} =
(A^{(k)}_i)_{l,r}`.

The vector :math:`\lvert \psi \rangle` can be a state, with the
density matrix given by :math:`\rho = \lvert \psi \rangle \langle \psi
\rvert \in \mathcal B`.

Matrix product operator (MPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent an operator :math:`M \in \mathcal B` as

.. math::

  \langle i_1 \ldots i_n \vert M \vert j_1 \ldots j_n \rangle
  = A^{(1)}_{i_1 j_1} \cdots A^{(n)}_{i_n j_n},
  \quad A^{(k)}_{i_k j_k} \in \mathbb C^{D_{k-1} \times D_k}, 
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,j,r} = (A^{(k)}_{i
j})_{l,r}`.

This representation can be used to represent a mixed state :math:`\rho
= M`, but it is not limited to positive semidefinite :math:`M`.

Locally purified matrix product state (PMPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent a positive semidefinite operator :math:`\rho \in \mathcal B`
as follows: Let :math:`\mathcal H_k' = \mathbb C^{d'_k}` with suitable
:math:`d'_k` and :math:`\mathcal P = \mathcal H_1 \otimes \mathcal
H'_1 \otimes \cdots \otimes \mathcal H_n \otimes \mathcal H'_n`. Find
:math:`\vert \Phi \rangle \in \mathcal P` such that

.. math::

   \rho = \operatorname{tr}_{\mathcal H'_1, \ldots, \mathcal H'_n}
   (\lvert \Phi \rangle \langle \Phi \rvert)

and represent :math:`\lvert \Phi \rangle` as

.. math::

   \langle i_1 i'_1 \ldots i_n i'_n \vert \Phi \rangle
   = A^{(1)}_{i_1 i'_1} \cdots A^{(n)}_{i_n i'_n},
   \quad A^{(k)}_{i_k j_k} \in \mathbb C^{D_{k-1} \times D_k}, 
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,i',r} = (A^{(k)}_{i
i'})_{l,r}`.

The ancillary dimensions :math:`d'_i` are not determined by the
:math:`d_i` but depend on the state. E.g. if :math:`\rho` is pure, one
can set all :math:`d_i = 1`.


.. todo:: Add references.

.. todo:: Are derived classes MPO/MPS/PMPS of any help?

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_array_equal

import mpnum.mparray as mp
from mpnum._tools import matdot
from six.moves import range


def reductions_mpo(mpa, width=None, startsites=None, stopsites=None):
    """Take an MPO and iterate over partial traces of the MPO.

    The support of the i-th result is :code:`range(startsites[i],
    stopsites[i])`.

    :param mpnum.mparray.MPArray mpa: An MPO

    :param startsites: Defaults to :code:`range(len(mpa) - width +
        1)`.

    :param stopsites: Defaults to :code:`[ start + width for start in
        startsites ]`. If specified, we require `startsites` to be
        given and `width` to be None.

    :param width: Number of sites in support of the results. Default
        `None`. Must be specified if one or both of `startsites` and
        `stopsites` are not given.

    :returns: Iterator over reduced_state_as_mpo

    """
    if stopsites is None:
        assert width is not None
        if startsites is None:
            startsites = range(len(mpa) - width + 1)
        stopsites = (start + width for start in startsites)
    else:
        assert width is None
        assert startsites is not None

    assert_array_equal(mpa.plegs, 2)
    rem_left = {0: np.array(1, ndmin=2)}
    rem_right = rem_left.copy()

    def get_remainder(rem_cache, num_sites, end):
        """Obtain the vectors resulting from tracing over
        the left or right end of a Matrix Product Operator.

        :param rem_cache: Save remainder terms with smaller num_sites here
        :param num_sites: Number of sites from left or right that have been
            traced over.
        :param end: +1 or -1 for tracing over the left or right end
        """
        try:
            return rem_cache[num_sites]
        except KeyError:
            rem = get_remainder(rem_cache, num_sites - 1, end)
            last_pos = num_sites - 1 if end == 1 else -num_sites
            add = np.trace(mpa[last_pos], axis1=1, axis2=2)
            if end == -1:
                rem, add = add, rem

            rem_cache[num_sites] = matdot(rem, add)
            return rem_cache[num_sites]

    num_sites = len(mpa)
    for start, stop in zip(startsites, stopsites):
        # FIXME we could avoid taking copies here, but then in-place
        # multiplication would have side effects. We could make the
        # affected arrays read-only to turn unnoticed side effects into
        # errors.
        # Is there something like a "lazy copy" or "copy-on-write"-copy?
        # I believe not.
        ltens = [lten.copy() for lten in mpa[start:stop]]
        rem = get_remainder(rem_left, start, 1)
        ltens[0] = matdot(rem, ltens[0])
        rem = get_remainder(rem_right, num_sites - stop, -1)
        ltens[-1] = matdot(ltens[-1], rem)
        yield mp.MPArray(ltens)


def reductions_pmps(pmps, width, startsites=None):
    """Take a local purification MPS and perform partial trace over the
    complement the sites startsites[i], ..., startsites[i] + width.

    Local purification pmps of the reduced states are obtained by
    normalizing suitably and combining the bond and ancilla indices at
    the edge into a larger ancilla dimension.

    :param MPArray mpa: An MPA with two physical legs (system and ancilla)
    :param width: number of sites in support of the results
    :param startsites: Iterator yielding the index of the leftmost sites of the
        supports of the results (default all possible reductions in ascending
        order)
    :returns: Iterator over reduced_state_as_pmps, same order as 'startsites'

    """
    if startsites is None:
        startsites = range(len(pmps) - width + 1)

    for site in startsites:
        pmps.normalize(left=site, right=site + width)

        # leftmost site
        lten = pmps[site]
        left_bd, system, ancilla, right_bd = lten.shape
        newshape = (1, system, left_bd * ancilla, right_bd)
        ltens = [lten.swapaxes(0, 1).copy().reshape(newshape)]

        # central ones
        ltens += (lten.copy() for lten in pmps[site + 1:site + width - 1])

        # rightmost site
        lten = pmps[site + width - 1]
        left_bd, system, ancilla, right_bd = lten.shape
        newshape = (left_bd, system, ancilla * right_bd, 1)
        ltens += [lten.copy().reshape(newshape)]

        reduced_mps = mp.MPArray(ltens)
        yield reduced_mps


def reductions_mps(mps, width, startsites=None):
    """PMPS reduced states of an MPS

    Convert `mps` to a PMPS and apply :func:`reductions_pmps()`.

    :param mps: MPS, will be converted to a PMPS with
        :func:`mps_to_pmps()`.
    :param width: See :func:`reductions_pmps()`
    :param startsites: See :func:`reductions_pmps()`
    :returns: Iterator over reduced_state_as_pmps

    """
    pmps = mps_to_pmps(mps)
    return reductions_pmps(pmps, width, startsites)


def reductions_mps_as_mpo(mps, width, startsites=None):
    """MPO reduced states of an MPS

    Convert the output from :func:`reductions_mps` to MPOs with
    :func:`pmps_to_mpo`.

    :param mps: MPS, will be converted to a PMPS with
        :func:`mps_to_pmps()`.
    :param width: See :func:`reductions_pmps()`
    :param startsites: See :func:`reductions_pmps()`
    :returns: Iterator over reduced_state_as_mpo

    """
    return map(pmps_to_mpo, reductions_mps(mps, width, startsites))


def pmps_to_mpo(pmps):
    """Convert a local purification MPS to a mixed state MPO.

    A mixed state on n sites is represented in local purification MPS
    form by a MPA with n sites and two physical legs per site. The
    first physical leg is a 'system' site, while the second physical
    leg is an 'ancilla' site.

    :param MPArray pmps: An MPA with two physical legs (system and ancilla)
    :returns: An MPO (density matrix as MPA with two physical legs)

    """
    return mp.dot(pmps, pmps.adj())


def mps_to_pmps(mps):
    """Convert a pure MPS into a local purification MPS mixed state.

    The ancilla legs will have dimension one, not increasing the
    memory required for the MPS.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPA with two physical legs (system and ancilla)

    """
    assert_array_equal(mps.plegs, 1)
    ltens = (lten.reshape(lten.shape[0:2] + (1, lten.shape[2])) for lten in mps)
    return mp.MPArray(ltens)


def mps_to_mpo(mps):
    """Convert a pure MPS to a mixed state MPO.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPO (density matrix as MPA with two physical legs)
    """
    return pmps_to_mpo(mps_to_pmps(mps))
