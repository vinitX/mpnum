#!/usr/bin/env python
# encoding: utf-8
"""TODO"""
# FIXME I think all the names are too long

from __future__ import absolute_import, division, print_function

import numpy as np

import mpnum.mparray as mp
from mpnum._tools import matdot


def partialtrace_operator(mpa, startsites, width):
    """Take an MPA with two physical legs per site and perform partial trace
    over the complement the sites startsites[i], ..., startsites[i] + width.

    :param mpa: MPArray with two physical legs (a Matrix Product Operator)
    :param startsites: Iterator yielding the index of the leftmost sites of the
        supports of the results
    :param width: number of sites in support of the results
    :returns: Iterator over (startsite, reduced_mpa)
    """
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
    for startsite in startsites:
        # FIXME we could avoid taking copies here, but then in-place
        # multiplication would have side effects. We could make the
        # affected arrays read-only to turn unnoticed side effects into
        # errors.
        # Is there something like a "lazy copy" or "copy-on-write"-copy?
        # I believe not.
        ltens = [lten.copy() for lten in mpa[startsite:startsite + width]]
        rem = get_remainder(rem_left, startsite, 1)
        ltens[0] = matdot(rem, ltens[0])
        rem = get_remainder(rem_right, num_sites - (startsite + width), -1)
        ltens[-1] = matdot(ltens[-1], rem)
        yield startsite, mp.MPArray(ltens)


def partialtrace_local_purification_mps(mps, startsites, width):
    """Take a local purification MPS and perform partial trace over the
    complement the sites startsites[i], ..., startsites[i] + width.

    Local purification mps of the reduced states are obtained by
    normalizing suitably and combining the bond and ancilla indices at
    the edge into a larger ancilla dimension.

    :param MPArray mpa: An MPA with two physical legs (system and ancilla)
    :param startsites: Iterator yielding the index of the leftmost sites of the
        supports of the results
    :param width: number of sites in support of the results
    :returns: Iterator over (startsite, reduced_locpuri_mps)

    """
    for startsite in startsites:
        mps.normalize(left=startsite, right=startsite + width)
        lten = mps[startsite]
        left_bd, system, ancilla, right_bd = lten.shape
        newshape = (1, system, left_bd * ancilla, right_bd)
        ltens = [lten.swapaxes(0, 1).copy().reshape(newshape)]
        ltens += (lten.copy()
                  for lten in mps[startsite + 1:startsite + width - 1])
        lten = mps[startsite + width - 1]
        left_bd, system, ancilla, right_bd = lten.shape
        newshape = (left_bd, system, ancilla * right_bd, 1)
        ltens += [lten.copy().reshape(newshape)]
        reduced_mps = mp.MPArray(ltens)
        yield startsite, reduced_mps


def local_purification_mps_to_mpo(mps):
    """Convert a local purification MPS to a mixed state MPO.

    A mixed state on n sites is represented in local purification MPS
    form by a MPA with n sites and two physical legs per site. The
    first physical leg is a 'system' site, while the second physical
    leg is an 'ancilla' site.

    :param MPArray mps: An MPA with two physical legs (system and ancilla)
    :returns: An MPO (density matrix as MPA with two physical legs)

    """
    return mp.dot(mps, mps.adj())


def mps_as_local_purification_mps(mps):
    """Convert a pure MPS into a local purification MPS mixed state.

    The ancilla legs will have dimension one, not increasing the
    memory required for the MPS.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPA with two physical legs (system and ancilla)

    """
    ltens = (m.reshape(m.shape[0:2] + (1, m.shape[2])) for m in mps)
    return mp.MPArray(ltens)


def mps_as_mpo(mps):
    """Convert a pure MPS to a mixed state MPO.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPO (density matrix as MPA with two physical legs)
    """
    mps_loc_puri = mps_as_local_purification_mps(mps)
    mpo = local_purification_mps_to_mpo(mps_loc_puri)
    return mpo
