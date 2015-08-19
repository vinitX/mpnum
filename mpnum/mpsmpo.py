#!/usr/bin/env python
# encoding: utf-8
"""TODO What considered an MPS/MPO/PMPS? What are the conventions?"""
# FIXME I think all the names are too long
# TODO Are derived classes MPO/MPS/PMPS of any help?

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_array_equal

import mpnum.mparray as mp
from mpnum._tools import matdot


def reductions_mpo(mpa, startsites, width):
    """Take an MPA with two physical legs per site and perform partial trace
    over the complement the sites startsites[i], ..., startsites[i] + width.

    :param mpa: MPArray with two physical legs (a Matrix Product Operator)
    :param startsites: Iterator yielding the index of the leftmost sites of the
        supports of the results
    :param width: number of sites in support of the results
    :returns: Iterator over reduced_state_as_mpo, same order as 'startsites'
    """
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
        yield mp.MPArray(ltens)


def reductions_pmps(pmps, startsites, width):
    """Take a local purification MPS and perform partial trace over the
    complement the sites startsites[i], ..., startsites[i] + width.

    Local purification pmps of the reduced states are obtained by
    normalizing suitably and combining the bond and ancilla indices at
    the edge into a larger ancilla dimension.

    :param MPArray mpa: An MPA with two physical legs (system and ancilla)
    :param startsites: Iterator yielding the index of the leftmost sites of the
        supports of the results
    :param width: number of sites in support of the results
    :returns: Iterator over reduced_state_as_pmps, same order as 'startsites'

    """
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


def mps_as_mpo(mps):
    """Convert a pure MPS to a mixed state MPO.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPO (density matrix as MPA with two physical legs)
    """
    return pmps_to_mpo(mps_to_pmps(mps))
