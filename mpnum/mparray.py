#!/usr/bin/env python
# encoding: utf-8
"""Module containing routines for dealing with general matrix product arrays.

References:
    [Sch11] U. Schollwöck, The density-matrix renormalization group in the age
        of matrix product states

"""
# FIXME Possible Optimization:
#   - replace integer-for loops with iterataor (not obviously possible
#   everwhere)
#   - replace internal structure as list of arrays with lazy generator of
#   arrays (might not be possible, since we often iterate both ways!)
#   - more in place operations for addition, subtraction, multiplication
# FIXME single site MPAs

from __future__ import absolute_import, division, print_function

import itertools as it
import numpy as np
from numpy.linalg import qr, svd
from numpy.testing import assert_array_equal

from mpnum._tools import matdot
from six.moves import range, zip


def _extract_factors(tens, plegs):
    """Extract iteratively the leftmost MPO tensor with given number of
    legs by a qr-decomposition

    :param np.ndarray tens: Full tensor to be factorized
    :param int plegs: Number of physical legs per site
    :returns: List of local tensors with given number of legs yielding a
        factorization of tens
    """
    if tens.ndim == plegs + 2:
        return [tens]
    elif tens.ndim < plegs + 2:
        raise AssertionError("Number of remaining legs insufficient.")
    else:
        unitary, rest = qr(tens.reshape((np.prod(tens.shape[:plegs + 1]),
                                         np.prod(tens.shape[plegs + 1:]))))

        unitary = unitary.reshape(tens.shape[:plegs + 1] + rest.shape[:1])
        rest = rest.reshape(rest.shape[:1] + tens.shape[plegs + 1:])

        return [unitary] + _extract_factors(rest, plegs)


class MPArray(object):
    """Efficient representation of a general N-partite array A in matrix
    product form with open boundary conditions:

            A^((i1),...,(iN)) = prod_k A^[k]_(ik)   (*)

    where the A^[k] are local tensors (with N legs). The matrix products in
    (*) are taken with respect to the left and right leg and the multi-
    index (ik) corresponds to the physical legs. Open boundary conditions
    imply that shape(A[0])[0] == shape(A[-1])[-1] == 1.

    By convention, the 0th and last dimension of the local tensors are reserved
    for the auxillary legs.
    """

    def __init__(self, ltens, **kwargs):
        """
        :param list ltens: List of local tensors for the MPA. In order to be
            valid the elements of `tens` need to be N-dimensional arrays
            with N > 1 and need to fullfill

                    shape(tens[i])[-1] == shape(tens[i])[0].
        :param **kwargs: Additional paramters to set protected variables, not
            for use by user

        """
        self._ltens = list(ltens)
        for i, (ten, nten) in enumerate(zip(self._ltens[:-1], self._ltens[1:])):
            if ten.shape[-1] != nten.shape[0]:
                raise ValueError("Shape mismatch on {}: {} != {}"
                                 .format(i, ten.shape[-1], nten.shape[0]))

        # Elements _ltens[m] with m < self._lnorm are in left-cannon. form
        self._lnormalized = kwargs.get('_lnormalized', None)
        # Elements _ltens[n] with n >= self._rnorm are in right-cannon. form
        self._rnormalized = kwargs.get('_rnormalized', None)

    def copy(self):
        """Makes a deep copy of the MPA"""
        result = type(self)([ltens.copy() for ltens in self._ltens],
                            _lnormalized=self._lnormalized,
                            _rnormalized=self._rnormalized)
        return result

    def __len__(self):
        return len(self._ltens)

    # FIXME Can we return immutable view into array without having to set the
    #   WRITEABLE flag for the local copy?
    def __iter__(self):
        """Use only for read-only access! Do not change arrays in place!"""
        return iter(self._ltens)

    def __getitem__(self, index):
        """Use only for read-only access! Do not change arrays in place!"""
        return self._ltens[index]

    def __setitem__(self, index, value):
        """Update a local tensor and keep track of normalization."""
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
        else:
            start = index
            stop = index + 1
        self._lnormalized = min(self._lnormalized, start)
        self._rnormalized = max(self._rnormalized, stop)
        self._ltens[index] = value

    @property
    def dims(self):
        """Tuple of shapes for the local tensors"""
        return tuple(m.shape for m in self._ltens)

    @property
    def bdims(self):
        """Tuple of bond dimensions"""
        return tuple(m.shape[0] for m in self._ltens[1:])

    @property
    def pdims(self):
        """Tuple of physical dimensions"""
        return tuple((m.shape[1:-1]) for m in self._ltens)

    @property
    def legs(self):
        """Tuple of total number of legs per site"""
        return tuple(lten.ndim for lten in self._ltens)

    @property
    def plegs(self):
        """Tuple of number of physical legs per site"""
        return tuple(lten.ndim - 2 for lten in self._ltens)

    @property
    def normal_form(self):
        """Tensors which are currently in left/right-canonical form."""
        return self._lnormalized or 0, self._rnormalized or len(self)

    @classmethod
    def from_array(cls, array, plegs=None, has_bond=False):
        """Computes the (exact) representation of `array` as MPA with open
        boundary conditions, i.e. bond dimension 1 at the boundary. This
        is done by factoring the off the left and the "physical" legs from
        the rest of the tensor by a QR decomposition and working its way
        through the tensor from the left. This yields a left-canonical
        representation of `array`. [Sch11, Sec. 4.3.1]

        The result is a chain of local tensors with `plegs` physical legs at
        each location and has array.ndim // plegs number of sites.

        has_bond = True allows to treat a part of the linear chain of
        an MPA as MPA as well. The bond dimension on the left and
        right can be different from one and different from each other
        in that case.  This is useful to apply SVD compression only to
        part of an MPA. It is used in
        linalg._mineig_minimize_locally().

        :param np.ndarray array: Array representation with global structure
            array[(i1), ..., (iN)], i.e. the legs which are factorized into
            the same factor are already adjacent. (For me details see
            :func:`_tools.global_to_local`)
        :param int plegs: Number of physical legs per site (default array.ndim)
        :param bool has_bond: True if array already has indices for
            the left and right bond

        """
        plegs = plegs if plegs is not None else array.ndim
        assert array.ndim % plegs == 0, \
            "plegs invalid: {} is not multiple of {}".format(array.ndim, plegs)
        if not has_bond:
            array = array[None, ..., None]
        ltens = _extract_factors(array, plegs=plegs)
        mpa = cls(ltens, _lnormalized=len(ltens) - 1)
        return mpa

    @classmethod
    def from_kron(cls, factors):
        """Returns the (exact) representation of an n-fold  Kronecker (tensor)
        product as MPA with bond dimensions 1 and n sites.

        :param factors: A list of arrays with arbitrary number of physical legs
        :returns: The kronecker product of the factors as MPA
        """
        # FIXME Do we still need this or shall we prefer mp.outer?
        return cls(a[None, ..., None] for a in factors)

    def to_array(self):
        """Returns the full array representation of the MPA
        :returns: Full matrix A as array of shape [(i1),...,(iN)]

        WARNING: This can be slow for large MPAs!
        """
        return _ltens_to_array(iter(self))

    ##########################
    #  Algebraic operations  #
    ##########################
    def T(self):
        """Transpose of the physical legs"""
        return type(self)([_local_transpose(tens) for tens in self._ltens])

    def adj(self):
        """Hermitian adjoint"""
        return type(self)([_local_transpose(tens).conjugate()
                           for tens in self._ltens])

    def C(self):
        """Complex conjugate"""
        return type(self)(np.conjugate(self._ltens))

    def __add__(self, summand):
        assert len(self) == len(summand), \
            "Length is not equal: {} != {}".format(len(self), len(summand))

        ltens = [np.concatenate((self[0], summand[0]), axis=-1)]
        ltens += [_local_add(l, r) for l, r in zip(self[1:-1], summand[1:-1])]
        ltens += [np.concatenate((self[-1], summand[-1]), axis=0)]
        return MPArray(ltens)

    def __sub__(self, subtr):
        return self + (-1) * subtr

    # TODO These could be made more stable by rescaling all non-normalized tens
    def __mul__(self, fact):
        if np.isscalar(fact):
            lnormal, rnormal = self.normal_form
            ltens = self._ltens[:lnormal] + [fact * self._ltens[lnormal]] + \
                self._ltens[lnormal + 1:]
            return type(self)(ltens, _lnormalized=lnormal,
                              _rnormalized=rnormal)

        raise NotImplementedError("Multiplication by non-scalar not supported")

    def __imul__(self, fact):
        if np.isscalar(fact):
            lnormal, _ = self.normal_form
            self._ltens[lnormal] *= fact
            return self

        raise NotImplementedError("Multiplication by non-scalar not supported")

    def __rmul__(self, fact):
        return self.__mul__(fact)

    def __truediv__(self, divisor):
        if np.isscalar(divisor):
            return self.__mul__(1 / divisor)
        raise NotImplementedError("Division by non-scalar not supported")

    def __itruediv__(self, divisor):
        if np.isscalar(divisor):
            return self.__imul__(1 / divisor)
        raise NotImplementedError("Division by non-scalar not supported")

    ################################
    #  Shape changes, conversions  #
    ################################

    def pleg_reshape(self, newshapes):
        """Apply reshape to physical legs.

        Use self.pdims to obtain the shapes of the physical legs.

        :param newshapes: A single new shape or a list of new shapes
        :returns: MPArray instance with different shape of physical legs
        """
        newshapes = tuple(newshapes)
        if not hasattr(newshapes[0], '__iter__'):
            newshapes = it.repeat(newshapes, times=len(self))
        ltens = (
            lten.reshape((lten.shape[0],) + tuple(newshape) + (lten.shape[-1],))
            for lten, newshape in zip(self._ltens, newshapes))
        mpa = MPArray(ltens, _lnormalized=self._lnormalized,
                      _rnormalized=self._rnormalized)
        return mpa

    def group_sites(self, sites_per_group):
        """Group several MPA sites into one site.

        The resulting MPA has length len(self) // sites_per_group and
        sites_per_group * self.plegs[i] physical legs on site i. The
        physical legs on each sites are in local form.

        :param int sites_per_group: Number of sites to be grouped into one
        :returns: An MPA with sites_per_group fewer sites and more plegs

        """
        assert (len(self) % sites_per_group) == 0, \
            'length not a multiple of sites_per_group'
        if sites_per_group == 1:
            return self
        ltens = []
        for i in range(len(self) // sites_per_group):
            ten = self[i * sites_per_group]
            for j in range(1, sites_per_group):
                ten = matdot(ten, self[i * sites_per_group + j])
            ltens.append(ten)
        return MPArray(ltens)

    def split_sites(self, sites_per_group):
        """Split MPA sites into several sites.

        The resulting MPA has length len(self) * sites_per_group and
        self.plegs[i] // sites_per_group physical legs on site i. The
        physical legs on before splitting must be in local form.

        :param int sites_per_group: Split each site in that many sites
        :returns: An mpa with sites_per_group more sites and fewer plegs

        """
        ltens = []
        for i in range(len(self)):
            plegs = self.plegs[i]
            assert (plegs % sites_per_group) == 0, \
                'plegs not a multiple of sites_per_group'
            ltens += _extract_factors(self[i], plegs//sites_per_group)
        return MPArray(ltens)

    ################################
    #  Normalizaton & Compression  #
    ################################
    # FIXME Maybe we should extract site-normalization logic to seperate funcs
    def normalize(self, **kwargs):
        """Brings the MPA to canonnical form in place. Note that we do not
        support full left- or right-normalization. The right- (left- resp.)
        most local tensor is not normalized since this can be done by
        simply calculating its norm (instead of using SVD)

        [Sch11, Sec. 4.4]

        Possible combinations:
            normalize() = normalize(left=len(self) - 1)
                -> full left-normalization
            normalize(left=m) for 0 <= m < len(self)
                -> self[0],..., self[m-1] are left-normalized
            normalize(right=n) for 0 < n <= len(self)
                -> self[n],..., self[-1] are right-normalized
            normalize(left=m, right=n) valid for m < n
                -> self[0],...,self[m-1] are left normalized and
                   self[n],...,self[-1] are right-normalized

        """
        if ('left' not in kwargs) and ('right' not in kwargs):
            self._lnormalize(len(self) - 1)
            return

        lnormalize = kwargs.get('left', 0)
        rnormalize = kwargs.get('right', len(self))

        assert lnormalize < rnormalize, \
            "Normalization {}:{} invalid".format(lnormalize, rnormalize)
        current_normalization = self.normal_form
        if current_normalization[0] < lnormalize:
            self._lnormalize(lnormalize)
        if current_normalization[1] > rnormalize:
            self._rnormalize(rnormalize)

    def _lnormalize(self, to_site):
        """Left-normalizes all local tensors _ltens[to_site:] in place

        :param to_site: Index of the site up to which normalization is to be
            performed

        """
        assert to_site < len(self), "Cannot left-normalize rightmost site: {} >= {}" \
            .format(to_site, len(self))

        lnormal, rnormal = self.normal_form
        for site in range(lnormal, to_site):
            ltens = self._ltens[site]
            matshape = (np.prod(ltens.shape[:-1]), ltens.shape[-1])
            q, r = qr(ltens.reshape(matshape))
            # if ltens.shape[-1] > prod(ltens.phys_shape) --> trivial comp.
            # can be accounted by adapting bond dimension here
            self._ltens[site] = q.reshape(ltens.shape[:-1] + (-1, ))
            self._ltens[site + 1] = matdot(r, self._ltens[site + 1])

        self._lnormalized = to_site
        self._rnormalized = max(to_site + 1, rnormal)

    def _rnormalize(self, to_site):
        """Right-normalizes all local tensors _ltens[:to_site] in place

        :param to_site: Index of the site up to which normalization is to be
            performed

        """
        assert to_site > 0, "Cannot right-normalize leftmost to_site: {} >= {}" \
            .format(to_site, len(self))

        lnormal, rnormal = self.normal_form
        for site in range(rnormal - 1, to_site - 1, -1):
            ltens = self._ltens[site]
            matshape = (ltens.shape[0], np.prod(ltens.shape[1:]))
            q, r = qr(ltens.reshape(matshape).T)
            # if ltens.shape[-1] > prod(ltens.phys_shape) --> trivial comp.
            # can be accounted by adapting bond dimension here
            self._ltens[site] = q.T.reshape((-1, ) + ltens.shape[1:])
            self._ltens[site - 1] = matdot(self._ltens[site - 1], r.T)

        self._lnormalized = min(to_site - 1, lnormal)
        self._rnormalized = to_site

    def compress(self, method='svd', inplace=True, **kwargs):
        """Compresses the MPA to a fixed maximal bond dimension

        :param method: Which implemention should be used for compression
            'svd': Compression based on SVD [Sch11, Sec. 4.5.1]
            'var': Variational compression [Sch11, Sec. 4.5.2]
        :param inplace: Compress the array in place or return new copy
        :returns: Depends on method and the options passed.

        For method='svd':
        -----------------
        :param max_bdim: Maximal bond dimension for the compressed MPA (default
            max of current bond dimensions, i.e. no compression)
        :param relerr: Maximal allowed error for each truncation step, that is
            the fraction of truncated singular values over their sum (default
            0.0, i.e. no compression)

        If both max_bdim and relerr is passed, the smaller resulting bond
        dimension is used.

        :param direction: In which direction the compression should operate.
            (default: depending on the current normalization, such that the
             number of sites that need to be normalized is smaller)
            'right': Starting on the leftmost site, the compression sweeps
                     to the right yielding a completely left-cannonical MPA
            'left': Starting on rightmost site, the compression sweeps
                    to the left yielding a completely right-cannoncial MPA
        :returns:
            inplace=true: Overlap <M|M'> of the original M and its compr. M'
            inplace=false: Compressed MPA, Overlap <M|M'> of the original M and
                           its compr. M',

        """
        if method == 'svd':
            ln, rn = self.normal_form
            default_direction = 'left' if len(self) - rn > ln else 'right'
            direction = kwargs.pop('direction', default_direction)
            max_bdim = kwargs.get('max_bdim', max(self.bdims))
            relerr = kwargs.get('relerr', 0.0)

            target = self if inplace else self.copy()

            if direction == 'right':
                target.normalize(right=1)
                overlap = target._compress_svd_r(max_bdim, relerr)
            elif direction == 'left':
                self.normalize(left=len(self) - 1)
                overlap = target._compress_svd_l(max_bdim, relerr)

            return overlap if inplace else target, overlap
        else:
            raise ValueError("{} is not a valid method.".format(method))

    def _compress_svd_r(self, max_bdim, relerr):
        """Compresses the MPA in place from left to right using SVD;
        yields a left-cannonical state

        See :func:`MPArray.compress` for parameters
        """
        assert self.normal_form == (0, 1)
        assert max_bdim > 0, "Cannot compress to bdim={}".format(max_bdim)
        assert (0. <= relerr) and (relerr <= 1.), \
            "Relerr={} not allowed".format(relerr)

        for site in range(len(self) - 1):
            ltens = self._ltens[site]
            u, sv, v = svd(ltens.reshape((-1, ltens.shape[-1])))

            svsum = np.cumsum(sv) / np.sum(sv)
            bdim_relerr = np.searchsorted(svsum, 1 - relerr) + 1
            bdim_t = min(ltens.shape[-1], u.shape[1], max_bdim, bdim_relerr)

            newshape = ltens.shape[:-1] + (bdim_t, )
            self._ltens[site] = u[:, :bdim_t].reshape(newshape)
            self._ltens[site + 1] = matdot(sv[:bdim_t, None] * v[:bdim_t, :],
                                           self._ltens[site + 1])

        self._lnormalized = len(self) - 1
        self._rnormalized = len(self)
        return np.sum(np.abs(self._ltens[-1])**2)

    def _compress_svd_l(self, max_bdim, relerr):
        """Compresses the MPA in place from right to left using SVD;
        yields a right-cannonical state

        See :func:`MPArray.compress` for parameters

        """
        assert self.normal_form == (len(self) - 1, len(self))
        assert max_bdim > 0, "Cannot compress to bdim={}".format(max_bdim)
        assert (0. <= relerr) and (relerr <= 1.), \
            "Relerr={} not allowed".format(relerr)

        for site in range(len(self) - 1, 0, -1):
            ltens = self._ltens[site]
            matshape = (ltens.shape[0], -1)
            u, sv, v = svd(ltens.reshape(matshape))

            svsum = np.cumsum(sv) / np.sum(sv)
            bdim_relerr = np.searchsorted(svsum, 1 - relerr) + 1
            bdim_t = min(ltens.shape[0], v.shape[0], max_bdim, bdim_relerr)

            newshape = (bdim_t, ) + ltens.shape[1:]
            self._ltens[site] = v[:bdim_t, :].reshape(newshape)
            self._ltens[site - 1] = matdot(self._ltens[site - 1],
                                           u[:, :bdim_t] * sv[None, :bdim_t])

        self._lnormalized = 0
        self._rnormalized = 1
        return np.sum(np.abs(self._ltens[0])**2)
    # TODO Adaptive/error based compression method


#############################################
#  General functions to deal with MPArrays  #
#############################################
def dot(mpa1, mpa2, axes=(-1, 0)):
    """Compute the matrix product representation of a.b over the given
    (physical) axes. [Sch11, Sec. 4.2]

    :param mpa1, mpa2: Factors as MPArrays
    :param axes: 2-tuple of axes to sum over. Note the difference in
        convention compared to np.tensordot(default: last axis of `mpa1`
        and first axis of `mpa2`)
    :returns: Dot product of the physical arrays

    """
    assert len(mpa1) == len(mpa2), \
        "Length is not equal: {} != {}".format(len(mpa1), len(mpa2))

    # adapt the axes from physical to true legs
    ax_l, ax_r = axes
    ax_l = ax_l + 1 if ax_l >= 0 else ax_l - 1
    ax_r = ax_r + 1 if ax_r >= 0 else ax_r - 1

    ltens = [_local_dot(l, r, (ax_l, ax_r)) for l, r in zip(mpa1, mpa2)]

    return MPArray(ltens)


# NOTE: I think this is a nice example how we could use Python's generator
#       expression to implement lazy evaluation of the matrix product structure
#       which is the whole point of doing this in the first place
def inner(mpa1, mpa2):
    """Compute the inner product <mpa1|mpa2>. Both have to have the same
    physical dimensions. If these represent a MPS, inner(...) corresponds to
    the cannoncial Hilbert space scalar product, if these represent a MPO,
    inner(...) corresponds to the Frobenius scalar product (with Hermitian
    conjugation in the first argument)

    :param mpa1: MPArray with same number of physical legs on each site
    :param mpa2: MPArray with same physical shape as mpa1
    :returns: <mpa1|mpa2>

    """
    assert len(mpa1) == len(mpa2), \
        "Length is not equal: {} != {}".format(len(mpa1), len(mpa2))
    ltens_new = (_local_dot(_local_ravel(l).conj(), _local_ravel(r), axes=(1, 1))
                 for l, r in zip(mpa1, mpa2))
    return _ltens_to_array(ltens_new)


def outer(mpas):
    """Performs the tensor product of MPAs given in *args

    :param mpas: Iterable of MPAs same order as they should appear in the chain
    :returns: MPA of length len(args[0]) + ... + len(args[-1])

    """
    # TODO Make this normalization aware
    # FIXME Is copying here a good idea?
    return MPArray(sum(([ltens.copy() for ltens in mpa] for mpa in mpas), []))


def norm(mpa):
    """Computes the norm (Hilbert space norm for MPS, Frobenius norm for MPO)
    of the matrix product operator. In contrast to `mparray.inner`, this can
    take advantage of the normalization

    :param mpa: MPArray
    :returns: l2-norm of that array

    """
    # FIXME Take advantage of normalization
    return np.sqrt(np.abs(inner(mpa, mpa)))


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
        ltens = [lten.copy() for lten in mpa[startsite : startsite + width]]
        rem = get_remainder(rem_left, startsite, 1)
        ltens[0] = matdot(rem, ltens[0])
        rem = get_remainder(rem_right, num_sites - (startsite + width), -1)
        ltens[-1] = matdot(ltens[-1], rem)
        yield startsite, MPArray(ltens)


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
                  for lten in mps[startsite + 1: startsite + width - 1])
        lten = mps[startsite + width - 1]
        left_bd, system, ancilla, right_bd = lten.shape
        newshape = (left_bd, system, ancilla * right_bd, 1)
        ltens += [lten.copy().reshape(newshape)]
        reduced_mps = MPArray(ltens)
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
    mps_adj = mps.adj()
    # The dot product here contracts the physical indices of two
    # ancilla sites, tracing them out.
    mpo = dot(mps, mps_adj)
    return mpo


def mps_as_local_purification_mps(mps):
    """Convert a pure MPS into a local purification MPS mixed state.

    The ancilla legs will have dimension one, not increasing the
    memory required for the MPS.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPA with two physical legs (system and ancilla)

    """
    ltens = (m.reshape(m.shape[0:2] + (1, m.shape[2])) for m in mps)
    return MPArray(ltens)


def mps_as_mpo(mps):
    """Convert a pure MPS to a mixed state MPO.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPO (density matrix as MPA with two physical legs)
    """
    mps_loc_puri = mps_as_local_purification_mps(mps)
    mpo = local_purification_mps_to_mpo(mps_loc_puri)
    return mpo


############################################################
#  Functions for dealing with local operations on tensors  #
############################################################
def _local_dot(ltens_l, ltens_r, axes):
    """Computes the local tensors of a dot product dot(l, r).

    Besides computing the normal dot product, this function rearranges the
    bond legs in such a way that the result is a valid local tensor again.

    :param ltens_l: Array with ndim > 1
    :param ltens_r: Array with ndim > 1
    :param axes: Axes to compute dot product using the convention of
        np.tensordot. Note that these correspond to the true (and not the
        physical) legs of the local tensors
    :returns: Correct local tensor representation

    """
    # number of contracted legs need to be the same
    clegs_l = len(axes[0]) if hasattr(axes[0], '__len__') else 1
    clegs_r = len(axes[1]) if hasattr(axes[0], '__len__') else 1
    assert clegs_l == clegs_r, \
        "Number of contracted legs differ: {} != {}".format(clegs_l, clegs_r)
    res = np.tensordot(ltens_l, ltens_r, axes=axes)
    # Rearrange the bond-dimension legs
    res = np.rollaxis(res, ltens_l.ndim - clegs_l, 1)
    res = np.rollaxis(res, ltens_l.ndim - clegs_l,
                      ltens_l.ndim + ltens_r.ndim - clegs_l - clegs_r - 1)
    return res.reshape((ltens_l.shape[0] * ltens_r.shape[0], ) +
                       res.shape[2:-2] +
                       (ltens_l.shape[-1] * ltens_r.shape[-1],))


def _local_add(ltens_l, ltens_r):
    """Computes the local tensors of a sum l + r (except for the boundary
    tensors)

    :param ltens_l: Array with ndim > 1
    :param ltens_r: Array with ndim > 1
    :returns: Correct local tensor representation

    """
    assert_array_equal(ltens_l.shape[1:-1], ltens_r.shape[1:-1])

    shape = (ltens_l.shape[0] + ltens_r.shape[0], )
    shape += ltens_l.shape[1:-1]
    shape += (ltens_l.shape[-1] + ltens_r.shape[-1], )
    res = np.zeros(shape, dtype=ltens_l.dtype)

    res[:ltens_l.shape[0], ..., :ltens_l.shape[-1]] = ltens_l
    res[ltens_l.shape[0]:, ..., ltens_l.shape[-1]:] = ltens_r
    return res


def _local_ravel(ltens):
    """Flattens the physical legs of ltens, the bond-legs remain untouched

    :param ltens: numpy.ndarray with ndim > 1
    :returns: Reshaped ltens with shape (ltens.shape[0], *, ltens.shape[-1]),
        where * is determined from the size of ltens

    """
    shape = ltens.shape
    return ltens.reshape((shape[0], -1, shape[-1]))


def _local_transpose(ltens):
    """Transposes the physical legs of the local tensor `ltens`

    :param ltens: Local tensor as numpy.ndarray with ndim >= 2
    :returns: Transpose of ltens except for first and last dimension

    """
    return np.transpose(ltens, axes=[0] + list(range(ltens.ndim - 2, 0, -1)) +
                        [ltens.ndim - 1])


def _ltens_to_array(ltens):
    """Computes the full array representation from an iterator yielding the
    local tensors.

    :param ltens: Iterator over local tensors
    :returns: numpy.ndarray representing the contracted MPA

    """
    res = next(ltens)
    for tens in ltens:
        res = matdot(res, tens)
    return res[0, ..., 0]
