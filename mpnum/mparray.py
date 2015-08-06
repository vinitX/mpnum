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

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.linalg import qr, svd
from numpy.testing import assert_array_equal
from scipy.sparse.linalg import eigs

import mpnum
from mpnum._tools import matdot, norm_2
from six.moves import range, zip


def _extract_factors(tens, plegs):
    """Extract iteratively the leftmost MPO tensor with given number of
    legs by a qr-decomposition

    :param np.ndarray tens: Full tensor to be factorized
    :param int plegs: Number of physical legs per site
    :returns: List of local tensors with given number of legs yielding a
        factorization of tens
    """
    if tens.ndim == plegs + 1:
        return [tens.reshape(tens.shape + (1,))]
    elif tens.ndim < plegs + 1:
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
        self._lnormalized = min(self._lnormalized, index)
        self._rnormalized = max(self._rnormalized, index + 1)
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
    def from_array(cls, array, plegs):
        """Computes the (exact) representation of `array` as MPA with open
        boundary conditions, i.e. bond dimension 1 at the boundary. This
        is done by factoring the off the left and the "physical" legs from
        the rest of the tensor by a QR decomposition and working its way
        through the tensor from the left. This yields a left-canonical
        representation of `array`. [Sch11, Sec. 4.3.1]

        The result is a chain of local tensors with `plegs` physical legs at
        each location and has array.ndim // plegs number of sites.

        :param np.ndarray array: Array representation with global structure
            array[(i1), ..., (iN)], i.e. the legs which are factorized into
            the same factor are already adjacent. (For me details see
            :func:`_tools.global_to_local`)
        :param int plegs: Number of physical legs per site

        """
        assert array.ndim % plegs == 0, \
            "plegs invalid: {} is not multiple of {}".format(array.ndim, plegs)
        ltens = _extract_factors(array[None], plegs=plegs)
        mpa = cls(ltens, _lnormalized=len(ltens) - 1)
        return mpa

    @classmethod
    def from_kron(cls, factors):
        """Returns the (exact) representation of an n-fold  Kronecker (tensor)
        product as MPA with bond dimensions 1 and n sites.

        :param factors: A list of arrays with arbitrary number of physical legs
        :returns: The kronecker product of the factors as MPA
        """
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

    def compress(self, method='svd', **kwargs):
        """Compresses the MPA to a fixed maximal bond dimension in place

        :param method: Which implemention should be used for compression
            'svd': Compression based on SVD [Sch11, Sec. 4.5.1]
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
        :returns: Overlap <M|M'> of the original M and its compression M'

        """
        if method == 'svd':
            ln, rn = self.normal_form
            default_direction = 'left' if len(self) - rn > ln else 'right'
            direction = kwargs.pop('direction', default_direction)
            max_bdim = kwargs.get('max_bdim', max(self.bdims))
            relerr = kwargs.get('relerr', 0.0)

            if direction == 'right':
                self.normalize(right=1)
                return self._compress_svd_r(max_bdim, relerr)
            elif direction == 'left':
                self.normalize(left=len(self) - 1)
                return self._compress_svd_l(max_bdim, relerr)
        else:
            raise ValueError("{} is not a valid method.".format(method))

    def _compress_svd_r(self, max_bdim, relerr):
        """Compresses the MPA in place from left to right using SVD;
        yields a left-cannonical state

        See :func:`MPArray.compress` for parameters
        """
        assert self.normal_form == (0, 1)
        assert (0. <= relerr) and (relerr <= 1.)
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
        assert (0. <= relerr) and (relerr <= 1.)
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


def norm(mpa):
    """Computes the norm (Hilbert space norm for MPS, Frobenius norm for MPO)
    of the matrix product operator. In contrast to `mparray.inner`, this can
    take advantage of the normalization

    :param mpa: MPArray
    :returns: l2-norm of that array

    """
    # FIXME Take advantage of normalization
    return np.sqrt(inner(mpa, mpa))


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


class named_ndarray(object):

    """Associate names to the axes of a ndarray.

    :property axisnames: The names of the axes.

    All methods which return arrays return named_ndarray instances.

    :method axispos(axisname): Return the position of the named axis
    :method rename(translate): Rename axes
    :method conj(): Return the complex conjugate array
    :method to_array(name_order): Return a ndarray with axis order
        specified by name_order.
    :method tensordot(other, axes): numpy.tensordot() with axis names
        instead of axis indices

    """

    def __init__(self, array, axisnames):
        """
        :param numpy.ndarray array: A numpy.ndarray instance
        :param axisnames: A iterable with a name for each axis
        """
        assert(len(array.shape) == len(axisnames)), \
            'number of names does not match number of dimensions'
        assert len(axisnames) == len(set(axisnames)), \
            'axisnames contains duplicates: {}'.format(axisnames)
        self._array = array
        self._axisnames = tuple(axisnames)

    def axispos(self, axisname):
        """Return the position of an axis.
        """
        return self._axisnames.index(axisname)

    def rename(self, translate):
        """Rename axes.

        An error will be raised if the resulting list of names
        contains duplicates.

        :param translate: List of (old_name, new_name) axis name pairs.

        """
        new_names = list(self._axisnames)
        for oldname, newname in translate:
            new_names[self.axispos(oldname)] = newname
        return named_ndarray(self._array, new_names)

    def conj(self):
        """Complex conjugate as named_ndarray.
        """
        return named_ndarray(self._array.conj(), self._axisnames)

    def to_array(self, name_order):
        """Convert to a normal ndarray with given axes ordering.

        :param name_order: Order of axes in the array
        """
        name_pos = [self.axispos(name) for name in name_order]
        array = self._array.transpose(name_pos)
        return array
        
    def tensordot(self, other, axes):
        """Compute tensor dot product along named axes.

        An error will be raised if the remaining axes of self and
        other contain duplicate names.

        :param other: Another named_ndarray instance
        :param axes: List of axis name pairs (self_name, other_name)
            to be contracted
        :returns: Result as named_ndarray
        """
        axes_self = [names[0] for names in axes]
        axes_other = [names[1] for names in axes]
        axespos_self = [self.axispos(name) for name in axes_self]
        axespos_other = [other.axispos(name) for name in axes_other]
        new_names = [name for name in self._axisnames if name not in axes_self]
        new_names += (name for name in other._axisnames if name not in axes_other)
        array = np.tensordot(self._array, other._array, (axespos_self, axespos_other))
        return named_ndarray(array, new_names)

    @property
    def axisnames(self):
        """The names of the array"""
        return _axisnames


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
        eigvec /= norm(eigvec)
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
