# encoding: utf-8
"""Module containing routines for dealing with general matrix product arrays.

References:

* .. _Sch11:

  [Sch11] Schollwöck, U. (2011). “The density-matrix renormalization
  group in the age of matrix product states”. Ann. Phys. 326(1),
  pp. 96–192. `DOI: 10.1016/j.aop.2010.09.012`_. `arXiv: 1008.3477`_.

  .. _`DOI: 10.1016/j.aop.2010.09.012`:
     http://dx.doi.org/10.1016/j.aop.2010.09.012

  .. _`arXiv: 1008.3477`: http://arxiv.org/abs/1008.3477

"""
# FIXME Possible Optimization:
#   - replace integer-for loops with iterataor (not obviously possible
#   everwhere)
#   - replace internal structure as list of arrays with lazy generator of
#   arrays (might not be possible, since we often iterate both ways!)
#   - more in place operations for addition, subtraction, multiplication
# FIXME single site MPAs
# TODO Replace all occurences of self._ltens with self[...] or similar &
#      benchmark. This will allow easier transition to lazy evaluation of
#      local tensors

from __future__ import absolute_import, division, print_function

import functools as ft
import itertools as it

import numpy as np
from numpy.linalg import qr, svd
from numpy.testing import assert_array_equal

from mpnum._tools import block_diag, matdot
from mpnum._named_ndarray import named_ndarray
from six.moves import range, zip


class MPArray(object):
    r"""Efficient representation of a general N-partite array A in matrix
    product form with open boundary conditions:

    .. math::
       :label: mpa

       A_{i_1, \ldots, i_N} = A^{[1]}_{i_1} \ldots A^{[N]}_{i_N}

    where the :math:`A^{[k]}` are local tensors (with N legs). The
    matrix products in :eq:`mpa` are taken with respect to the left
    and right leg and the multi-index :math:`i_k` corresponds to the
    physical legs. Open boundary conditions imply that :math:`A^{[1]}`
    is 1-by-something and :math:`A^{[N]}` is something-by-1.

    By convention, the 0th and last dimension of the local tensors are reserved
    for the auxillary legs.
    """

    def __init__(self, ltens, **kwargs):
        """
        :param list ltens: List of local tensors for the MPA. In order
            to be valid the elements of `tens` need to be
            :code:`N`-dimensional arrays with :code:`N > 1` and need
            to fullfill::

                shape(tens[i])[-1] == shape(tens[i])[0].


        :param `**kwargs`: Additional paramters to set protected
            variables, not for use by user

        """
        self._ltens = list(ltens)
        for i, (ten, nten) in enumerate(zip(self._ltens[:-1], self._ltens[1:])):
            if ten.shape[-1] != nten.shape[0]:
                raise ValueError("Shape mismatch on {}: {} != {}"
                                 .format(i, ten.shape[-1], nten.shape[0]))

        # Elements _ltens[m] with m < self._lnorm are in left-canon. form
        self._lnormalized = kwargs.get('_lnormalized', None)
        # Elements _ltens[n] with n >= self._rnorm are in right-canon. form
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
        if type(index) == tuple:
            assert len(index) == len(self)
            return MPArray(ltens[:, i, ..., :]
                           for i, ltens in zip(index, self._ltens))
        else:
            # FIXME Maybe this should be moved to another Function
            return self._ltens[index]

    def __setitem__(self, index, value):
        """Update a local tensor and keep track of normalization."""
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
        else:
            start = index
            stop = index + 1
        if self._lnormalized is not None:
            self._lnormalized = min(self._lnormalized, start)
        if self._rnormalized is not None:
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
        representation of `array`. [Sch11_, Sec. 4.3.1]

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
        return cls(ltens, _lnormalized=len(ltens) - 1)

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
        return _ltens_to_array(iter(self))[0, ..., 0]

    def paxis_iter(self, axes=0):
        """Returns an iterator yielding Sub-MPArrays of `self` by iterating
        over the specified physical axes.

        **Example:** If `self` represents a bipartite (i.e. length 2)
        array with 2 physical dimensions on each site A[(k,l), (m,n)],
        self.paxis_iter(0) is equivalent to::

            (A[(k, :), (m, :)] for m in range(...) for k in range(...))

        FIXME: The previous code is not highlighted because
        :code:`A[(k, :)]` is invalid syntax. Example of working
        highlighting::

            (x**2 for x in range(...))

        :param axes: Iterable or int specifiying the physical axes to iterate
            over (default 0 for each site)
        :returns: Iterator over MPArray

        """
        if not hasattr(axes, '__iter__'):
            axes = it.repeat(axes, len(self))

        ltens_iter = it.product(*(iter(np.rollaxis(lten, i + 1))
                                  for i, lten in zip(axes, self._ltens)))
        return (MPArray(ltens) for ltens in ltens_iter)

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

    def conj(self):
        """Complex conjugate"""
        return type(self)(np.conjugate(self._ltens))

    def __add__(self, summand):
        assert len(self) == len(summand), \
            "Length is not equal: {} != {}".format(len(self), len(summand))
        if len(self) == 1:
            # The code below assumes at least two sites.
            return MPArray((self[0] + summand[0],))

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

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

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
    def reshape(self, newshapes):
        """Reshape physical legs in place.

        Use self.pdims to obtain the shapes of the physical legs.

        :param newshapes: A single new shape or a list of new shapes
        :returns: Reshaped MPA

        """
        newshapes = tuple(newshapes)
        if not hasattr(newshapes[0], '__iter__'):
            newshapes = it.repeat(newshapes, times=len(self))

        return MPArray([_local_reshape(lten, newshape)
                       for lten, newshape in zip(self._ltens, newshapes)])

    def ravel(self):
        """Flatten the MPA to an MPS, shortcut for self.reshape((-1,))

        """
        return self.reshape((-1,))

    def group_sites(self, sites_per_group):
        """Group several MPA sites into one site.

        The resulting MPA has length len(self) // sites_per_group and
        sites_per_group * self.plegs[i] physical legs on site i. The
        physical legs on each sites are in local form.

        :param int sites_per_group: Number of sites to be grouped into one
        :returns: An MPA with sites_per_group fewer sites and more plegs

        """
        assert (len(self) % sites_per_group) == 0, \
            'Cannot group: {} not a multiple of {}'.format(len(self), sites_per_group)

        if sites_per_group == 1:
            return self
        ltens = [_ltens_to_array(self._ltens[i:i + sites_per_group])
                 for i in range(0, len(self), sites_per_group)]
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
            ltens += _extract_factors(self[i], plegs // sites_per_group)
        return MPArray(ltens)

    ################################
    #  Normalizaton & Compression  #
    ################################
    def normalize(self, **kwargs):
        """Brings the MPA to canonnical form in place. Note that we do not
        support full left- or right-normalization. The right- (left- resp.)
        most local tensor is not normalized since this can be done by
        simply calculating its norm (instead of using SVD)

        [Sch11_, Sec. 4.4]

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
        current_lnorm, current_rnorm = self.normal_form
        if ('left' not in kwargs) and ('right' not in kwargs):
            if current_lnorm < len(self) - current_rnorm:
                self._rnormalize(1)
            else:
                self._lnormalize(len(self) - 1)
            return

        lnormalize = kwargs.get('left', 0)
        rnormalize = kwargs.get('right', len(self))

        assert lnormalize < rnormalize, \
            "Normalization {}:{} invalid".format(lnormalize, rnormalize)
        if current_lnorm < lnormalize:
            self._lnormalize(lnormalize)
        if current_rnorm > rnormalize:
            self._rnormalize(rnormalize)

    def _lnormalize(self, to_site):
        """Left-normalizes all local tensors _ltens[:to_site] in place

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
        """Right-normalizes all local tensors _ltens[to_site:] in place

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
        """Unified interface for the compression functions

        :param method: Which implemention should be used for compression
            'svd': Compression based on SVD :func:`MPArray.compress_svd`
            'var': Variational compression :func:`MPArray.compress_var`
        :param inplace: Compress the array in place or return new copy
        :returns: `self` if inplace is True or the compressed copy

        """
        if method == 'svd':
            target = self if inplace else self.copy()
            target.compress_svd(**kwargs)
            return target

        elif method == 'var':
            compr = self.compress_var(**kwargs)
            if inplace:
                self._ltens = compr[:]  # no copy necessary, compr is local
                return self
            else:
                return compr
        else:
            raise ValueError("{} is not a valid method.".format(method))

    def compress_svd(self, bdim=None, relerr=0.0, direction=None):
        """Compresses the MPA inplace using SVD [Sch11_, Sec. 4.5.1]

        :param bdim: Maximal bond dimension for the compressed MPA (default
            max of current bond dimensions, i.e. no compression)
        :param relerr: Maximal allowed error for each truncation step, that is
            the fraction of truncated singular values over their sum (default
            0.0, i.e. no compression)

        If both bdim and relerr is passed, the smaller resulting bond
        dimension is used.

        .. todo:: The documentation of `inplace` must be moved to
                  :func:`compress()`.

        :param direction: In which direction the compression should
            operate. (default: depending on the current normalization,
            such that the number of sites that need to be normalized
            is smaller)

            * 'right': Starting on the leftmost site, the compression
              sweeps to the right yielding a completely
              left-canonical MPA
            * 'left': Starting on rightmost site, the compression
              sweeps to the left yielding a completely
              right-canoncial MPA

        :returns:
            * inplace=True: Overlap <M|M'> of the original M and its
              compr. M'
            * inplace=False: Compressed MPA, Overlap <M|M'> of the
              original M and its compr. M',

        """
        if len(self) == 1:
            # Cannot do anything. Return perfect overlap.
            return norm(self)**2

        ln, rn = self.normal_form
        default_direction = 'left' if len(self) - rn > ln else 'right'
        direction = default_direction if direction is None else direction
        bdim = max(self.bdims) if bdim is None else bdim

        if direction == 'right':
            self.normalize(right=1)
            return self._compress_svd_r(bdim, relerr)
        elif direction == 'left':
            self.normalize(left=len(self) - 1)
            return self._compress_svd_l(bdim, relerr)

        raise ValueError('{} is not a valid direction'.format(direction))

    def compress_var(self, initmpa=None, bdim=None, randstate=np.random,
                     num_sweeps=5, sweep_sites=1):
        """Compresses the MPA using variational compression [Sch11_, Sec. 4.5.2]

        Does not change the current instance.

        :param initmpa: Initial MPA for the interative optimization, should
            have same physical shape as `self` (default random start vector
            with same norm as self)
        :param bdim: Maximal bond dimension for the random start vector
            (default max of current bond dimensions, i.e. no compression)
        :param randstate: numpy.random.RandomState instance or something
            suitable for :func:`factory.zrandn` (default numpy.random)
        :param num_sweeps: Maximum number of sweeps to do
        :param sweep_sites: Number of neighboaring sites minimized over
            simultaniously; for too small value the algorithm may get stuck
            in local minima (default 1)
        :returns: Compressed MPArray

        """
        if len(self) == 1:
            # Cannot do anything.
            return self

        if initmpa is None:
            from mpnum.factory import random_mpa
            bdim = max(self.bdims) if bdim is None else bdim
            compr = random_mpa(len(self), self.pdims, bdim, randstate=randstate)
            compr *= norm(self) / norm(compr)
        else:
            compr = initmpa.copy()
            assert all(d1 == d2 for d1, d2 in zip(self.pdims, compr.pdims))

        # flatten the array since MPS is expected & bring back
        shape = self.pdims
        compr = compr.ravel()
        compr._adapt_to(self.ravel(), num_sweeps, sweep_sites)
        compr = compr.reshape(shape)
        return compr

    def _compress_svd_r(self, bdim, relerr):
        """Compresses the MPA in place from left to right using SVD;
        yields a left-canonical state

        See :func:`MPArray.compress` for parameters
        """
        assert self.normal_form == (0, 1)
        assert bdim > 0, "Cannot compress to bdim={}".format(bdim)
        assert (0. <= relerr) and (relerr <= 1.), \
            "Relerr={} not allowed".format(relerr)

        for site in range(len(self) - 1):
            ltens = self._ltens[site]
            u, sv, v = svd(ltens.reshape((-1, ltens.shape[-1])))

            svsum = np.cumsum(sv) / np.sum(sv)
            bdim_relerr = np.searchsorted(svsum, 1 - relerr) + 1
            bdim_t = min(ltens.shape[-1], u.shape[1], bdim, bdim_relerr)

            newshape = ltens.shape[:-1] + (bdim_t, )
            self._ltens[site] = u[:, :bdim_t].reshape(newshape)
            self._ltens[site + 1] = matdot(sv[:bdim_t, None] * v[:bdim_t, :],
                                           self._ltens[site + 1])

        self._lnormalized = len(self) - 1
        self._rnormalized = len(self)
        return np.sum(np.abs(self._ltens[-1])**2)

    def _compress_svd_l(self, bdim, relerr):
        """Compresses the MPA in place from right to left using SVD;
        yields a right-canonical state

        See :func:`MPArray.compress` for parameters

        """
        assert self.normal_form == (len(self) - 1, len(self))
        assert bdim > 0, "Cannot compress to bdim={}".format(bdim)
        assert (0. <= relerr) and (relerr <= 1.), \
            "Relerr={} not allowed".format(relerr)

        for site in range(len(self) - 1, 0, -1):
            ltens = self._ltens[site]
            matshape = (ltens.shape[0], -1)
            u, sv, v = svd(ltens.reshape(matshape))

            svsum = np.cumsum(sv) / np.sum(sv)
            bdim_relerr = np.searchsorted(svsum, 1 - relerr) + 1
            bdim_t = min(ltens.shape[0], v.shape[0], bdim, bdim_relerr)

            newshape = (bdim_t, ) + ltens.shape[1:]
            self._ltens[site] = v[:bdim_t, :].reshape(newshape)
            self._ltens[site - 1] = matdot(self._ltens[site - 1],
                                           u[:, :bdim_t] * sv[None, :bdim_t])

        self._lnormalized = 0
        self._rnormalized = 1
        return np.sum(np.abs(self._ltens[0])**2)

    #  Possible TODOs:
    #
    #  - implement calculating the overlap between 'compr' and 'target' from
    #  the norm of 'compr', given that 'target' is normalized
    #  - track overlap between 'compr' and 'target' and stop sweeping if it
    #  is small
    #  - maybe increase bond dimension of given error cannot be reached
    #  - Shall we track the error in the SVD truncation for multi-site
    #  updates? [Sch11_] says it turns out to be useful in actual DMRG.
    #  - return these details for tracking errors in larger computations
    # TODO Refactor. Way too involved!
    # FIXME Does this play nice with different bdims?
    def _adapt_to(self, target, num_sweeps, sweep_sites):
        """Iteratively minimize the l2 distance between `self` and `target`.
        This is especially important for variational compression, where `self`
        is the initial guess and target the MPA to be compressed.

        :param target: MPS to compress; i.e. MPA with only one physical leg per
            site
        :param num_sweeps: Maximum number of sweeps to do
        :param sweep_sites: Number of neighboaring sites minimized over
            simultaniously; for too small value the algorithm may get stuck
            in local minima (default 1)
        """
        # For
        #
        #   pos in range(nr_sites - sweep_sites),
        #
        # we find the ground state of an operator supported on
        #
        #   range(pos, pos_end),  pos_end = pos + minimize_sites
        #
        # lvecs[pos] and rvecs[pos] contain the vectors needed to construct that
        # operator for that. Therefore, lvecs[pos] is constructed from matrices on
        #
        #   range(0, pos - 1)
        #
        # and rvecs[pos] is constructed from matrices on
        #
        #   range(pos_end, nr_sites),  pos_end = pos + minimize_sites
        assert_array_equal(self.plegs, 1, "Self is not a MPS")
        assert_array_equal(target.plegs, 1, "Target is not a MPS")

        nr_sites = len(target)
        lvecs = [np.array(1, ndmin=2)] + [None] * (nr_sites - sweep_sites)
        rvecs = [None] * (nr_sites - sweep_sites) + [np.array(1, ndmin=2)]
        self.normalize(right=1)
        for pos in reversed(range(nr_sites - sweep_sites)):
            pos_end = pos + sweep_sites
            rvecs[pos] = _adapt_to_add_r(rvecs[pos + 1], self[pos_end],
                                         target[pos_end])

        max_bonddim = max(self.bdims)
        for num_sweep in range(num_sweeps):
            # Sweep from left to right
            for pos in range(nr_sites - sweep_sites + 1):
                if pos == 0 and num_sweep > 0:
                    # Don't do first site again if we are not in the first sweep.
                    continue
                if pos > 0:
                    self.normalize(left=pos)
                    rvecs[pos - 1] = None
                    lvecs[pos] = _adapt_to_add_l(lvecs[pos - 1], self[pos - 1],
                                                 target[pos - 1])
                pos_end = pos + sweep_sites
                self[pos:pos_end] = _adapt_to_new_lten(lvecs[pos],
                                                       target[pos:pos_end],
                                                       rvecs[pos], max_bonddim)

            # NOTE Why no num_sweep > 0 here???
            # Sweep from right to left (don't do last site again)
            for pos in reversed(range(nr_sites - sweep_sites)):
                pos_end = pos + sweep_sites
                if pos < nr_sites - sweep_sites:
                    # We always do this, because we don't do the last site again.
                    self.normalize(right=pos_end)
                    lvecs[pos + 1] = None
                    rvecs[pos] = _adapt_to_add_r(rvecs[pos + 1], self[pos_end],
                                                 target[pos_end])

                self[pos:pos_end] = _adapt_to_new_lten(lvecs[pos],
                                                       target[pos:pos_end],
                                                       rvecs[pos], max_bonddim)

        return self


#############################################
#  General functions to deal with MPArrays  #
#############################################
def dot(mpa1, mpa2, axes=(-1, 0)):
    """Compute the matrix product representation of a.b over the given
    (physical) axes. [Sch11_, Sec. 4.2]

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


def partialdot(mpa1, mpa2, start_at, axes=(-1, 0)):
    """Partial dot product of two MPAs of inequal length.

    The shorter MPA will start on site 'start_at'. Local dot products
    will be carried out on all sites of the shorter MPA. Other sites
    will remain unmodified.

    mpa1 and mpa2 can also have equal length with start_at = 0. Then
    we do the same as dot(), with the axes argument being more
    flexible.

    :param mpa1, mpa2: Factors as MPArrays, length must be inequal.
    :param start_at: The shorter MPA will start on this site.
    :param axes: 2-tuple of axes to sum over. Note the difference in
        convention compared to np.tensordot(default: last axis of `mpa1`
        and first axis of `mpa2`)
    :returns: MPA with length of the longer MPA.

    """
    # adapt the axes from physical to true legs
    axes = tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes)

    # Make the MPAs equal length (in fact, the shorter one will be
    # infinite length, but that's fine because we use zip()).
    shorter = mpa1 if len(mpa1) < len(mpa2) else mpa2
    shorter = it.chain(
        it.repeat(None, times=start_at), shorter, it.repeat(None))
    if len(mpa1) < len(mpa2):
        mpa1 = shorter
    else:
        mpa2 = shorter

    ltens_new = (
        l if r is None else (r if l is None else _local_dot(l, r, axes))
        for l, r in zip(mpa1, mpa2)
        )
    return MPArray(ltens_new)


# NOTE: I think this is a nice example how we could use Python's generator
#       expression to implement lazy evaluation of the matrix product structure
#       which is the whole point of doing this in the first place
def inner(mpa1, mpa2):
    """Compute the inner product <mpa1|mpa2>. Both have to have the same
    physical dimensions. If these represent a MPS, inner(...) corresponds to
    the canoncial Hilbert space scalar product, if these represent a MPO,
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
    return _ltens_to_array(ltens_new)[0, ..., 0]


def outer(mpas):
    """Performs the tensor product of MPAs given in `*args`

    :param mpas: Iterable of MPAs same order as they should appear in the chain
    :returns: MPA of length len(args[0]) + ... + len(args[-1])

    """
    # TODO Make this normalization aware
    # FIXME Is copying here a good idea?
    return MPArray(sum(([ltens.copy() for ltens in mpa] for mpa in mpas), []))


def inject(mpa, pos, num, inject_ten=None):
    """Like outer(), but place second factor somewhere inside mpa.

    Return the outer product between mpa and 'num' copies of the local
    tensor 'inject_ten', but place the copies of 'inject_ten' before
    site 'pos' inside 'mpa'. Placing at the edges of 'mpa' is not
    supported (use outer() for that).

    If 'inject_ten' is omitted, use a square identity matrix of size
    mpa.pdims[pos][0].

    :param mpa: An MPA.
    :param pos: Inject sites into the MPA before site 'pos'.
    :param num: Inject 'num' copies.
    :param inject_ten: Inject this physical tensor (if None use
       np.eye(mpa.pdims[pos][0]))
    :returns: An MPA of length len(mpa) + num

    """
    if inject_ten is None:
        inject_ten = np.eye(mpa.pdims[pos][0])
    bdim = mpa.bdims[pos - 1]
    inject_lten = np.tensordot(np.eye(bdim), inject_ten, axes=((), ()))
    inject_lten = np.rollaxis(inject_lten, 1, inject_lten.ndim)
    ltens = it.chain(
        mpa[:pos], it.repeat(inject_lten, times=num), mpa[pos:])
    return MPArray(ltens)


def norm(mpa):
    """Computes the norm (Hilbert space norm for MPS, Frobenius norm for MPO)
    of the matrix product operator. In contrast to `mparray.inner`, this can
    take advantage of the normalization

    WARNING This also changes the MPA inplace by normalizing.

    :param mpa: MPArray
    :returns: l2-norm of that array

    """
    mpa.normalize()
    current_lnorm, current_rnorm = mpa.normal_form

    if current_rnorm == 1:
        return np.sqrt(np.vdot(mpa[0], mpa[0]))
    elif current_lnorm == len(mpa) - 1:
        return np.sqrt(np.vdot(mpa[-1], mpa[-1]))
    else:
        raise ValueError("Normalization error in MPArray.norm")


def normdist(mpa1, mpa2):
    """More efficient version of norm(mpa1 - mpa2)

    :param mpa1: MPArray
    :param mpa2: MPArray
    :returns: l2-norm of mpa1 - mpa2

    """
    return norm(mpa1 - mpa2)
    #  return np.sqrt(norm(mpa1)**2 + norm(mpa2)**2 - 2 * np.real(inner(mpa1, mpa2)))


def _prune_ltens(mpa):
    """Contract local tensors with no physical legs.

    By default, contract to the left. At the start of the chain,
    contract to the right.

    If there are no physical legs at all, yield a single scalar.

    :param mpa: MPA to take local tensors from.
    :returns: An iterator over local tensors.

    """
    mpa_iter = iter(mpa)
    last_lten = next(mpa_iter)
    last_lten_plegs = last_lten.ndim - 2
    for lten in mpa_iter:
        num_plegs = lten.ndim - 2
        if num_plegs == 0 or last_lten_plegs == 0:
            last_lten = matdot(last_lten, lten)
            last_lten_plegs = last_lten.ndim - 2
        else:
            # num_plegs > 0 and last_lten_plegs > 0
            yield last_lten
            last_lten = lten
            last_lten_plegs = num_plegs
    # If there are no physical legs at all, we will yield one scalar
    # here.
    yield last_lten


def prune(mpa):
    """Contract sites with zero physical legs.

    :param mpa: MPA or iterator over local tensors
    :returns: An MPA of smaller length

    """
    return MPArray(_prune_ltens(mpa))


def partialtrace(mpa, axes=(0, 1)):
    """Computes the trace or partial trace of an MPA.

    By default (axes=(0, 1)) compute the trace and return the value as
    length-one MPA with zero physical legs.

    For axes=(m, n) with integer m, trace over the given axes at all
    sites and return a length-one MPA with zero physical legs. (Use
    trace() to get the value directly.)

    For axes=(axes1, axes2, ...) trace over axesN at site N, with
    axesN=(axisN_1, axisN_2) tracing the given physical legs and
    axesN=None leaving the site invariant. Afterwards, prune() is
    called to remove sites with zero physical legs from the result.

    If you need the reduced state of an MPO on all blocks of k
    consecutive sites, see mpnum.mpsmpa.partialtrace_mpo() for a more
    convenient and faster function.

    :param mpa: MPArray
    :param axes: Axes for trace, (axis1, axis2) or (axes1, axes2, ...)
        with axesN=(axisN_1, axisN_2) or axesN=None.
    :returns: An MPArray (possibly one site with zero physical legs)

    """
    if axes[0] is not None and not hasattr(axes[0], '__iter__'):
        axes = it.repeat(axes)
    axes = (None if axesitem is None else tuple(ax + 1 if ax >= 0 else ax - 1
                                                for ax in axesitem)
            for axesitem in axes)
    ltens = (
        lten if ax is None else np.trace(lten, axis1=ax[0], axis2=ax[1])
        for lten, ax in zip(mpa, axes))
    return prune(ltens)


def trace(mpa, axes=(0, 1)):
    """Compute the trace of the given MPA.

    By default, just compute the trace.

    If you specify axes (see partialtrace() for details), you must
    ensure that the result has no physical legs anywhere.

    :param mpa: MParray
    :param axes: Axes for trace, (axis1, axis2) or (axes1, axes2, ...)
        with axesN=(axisN_1, axisN_2) or axesN=None.
    :returns: A single scalar (int/float/complex, depending on mpa)

    """
    out = partialtrace(mpa, axes)
    out = out.to_array()
    assert out.size == 1, 'trace must return a single scalar'
    return out[None][0]


class regular_slices:

    def __init__(self, length, width, offset):
        """
    
        .. todo:: This table needs cell borders (-> CSS), and the
                  tabularcolumns command doesn't  work.
    
        .. tabularcolumns:: |c|c|c|
    
        +------------------+--------+
        | #### width ##### |        |
        +--------+---------+--------+
        | offset | overlap | offset |
        +--------+---------+--------+
        |        | ##### width #### |
        +--------+------------------+
    
        >>> n = 5
        >>> [tuple(range(*s.indices(n))) for s in regular_slices(n, 3, 2)]
        [(0, 1, 2), (2, 3, 4)]
        >>> n = 7
        >>> [tuple(range(*s.indices(n))) for s in regular_slices(n, 3, 2)]
        [(0, 1, 2), (2, 3, 4), (4, 5, 6)]
    
        """
        assert ((length - width) % offset) == 0
        num_slices = (length - width) // offset + 1

        self.length, self.width, self.offset = length, width, offset
        self.num_slices = num_slices
    
    def __iter__(self):
        for i in range(self.num_slices):
            yield slice(self.offset * i, self.offset * i + self.width)


def default_embed_ltens(mpa, embed_tensor):
    if embed_tensor is None:
        pdims = mpa.pdims[0]
        assert len(pdims) == 2 and pdims[0] == pdims[1], (
            "For plegs != 2 or non-square pdims, you must supply a tensor"
            "for embedding")
        embed_tensor = np.eye(pdims[0])
    embed_ltens = embed_tensor[None, ..., None]
    return embed_ltens


def embed_slice(length, slice_, mpa, embed_tensor=None):
    start, stop, step = slice_.indices(length)
    assert step == 1
    assert len(mpa) == stop - start
    embed_ltens = default_embed_ltens(mpa, embed_tensor)
    left = it.repeat(embed_ltens, times=start)
    right = it.repeat(embed_ltens, times=length - stop)
    return MPArray(it.chain(left, mpa, right))


def local_sum(mpas, embed_tensor=None, length=None, slices=None):
    if isinstance(slices, regular_slices) and slices.offset == 1:
        mpas = tuple(mpas)
        assert len(tuple(slices)) == len(mpas)
        assert all(slices.width == len(mpa) for mpa in mpas)
        slices = None
    if slices is None:
        return local_sum_simple(mpas, embed_tensor)
    mpas = (embed_slice(length, slice_, mpa, embed_tensor)
            for mpa, slice_ in zip(mpas, slices))
    return ft.reduce(MPArray.__add__, mpas)


def local_sum_simple(mpas, embed_tensor=None):
    """Embed local MPAs on a linear chain and sum as MPA.

    The resulting MPA has smaller bond dimension than naive
    embed+MPA-sum.

    mpas is a list of MPAs. The width 'width' of all the mpas[i] must
    be the same. mpas[i] is embedded onto a linear chain on sites i,
    ..., i + width - 1. Let D the bond dimension of the mpas[i]. Then
    the MPA we return has bond dimension width * D + 1 instead of
    width * D + len(mpas).

    The basic idea behind the construction we use is similar to
    [Sch11_, Sec. 6.1].

    :param mpas: A list of MPArrays with the same length.
    :param embed_tensor: If the MPAs do not have two physical legs or
        have non-square physical dimensions, you must provide an
        embedding tensor instead of the identity matrix.

    """
    width = len(mpas[0])
    nr_sites = len(mpas) + width - 1
    ltens = []
    embed_ltens = default_embed_ltens(mpas[0], embed_tensor)

    for pos in range(nr_sites):
        # At this position, we local summands mpas[i] starting at the
        # following startsites are present:
        startsites = range(max(0, pos - width + 1),
                           min(len(mpas), pos + 1))
        mpas_ltens = [mpas[i]._ltens[pos - i] for i in startsites]
        # The embedding tensor embed_ltens has to be added if
        # - we need an embedding tensor on the right of an mpas[i]
        #   (pos is large enough)
        # - we need an embedding tensor on the left of an mpas[i]
        #   (pos is small enough)
        if pos >= width:
            mpas_ltens[0] = np.concatenate((embed_ltens, mpas_ltens[0]), axis=0)
        if pos < len(mpas) - 1:
            mpas_ltens[-1] = np.concatenate((mpas_ltens[-1], embed_ltens), axis=-1)

        lten = block_diag(mpas_ltens, axes=(0, -1))
        ltens.append(lten)

    mpa = MPArray(ltens)
    return mpa


############################################################
#  Functions for dealing with local operations on tensors  #
############################################################
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
    res = np.zeros(shape, dtype=max(ltens_l.dtype, ltens_r.dtype))

    res[:ltens_l.shape[0], ..., :ltens_l.shape[-1]] = ltens_l
    res[ltens_l.shape[0]:, ..., ltens_l.shape[-1]:] = ltens_r
    return res


def _local_ravel(ltens):
    """Flattens the physical legs of ltens, the bond-legs remain untouched

    :param ltens: :func:`numpy.ndarray` with :code:`ndim > 1`

    :returns: Reshaped `ltens` with shape :code:`(ltens.shape[0], ...,
        ltens.shape[-1])`, where :code`...` is determined from the
        size of ltens

    """
    return _local_reshape(ltens, (-1, ))


def _local_reshape(ltens, shape):
    """Reshapes the physical legs of ltens, the bond-legs remain untouched

    :param ltens: numpy.ndarray with ndim > 1
    :param shape: New shape of physical legs
    :returns: Reshaped ltens

    """
    full_shape = ltens.shape
    return ltens.reshape((full_shape[0], ) + tuple(shape) + (full_shape[-1], ))


def _local_transpose(ltens):
    """Transposes the physical legs of the local tensor `ltens`

    :param ltens: Local tensor as numpy.ndarray with ndim >= 2
    :returns: Transpose of ltens except for first and last dimension

    """
    return np.transpose(ltens, axes=[0] + list(range(ltens.ndim - 2, 0, -1)) +
                        [ltens.ndim - 1])


def _ltens_to_array(ltens):
    """Computes the full array representation from an iterator yielding the
    local tensors. Note that it does not get rid of bond legs.

    :param ltens: Iterator over local tensors
    :returns: numpy.ndarray representing the contracted MPA

    """
    ltens = ltens if hasattr(ltens, '__next__') else iter(ltens)
    res = next(ltens)
    for tens in ltens:
        res = matdot(res, tens)
    return res


################################################
#  Helper methods for variational compression  #
################################################
def _adapt_to_add_l(leftvec, compr_lten, tgt_lten):
    """Add one column to the left vector.

    :param leftvec: existing left vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param compr_lten: Local tensor of the compressed MPS
    :param tgt_lten: Local tensor of the target MPS

    Construct L from [Sch11_, Fig. 27, p. 48]. We have compr_lten in
    the top row of the figure without complex conjugation and tgt_lten
    in the bottom row with complex conjugation.

    """
    leftvec_names = ('compr_bond', 'tgt_bond')
    compr_names = ('compr_left_bond', 'compr_phys', 'compr_right_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    leftvec = named_ndarray(leftvec, leftvec_names)
    compr_lten = named_ndarray(compr_lten, compr_names)
    tgt_lten = named_ndarray(tgt_lten, tgt_names)

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


def _adapt_to_add_r(rightvec, compr_lten, tgt_lten):
    """Add one column to the right vector.

    :param rightvec: existing right vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param compr_lten: Local tensor of the compressed MPS
    :param tgt_lten: Local tensor of the target MPS

    Construct R from [Sch11_, Fig. 27, p. 48]. See comments in
    _variational_compression_leftvec_add() for further details.

    """
    rightvec_names = ('compr_bond', 'tgt_bond')
    compr_names = ('compr_left_bond', 'compr_phys', 'compr_right_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    rightvec = named_ndarray(rightvec, rightvec_names)
    compr_lten = named_ndarray(compr_lten, compr_names)
    tgt_lten = named_ndarray(tgt_lten, tgt_names)

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


def _adapt_to_new_lten(leftvec, tgt_ltens, rightvec, max_bonddim):
    """Create new local tensors for the compressed MPS.

    :param leftvec: Left vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param tgt_ltens: List of local tensor of the target MPS
    :param rightvec: Right vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param int max_bonddim: Maximal bond dimension of the result

    Compute the right-hand side of [Sch11_, Fig. 27, p. 48]. We have
    compr_lten in the top row of the figure without complex
    conjugation and tgt_lten in the bottom row with complex
    conjugation.

    For len(tgt_ltens) > 1, compute the right-hand side of [Sch11_,
    Fig. 29, p. 49].

    """
    # Produce one MPS local tensor supported on len(tgt_ltens) sites.
    tgt_lten = _ltens_to_array(tgt_ltens)
    tgt_lten_shape = tgt_lten.shape
    tgt_lten = tgt_lten.reshape((tgt_lten_shape[0], -1, tgt_lten_shape[-1]))

    # Contract the middle part with the left and right parts.
    leftvec_names = ('compr_left_bond', 'tgt_left_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    rightvec_names = ('compr_right_bond', 'tgt_right_bond')
    leftvec = named_ndarray(leftvec, leftvec_names)
    tgt_lten = named_ndarray(tgt_lten, tgt_names)
    rightvec = named_ndarray(rightvec, rightvec_names)

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
        # [Sch11_, p. 49] says that we can go with QR instead of SVD
        # here. However, this will generally increase the bond dimension of
        # our compressed MPS, which we do not want.
        compr_ltens = MPArray.from_array(compr_lten, plegs=1, has_bond=True)
        compr_ltens.compress_svd(bdim=max_bonddim)
    return compr_ltens
