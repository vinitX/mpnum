# encoding: utf-8
"""mpnum: A matrix-product-representation library for Python

* :mod:`mpnum.mparray`: Basic matrix product array (MPA) routines and
  compression

* :mod:`mpnum.mpsmpo`: Convert matrix product state (MPS), matrix
  product operator (MPO) and locally purifying MPS (PMPS)
  representations and compute local reduced states.

* :mod:`mpnum.factory`: Generate random, MPS, MPOs, MPDOs, MPAs, etc.

* :mod:`mpnum.linalg`: Compute the smallest eigenvalues & vectors of MPOs

* :mod:`mpnum.special`: Optimized versions of some routines for special cases

* :mod:`mppnum.mpstruct`: Underlying structure of MPAs to manage the local
  tensors

* :mod:`mpnum.povm`: Matrix product representation of Positive operator valued
  measures (POVM)

 * :mod:`mpnum.povm.localpovm`: Pauli-like POVM on a single site

 * :mod:`mpnum.povm.mppovm`: Matrix product POVM based on the
   Pauli-like POVM

References
----------
* .. _Sch11:

  [Sch11] Schollwöck, U. (2011). “The density-matrix renormalization
  group in the age of matrix product states”. Ann. Phys. 326(1),
  pp. 96–192. `DOI: 10.1016/j.aop.2010.09.012`_. `arXiv: 1008.3477`_.

  .. _`DOI: 10.1016/j.aop.2010.09.012`:
     http://dx.doi.org/10.1016/j.aop.2010.09.012

  .. _`arXiv: 1008.3477`: http://arxiv.org/abs/1008.3477


"""


from .factory import *  # noqa: F401, F403
from .linalg import *   # noqa: F401, F403
from .mparray import *  # noqa: F401, F403
from .mpsmpo import *   # noqa: F401, F403


__version__ = "0.2.2"
