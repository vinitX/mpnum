# encoding: utf-8
"""MPNUM: A matrix-product-representation library for Python

* :mod:`mpnum.mparray`: Basic matrix product array (MPA) routines and
  compression

* :mod:`mpnum.mpsmpo`: Convert matrix product state (MPS), matrix
  product operator (MPO) and locally purifying MPS (PMPS)
  representations and compute local reduced states.

* :mod:`mpnum.factory`: Generate random, MPS, MPOs, MPDOs, MPAs, etc.

* :mod:`mpnum.linalg`: Compute ground states (smallest eigenvalue and
  eigenvector) of MPOs

* :mod:`mpnum.povm`: Positive operator valued measures (POVM)

 * :mod:`mpnum.povm.localpovm`: Pauli-like POVM on a single site

 * :mod:`mpnum.povm.mppovm`: Matrix product POVM based on the
   Pauli-like POVM

"""


from .factory import *  # noqa: F401, F403
from .linalg import *   # noqa: F401, F403
from .mparray import *  # noqa: F401, F403
from .mpsmpo import *   # noqa: F401, F403


__version__ = "0.2.2"
