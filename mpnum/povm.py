#!/usr/bin/env python
# encoding: utf-8
"""An informationally d-level POVM that simplifies to measuring Paulis matrices in the case of qubits.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from six.moves import range, zip #@UnresolvedImport
from numpy.linalg.linalg import pinv
import mpnum.mparray as mp
import itertools


class Base(object):
    """
    Subclasses must supply the following attributes and methods:
    
    :attribute opts: A dictionary with options; the key 'd' (local dimension) must exist
    :attribute informationally_complete: Whether the POVM is informationally complete
    :attribute elements: A num_elements-by-d^2 array containing the reshaped POVM elements
    :attribute vectors: A num_elements-by-d array containing vectors (if the elements are rank-1)
    
    :method __init__(self, opts): Constructor that accepts self.opts
    """
    
    def _init_elements_from_vectors(self):
        """Populate self.elements from self.vectors
        
        Can be called in the constructor of a subclass.
        """
        d = self.opts['d']
        num_elements = self.vectors.shape[0]
        self.elements = np.zeros((num_elements, d**2), dtype=complex)
        for k in range(num_elements):
            vec = self.vectors[k, :]
            self.elements[k, :] = np.outer(vec, vec.conj()).reshape(d**2)
    
    @property
    def probability_map(self):
        """Return the map that takes a (reshaped) density matrix to the POVM probabilities.
        
        For the POVMs defined in this module, the ".conj()" doesn't really matter because it amounts to changing only the order of the POVM elements.
        """
        try:
            return self._probability_map
        except AttributeError:
            self._probability_map = self.elements.conj()
            return self._probability_map
    
    @property
    def linear_inversion_map(self):
        """Return the map that reconstructs a density matrix with linear inversion.
        
        Linear inversion is performed by taking the Moore--Penrose pseudoinverse of self.elements.conj().
        """
        try:
            return self._linear_inversion_map
        except AttributeError:
            self._linear_inversion_map = pinv(self.probability_map, **self.opts.get('pinv_kwargs', {}))
            return self._linear_inversion_map
    
    def get_probability_map_mpa(self, nr_sites):
        """Return the nr_sites-fold Kronecker (tensor) product of the probability map as MPA. 
        """
        try:
            return self._probability_map_mpo[nr_sites]
        except AttributeError:
            self._probability_map_mpo = {}
        except KeyError:
            pass
        self._probability_map_mpo[nr_sites] = mp.MPArray.from_kron(itertools.repeat(self.probability_map, nr_sites))
        return self._probability_map_mpo[nr_sites]
        
    def get_linear_inversion_map_mpo(self, nr_sites):
        """Return the nr_sites-fold Kronecker (tensor) product of the linear inversion map as MPA. 
        """
        try:
            return self._linear_inversion_mpo[nr_sites]
        except AttributeError:
            self._linear_inversion_mpo = {}
        except KeyError:
            pass
        self._linear_inversion_mpo[nr_sites] = mp.MPArray.from_kron(itertools.repeat(self.linear_inversion_map, nr_sites))
        return self._linear_inversion_mpo[nr_sites]
    
    def get_element_sum_map_mpa(self, nr_sites):
        """for self.maxlik_R()
        """
        return mp.MPArray.from_kron(itertools.repeat(self.elements.T, nr_sites))
    
    def maxlik_R(self, rho_mpa, frequencies):
        """Implement the local R operator required for the MLE iterations.
        
        rho_mpa must be an mpa with 1 physical leg (the two physical legs must be reshaped into one)
        
        UNTESTED 
        """
        nr_sites = len(rho_mpa)
        probab_rho = mp.dot(self.get_probability_map_mpa(nr_sites), rho_mpa).to_array()
        fraction = frequencies / probab_rho
        fraction_mpa = mp.MPArray.from_array(fraction, 1)
        m = self.get_element_sum_map_mpa(nr_sites)
        R = mp.dot(m, fraction_mpa)
        return R
        


class X(Base):
    """
    The X POVM simplifies to measuring Pauli X eigenvectors for d=2.
    
    For d > 2, we embed the X eigenvectors at all d*(d-1)/2 possible positions. 
    """
    
    informationally_complete = False
    
    def __init__(self, opts):
        self.opts = opts
        d = self.opts['d']
        self.vectors = np.zeros([d*(d-1), d])
        k = 0
        for i in range(d-1):
            for j in range(i + 1, d):
                self.vectors[k, i] = 1
                self.vectors[k, j] = 1
                k += 1
                self.vectors[k, i] = 1
                self.vectors[k, j] = -1
                k += 1
        self.vectors /= np.sqrt(2 * (d-1))
        self._init_elements_from_vectors()


class Y(Base):
    """
    The Y POVM simplifies to measuring Pauli Y eigenvectors for d=2.
    
    For d > 2, we embed the Y eigenvectors at all d*(d-1)/2 possible positions. 
    """
    
    informationally_complete = False
    
    def __init__(self, opts):
        self.opts = opts
        d = self.opts['d']
        self.vectors = np.zeros([d*(d-1), d], dtype=complex)
        k = 0
        for i in range(d-1):
            for j in range(i + 1, d):
                self.vectors[k, i] = 1
                self.vectors[k, j] = 1j
                k += 1
                self.vectors[k, i] = 1
                self.vectors[k, j] = -1j
                k += 1
        self.vectors /= np.sqrt(2 * (d-1))
        self._init_elements_from_vectors()


class Z(Base):
    """
    The Z POVM simplifies to measuring Pauli Z eigenvectors for d=2.
    
    For d > 2, we embed the Z eigenvectors at all d possible positions. 
    """
    
    informationally_complete = False
    
    def __init__(self, opts):
        self.opts = opts
        d = self.opts['d']
        self.vectors = np.zeros([d, d])
        for i in range(d):
            self.vectors[i, i] = 1
        self._init_elements_from_vectors()


class Combined(Base):
    """
    Create a new POVM by concatenating the elements from other POVMs.
    
    Subclasses must define:
    
    :attribute povms: A sequence of POVM classes.
    :attribute weights: A weight for each POVM, must sum to 1.
    """
    
    def _init_combined(self):
        povms = [ povm(self.opts) for povm in self.povms ]
        self.vectors = np.concatenate([povm.vectors * weight for povm, weight in zip(povms, self.weights)], axis=0)
        self.elements = np.concatenate([povm.elements * weight for povm, weight in zip(povms, self.weights)], axis=0)

class PauliGen(Combined):
    """
    An informationally complete d-level POVM that simplifies to measuring Pauli matrices in the case of qubits.
    
    For d = 2, combine the X, Y and Z POVMs with equal weights.
    
    For d > 3, combine the X and Y POVMs with equal weights. 
    They are informationally complete already. 
    """
    
    povms = (X, Y)
    weights = (0.5, 0.5)
    informationally_complete = True
    
    def __init__(self, opts):
        self.opts = opts
        d = self.opts['d']
        if d == 2:
            self.povms += (Z,)
            self.weights = (1/3., 1/3., 1/3.)
        self._init_combined()


all_povms = [X, Y, Z, PauliGen]


