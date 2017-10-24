---
title: 'mpnum: A matrix product representation library for python'
tags:
  - matrix-product
  - tensor-train
  - dmrg
authors:
 - name: Daniel Suess
   orcid: 0000-0002-6354-457X
   affiliation: 1
 - name: Milan Holzäpfel
   orcid: 0000-0002-4687-5027
   affiliation: 2
affiliations:
 - name: University of Cologne
   index: 1
 - name: Ulm University
   index: 2
date: 10 August 2017
bibliography: references.bib
---

# Summary

Tensors -- or high-dimensional arrays -- are ubiquitous in science and provide the foundation for numerous numerical algorithms in scientific computing, machine learning, signal processing, and other fields.
With their high demands in memory and computational time, tensor computations constitute the bottleneck of many such algorithms.
This has led to the development of sparse and low-rank tensor decompositions [@Decompositions].
One such decomposition, which was first developed under the name _"matrix product state"_ (MPS) in the study of entanglement in quantum physics[@Werner], is the _matrix product_ or _tensor train_ (TT) representation [@Schollwoeck,@Osedelets].

The matrix product tensor format is often used in practice (see e.g. [@Latorre,@NMR,@QuantumChemistry,@Uncertainty,@NeuralNetworks,@Stoudenmire]) for two reasons:
On the one hand, it captures the low-dimensional structure of many problems well. Therefore, it can be used model those problems computationally in an efficient way.
On the other hand, the matrix product tensor format also allows for performing crucial tensor operations -- such as addition, contraction, or low-rank approximation -- efficiently [@Schollwoeck,@Osedelets,@Orus,@Dance].

The library **mpnum** [@mpnum] provides a flexible, user-friendly, and expandable toolbox for prototyping algorithms based on the matrix-product tensor format.
Its fundamental data structure is the `MPArray` which represents a tensor with an arbitrary number of dimensions and local structure.
Based on the `MPArray`, **mpnum** implements basic linear algebraic operations such as addition, contraction, approximate eigenvalue computation, etc. as well as specialized matrix-product decomposition operations such as compression or canonicalization.
With these facilities, the user can express algorithms in high-level, readable code.
Examples from quantum physics include matrix-product state (MPS) and matrix-product operator (MPO) computations, DMRG, low-rank tensor recovery, and efficient quantum state estimation.


# References
