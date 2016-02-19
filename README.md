MPNUM: A matrix-product-representation library for Python
=========================================================

This code is work in progress.

This library implements various matrix product based representations,
such as matrix product states (MPS), matrix product operators (MPO)
and arbitrary matrix product arrays (MPA).  It implements basic
routines, SVD compression, variational compression and computing
ground states (the smallest eigenvalue and eigenvector) of MPOs.  In
addition, it provides flexible tools to easily implement new schemes
based on matrix product representations.

Required packages:

* six, numpy, scipy, sphinx (to build the documentation)

Supported Python versions:

* 2.7, 3.5


## License

(c) 2015, Daniel Suess <daniel@dsuess.me>, Milan Holzaepfel <mail@mjh.name>


## Building the HTML documentation

    export PYTHONPATH=$PWD
    cd docs
    make html
