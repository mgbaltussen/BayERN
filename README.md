# BayERN

A Python package to perform Bayesian inference on enzyme kinetics data. 

Currently, the package implements custom Theano-operators to perform inference both on steady-state data of enzymatic systems and time-dynamic data. 
These operators can be used together with PyMC3 and implement full gradient support for the usage of Hamiltonian Monte Carlo methods.

# Installation

The current supported version requires a working installation of [Amici](https://amici.readthedocs.io/en/latest/python_installation.html#amici-python-installation).
This requires the following Ubuntu packages:

`sudo apt install libatlas-base-dev swig`
`sudo apt install libhdf5-serial-dev`

As well as the following Anaconda installation and environment variable:
`conda install -c conda-forge openblas`
`export BLAS_LIBS=-lopenblas`

The BayERN package can then be installed via pip:

`pip install git+https://github.com/mgbaltussen/BayERN.git@v0.2.0`

## Acknowledgements
Developed by Mathieu G. Baltussen in the Huck Group, Department of Physical-Organic Chemistry, Institute for Molecules and Materials, Radboud University.
