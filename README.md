# BayERN

A Python package to perform Bayesian inference on enzyme kinetics data. 

Currently, the package implements custom Theano-operators to calculate steady-states of enzymatic systems. 
These operators can be used together with PyMC3 and implement full gradient support for the usage of Hamiltonian Monte Carlo methods.

# Installation


## Stable version

`pip install git+https://github.com/mgbaltussen/BayERN.git@v0.1.0`

## Unstable developer branch
### Requirements

The unstable main branch requires a working version of [Amici](https://amici.readthedocs.io/en/latest/python_installation.html#amici-python-installation).
This requires the following to Ubuntu packages:

`sudo apt install libatlas-base-dev swig`
`sudo apt install libhdf5-serial-dev`

As well as the following Anaconda installation and environment variable:
`conda install -c conda-forge openblas`
`export BLAS_LIBS=-lopenblas`

The developer branch can then be installed directly from the git repository:

`pip install git+https://github.com/mgbaltussen/BayERN.git`



## Acknowledgements
Developed by Mathieu G. Baltussen in the Huck Group, Department of Physical-Organic Chemistry, Institute for Molecules and Materials, Radboud University.
