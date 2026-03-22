"""
mhr_numerics
============
Numerical mathematics library for lattice reduction, finite-field linear
algebra, polynomial interpolation, and variational optimisation.

Published as a reproducible software artefact accompanying the manuscript:
  "Spectral Modular Identity and Variational Lattice Reduction"

Modules
-------
linalg_Fp       Linear algebra over finite prime fields F_p
linalg_GF2      Linear algebra over GF(2)
lattice         Lattice reduction: Gaussian 2D, Babai CVP, LLL, BKZ
interpolation   Lagrange interpolation and Berlekamp–Massey over F_p
variational     MHR variational solver (Gaussian ansatz, L-BFGS-B)

Citation
--------
If you use this library, please cite the accompanying paper and this
Zenodo software record.

License
-------
MIT License. See LICENSE for details.
"""

__version__ = "1.0.0"
__author__ = "A. Timakov"
