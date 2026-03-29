"""
Part_I — Foundations of Arithmetic Spectroscopy
================================================
Companion code to:
  "Spectral gap and modular degree of elliptic curves.
   Part I: Foundations of arithmetic spectroscopy"
  DOI: 10.5281/zenodo.19183116

Exports the Gaussian ansatz variational solver establishing
lim_{λ→∞} ΔE(λ) = deg φ_E for rank-0 curves (Theorem I.1).
"""
__version__ = "2.1"

from Part_I.variational import mhr_solve

__all__ = ["mhr_solve"]
