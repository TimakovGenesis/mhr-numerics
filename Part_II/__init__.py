"""
Part_II — The Spectral Rank Criterion and Selmer Theory
========================================================
Companion code to:
  "Spectral gap and modular degree of elliptic curves.
   Part II: The spectral rank criterion and Selmer theory"
  DOI: 10.5281/zenodo.19165246

Main entry point:

    from Part_II.mhr_real_solver import mhr_real_solver

    delta_E, ok, iters, res = mhr_real_solver(1000.0, {"deg_phi": 2, "rank": 0})
    # → delta_E ≈ 1.9995501  (curve 37.a1)

Verification [F: Part I, Tab. 2-4]:
    λ=100  → ΔE = 1.9940956,  error = 2.2×10⁻⁶
    λ=1000 → ΔE = 1.9995501,  error = 4.7×10⁻⁸
    λ=5000 → ΔE = 1.9999152,  error = 8.9×10⁻⁸
"""
__version__ = "2.1"

from Part_II.mhr_real_solver import mhr_real_solver

__all__ = ["mhr_real_solver"]
