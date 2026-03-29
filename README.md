# mhr-numerics

[![Python Test Suite](https://github.com/TimakovGenesis/mhr-numerics/actions/workflows/pytest.yml/badge.svg)](https://github.com/TimakovGenesis/mhr-numerics/actions/workflows/pytest.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19165246.svg)](https://doi.org/10.5281/zenodo.19165246)

**Numerical Mathematics Library for Arithmetic Spectroscopy**

Companion code to the paper series:

| Part | Title | Status | DOI |
|------|-------|--------|-----|
| I | Foundations of arithmetic spectroscopy | Submitted: *Exp. Math.* ID 267435039 | [10.5281/zenodo.19183116](https://doi.org/10.5281/zenodo.19183116) |
| II | The spectral rank criterion and Selmer theory | In preparation | [10.5281/zenodo.19165246](https://doi.org/10.5281/zenodo.19165246) |
| III | Néron–Tate regulator from spectral deformation | Planned | — |

---

## Overview

The library implements the **MHR (Modular Hamiltonian Relaxation)** operator framework for extracting arithmetic invariants of elliptic curves from spectral data. The central result:

```
lim_{λ→∞} ΔE(λ) = deg φ_E        (rank 0,  Theorem I.1)
ΔE(λ) ≡ 0                         (rank ≥ 1, Proposition I.2)
```

Part II adds the real variational solver and the full asymptotic rank signal `a₁ = −D²/π²`.

### Repository structure

```
mhr-numerics/
├── core/           Mathematical foundations (linalg, lattice, interpolation)
├── Part_I/         Gaussian ansatz variational solver (Part I)
├── Part_II/        Real MHR solver, deformation engine, verification (Part II)
├── figures/        Vector PDF figures for Part II
└── tests/          Test suite (< 30 s, CI-compatible)
```

---

## Installation

```bash
pip install numpy scipy
# Optional — for LLL/BKZ in core/lattice.py:
pip install fpylll
# Optional — for root finding in core/interpolation.py:
pip install sympy
```

---

## Quick start

### Part II: rank signal from spectral gap

```python
from Part_II.mhr_real_solver import mhr_real_solver

# Curve 37.a1: rank=0, deg φ_E = 2
delta_E, ok, iters, res = mhr_real_solver(1000.0, {"deg_phi": 2, "rank": 0})
print(f"ΔE(1000) = {delta_E:.7f}")   # → 1.9995501
```

### Part II: full deformation scan

```python
from Part_II.deformation import DeformationEngine
from Part_II.mhr_real_solver import mhr_real_solver
import numpy as np

eng = DeformationEngine(
    curve_params={"label": "37a1", "deg_phi": 2, "rank": 0},
    solver=mhr_real_solver,
)
result = eng.analyse(
    lambda_range=np.logspace(2, 5, 40),
    do_bootstrap=True,
    n_bootstrap=400,
)
print(f"a₁ = {result['fit'].a1:.4f}")
print(f"collapse: {result['fit'].spectral_collapse}")
```

### Core: linear algebra over F_p

```python
from core.linalg_Fp import gauss_jordan_Fp, det_Fp

A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
R, pivots, rank = gauss_jordan_Fp(A, p=101)
print(f"rank = {rank}")
```

---

## Verification (Part II)

Real solver vs Part I control values [F: Tab. 2–4]:

| λ | ΔE_real | ΔE_Part_I | Error |
|---|---------|-----------|-------|
| 100 | 1.9940956 | 1.9941000 | 2.2×10⁻⁶ |
| 1000 | 1.9995501 | 1.9995500 | 4.7×10⁻⁸ |
| 5000 | 1.9999152 | 1.9999150 | 8.9×10⁻⁸ |

Spectral collapse (37.b1, rank=1): ΔE from 4.7×10⁻² at λ=100 to < 10⁻⁹ at λ=1000.

```bash
python tests/smoke_test.py           # < 30 s
python Part_II/verify_real_solver.py # full table vs Part I
python -m pytest tests/ -v           # full suite
```

---

## Module reference

### `Part_II/`

| File | Description |
|------|-------------|
| `mhr_real_solver.py` | Real variational MHR solver. L-BFGS-B on Gaussian ansatz (Part I eq. 4.2). Bloch band width for rank ≥ 1. |
| `deformation.py` | Asymptotic deformation engine v2.0. Extracts a₁ = −D²/π² via regression + bootstrap. |
| `mhr_verifier.py` | Validates a₁ against −D²/π² (Proposition 5.1). Verdicts: RANK_ZERO / RANK_GE1. |
| `a1_calibrator.py` | Calibrates deformation.py parameters against benchmark curves. |
| `verify_real_solver.py` | Point-by-point comparison vs Part I Tab. 2–4. |
| `integration_test.py` | Full pipeline test: DeformationEngine + mhr_real_solver. |

### `Part_I/`

| File | Description |
|------|-------------|
| `variational.py` | Gaussian ansatz + L-BFGS-B. Proves `lim ΔE = deg φ_E` numerically. |

### `core/`

| File | Description |
|------|-------------|
| `linalg_Fp.py` | Gauss–Jordan, det, inverse, rank, solver over F_p. |
| `linalg_GF2.py` | Bit-packed RREF, null space, matrix product over GF(2). |
| `lattice.py` | Gaussian 2D reduction, Babai CVP, LLL/BKZ (fpylll), Hermite factor. |
| `interpolation.py` | Lagrange interpolation, Berlekamp–Massey LFSR synthesis over F_p. |

---

## Mathematical background

The MHR operator: `H_λ = −λ Δ + λ² V`, `V(x) = ‖Bx‖²`

Gaussian ansatz energy functional (Part I, eq. 4.2):
```
E(μ, σ) = 1/(4σ²) + D(μ² + σ²) + λ[sin²(πμ)·exp(−2π²σ²) + ½(1 − exp(−2π²σ²))]
```

Asymptotic expansion (Part II, Lemma A.1 [F]):
```
ΔE(λ) = D − (D²/π²)·λ⁻¹ + O(λ⁻³/²),   a₁ = −D²/π² < 0
```

---

## Citation

```bibtex
@software{timakov2026code,
  author = {Timakov, Andrew},
  title  = {mhr-numerics: Numerical Mathematics Library for Arithmetic Spectroscopy},
  year   = {2026},
  doi    = {10.5281/zenodo.19165246},
  url    = {https://github.com/TimakovGenesis/mhr-numerics}
}
```

---

## License

MIT. See `LICENSE` for details.

