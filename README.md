[README.md](https://github.com/user-attachments/files/26167805/README.md)
# mhr_numerics

**Numerical mathematics library for lattice reduction, finite-field linear algebra, and variational optimisation.**

Reproducible software artefact accompanying the manuscript:
> *Spectral Modular Identity and Variational Lattice Reduction*

---

## Overview

`mhr_numerics` is a self-contained Python library providing rigorous, documented implementations of:

| Module | Contents |
|---|---|
| `linalg_Fp.py` | Gauss–Jordan elimination, determinant, inverse, rank, and linear system solving over arbitrary prime fields F_p |
| `linalg_GF2.py` | Bit-packed linear algebra over GF(2): RREF, rank, null space, matrix product, system solving |
| `lattice.py` | Gaussian 2D reduction, Babai nearest-plane CVP, LLL and BKZ reduction wrappers (via fpylll), Hermite factor |
| `interpolation.py` | Lagrange interpolation over F_p, polynomial arithmetic, root finding (Cantor–Zassenhaus via sympy), Berlekamp–Massey LFSR synthesis |
| `variational.py` | MHR variational solver: Gaussian ansatz energy functional, adiabatic coupling schedule, L-BFGS-B optimisation |

All functions carry **mathematical docstrings** with explicit formulas, algorithm descriptions, complexity bounds, and references to the primary literature.

---

## Installation

```bash
pip install numpy scipy sympy
# Optional (for LLL/BKZ):
pip install fpylll
```

No other dependencies are required.

---

## Quickstart

### Linear algebra over F_p

```python
from linalg_Fp import gauss_jordan_Fp, det_Fp, solve_Fp

A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
p = 101
R, pivots, rank = gauss_jordan_Fp(A, p)
print(f"rank = {rank}, pivots = {pivots}")

d = det_Fp(A, p)
print(f"det(A) mod {p} = {d}")

x = solve_Fp([[2, 1], [1, 3]], [5, 7], p=11)
print(f"solution: {x}")
```

### Lattice reduction

```python
from lattice import gauss_reduce_2d, lll_reduce, hermite_factor
import numpy as np

# 2D Gaussian reduction
b1, b2 = gauss_reduce_2d([13, 21], [8, 13])
print(f"Reduced: {b1}, {b2}")

# LLL reduction (requires fpylll)
B = [[1, 0, 2, 1], [0, 2, 1, 3], [3, 1, 0, 2], [1, 2, 3, 0]]
B_lll = lll_reduce(B, delta=0.99)
norm0 = np.linalg.norm(B_lll[0])
vol   = abs(np.linalg.det(np.array(B, dtype=float)))
delta = hermite_factor(norm0, vol, n=4)
print(f"Hermite factor δ = {delta:.4f}")
```

### Polynomial interpolation over F_p

```python
from interpolation import lagrange_interpolate_Fp, poly_eval_Fp, berlekamp_massey_Fp

# Reconstruct f(x) = x² + x + 1 over F_11
p = 11
f = lambda x: (x**2 + x + 1) % p
pts = [(i, f(i)) for i in range(3)]
poly = lagrange_interpolate_Fp(pts, p)
print(poly)   # [1, 1, 1]

# Berlekamp–Massey: find LFSR for a sequence
seq = [1, 1, 2, 3, 5, 1, 6, 0]   # Fibonacci mod 7
C = berlekamp_massey_Fp(seq, p=7)
print(f"Connection polynomial: {C}")
```

### MHR variational solver

```python
from lattice import lll_reduce
from variational import mhr_solve
import numpy as np

rng = np.random.default_rng(42)
B_raw = rng.integers(-10, 10, size=(8, 8)).tolist()
B_lll = lll_reduce(B_raw)

result = mhr_solve(B_lll, lambda_max=300, n_outer=60, seed=42)
print(f"Candidate: {result.candidate}")
print(f"‖B·v‖ = {result.candidate_norm:.4f}")
print(f"Converged: {result.converged}")
```

---

## Running the test suite

```bash
cd mhr_numerics
python -m pytest tests/ -v
```

All tests use **synthetic mathematical inputs** (random matrices, Mersenne primes, Fibonacci lattices) and do not depend on any external datasets.

Expected output: ≥ 40 tests, all passing.

---

## Mathematical foundations

### MHR energy functional

The variational energy for the Gaussian ansatz ψ_{μ,σ} on a lattice L = BZ^n is:

    E[μ, σ] = ⟨‖B·x‖²⟩_{N(μ,σ²)} + λ · ⟨Σ_j sin²(π·x_j)⟩_{N(μ,σ²)}

The quadratic term localises the ansatz near short lattice vectors; the confinement term λ·sin²(π·x_j) penalises non-integer coordinates. As λ → ∞, the energy minimum converges to ‖shortest vector‖² [Helffer–Sjöstrand 1984].

### Isotypic lattice construction

Given a generator γ_E of the isotypic component Λ_E ⊂ H₁(X₀(N), ℤ)⁺ with intersection norm ‖γ_E‖²_int = deg φ_E, the MHR operator is built on the lattice B = ‖γ_E‖_int · ℤ. The variational energy minimum then encodes deg φ_E:

    lim_{λ→∞} ΔE(λ) = deg φ_E

where ΔE(λ) = E₁(λ) − E_vac(λ) is the spectral gap. The convergence rate for the spectral gap is ε ∼ O(λ⁻¹), as derived in Timakov (2026). See [Helffer–Sjöstrand 1984] for the semiclassical localisation theory underlying this convergence.

---

## References

- Gauss, C.F. (1801). *Disquisitiones Arithmeticae*.
- Lenstra, A.K., Lenstra, H.W. & Lovász, L. (1982). Factoring polynomials with rational coefficients. *Math. Ann.* 261, 515–534.
- Babai, L. (1986). On Lovász' lattice reduction and the nearest lattice point problem. *Combinatorica* 6(1), 1–13.
- Schnorr, C.P. & Euchner, M. (1994). Lattice basis reduction. *Math. Programming* 66, 181–199.
- Helffer, B. & Sjöstrand, J. (1984). Multiple wells in the semi-classical limit I. *Comm. PDE* 9(4), 337–408.
- von zur Gathen, J. & Gerhard, J. (2013). *Modern Computer Algebra* (3rd ed.). Cambridge University Press.
- Berlekamp, E.R. (1968). *Algebraic Coding Theory*. McGraw-Hill.
- Massey, J.L. (1969). Shift-register synthesis and BCH decoding. *IEEE Trans. Inf. Theory* 15(1), 122–127.
- Nguyen, P.Q. & Vallée, B. (Eds.) (2010). *The LLL Algorithm*. Springer.

---

## License

MIT License. Copyright (c) 2026 A. Timakov.
