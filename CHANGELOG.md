# Changelog

All notable changes to mhr-numerics are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — Part III (planned)

### Planned
- `Part_III/regulator_solver.py` — Néron–Tate regulator from quadratic form on flat directions
- `Part_III/nt_pairing.py` — spectral deformation → Néron–Tate pairing

---

## [2.1] — 2026-03-01 — Part II companion release

### Added
- `Part_II/mhr_real_solver.py` — real variational MHR solver (L-BFGS-B on
  Gaussian ansatz functional); replaces mock solver used in earlier drafts.
  Verified against Part I control values to errors ≤ 9×10⁻⁸.
- `Part_II/deformation.py` v2.0 — asymptotic deformation engine; extracts
  the leading correction coefficient a₁ = −D²/π² via regression and bootstrap.
- `Part_II/verify_real_solver.py` — systematic comparison vs Part I Tab. 2–4
  for curves 37.a1, 11.a1, 389.a1.
- `Part_II/integration_test.py` — full pipeline test through DeformationEngine
  for rank-0 (37.a1) and rank-1 (37.b1) curves.
- `Part_II/mhr_verifier.py` — validates a₁ against analytic −D²/π²
  (Part II, Proposition 5.1); verdicts RANK_ZERO / RANK_GE1.
- `Part_II/a1_calibrator.py` — calibrates deformation.py parameters against
  benchmark curves; writes `data/calibration.json`.
- `tests/smoke_test.py` — fast end-to-end test (< 30 s), CI-compatible.
- `figures/` — vector PDF figures for Part II (fig1–fig3).
- `CITATION.cff` — machine-readable citation metadata.

### Changed
- Repository restructured: `core/`, `Part_I/`, `Part_II/` subfolders.
- `README.md` updated to reflect Part II companion role.
- `zenodo_metadata.json` updated for v2.1.

### Verification
- ΔE errors < 9×10⁻⁸ vs Part I reference values (Tab. 2–4).
- Spectral collapse confirmed for rank ≥ 1 curves (Bloch band width → 0
  exponentially: 4.7×10⁻² at λ=100, < 10⁻⁹ at λ=1000 for 37.b1).
- All six test curves correctly classified (Table 2, Part II).

---

## [1.0] — 2026-01-15 — Part I companion release

### Added
- `Part_I/variational.py` — MHR variational solver: Gaussian ansatz energy
  functional, adiabatic coupling schedule, L-BFGS-B optimisation.
  Implements eq. (4.2) of Part I; control values in Tab. 2–4.
- `core/linalg_Fp.py` — exact linear algebra over F_p (Gauss–Jordan,
  determinant, inverse, rank, linear system solver).
- `core/linalg_GF2.py` — bit-packed linear algebra over GF(2) (RREF, rank,
  null space, matrix product, system solver).
- `core/lattice.py` — Gaussian 2D reduction, Babai CVP, LLL/BKZ wrappers
  (via fpylll), Hermite factor.
- `core/interpolation.py` — Lagrange interpolation and Berlekamp–Massey
  LFSR synthesis over F_p.
- `tests/` — initial test suite (≥ 40 synthetic tests, all passing).
- `.github/workflows/pytest.yml` — CI via GitHub Actions.

---

[Unreleased]: https://github.com/TimakovGenesis/mhr-numerics/compare/v2.1...HEAD
[2.1]: https://github.com/TimakovGenesis/mhr-numerics/compare/v1.0...v2.1
[1.0]: https://github.com/TimakovGenesis/mhr-numerics/releases/tag/v1.0
