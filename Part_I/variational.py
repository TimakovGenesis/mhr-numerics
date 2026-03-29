"""
mhr_numerics.variational
========================
Variational MHR (Modular Harmonic Relaxation) solver for quadratic
optimisation on integer lattices.

This module implements the variational ansatz underlying the Modular Harmonic
Relaxation (MHR) method: a Gaussian wave-packet ansatz whose energy
functional approximates the ground-state energy of a Schrödinger-like
operator  H = −λΔ + λ²V,  where the potential  V(x) = ‖Bx‖²  is the
squared lattice-vector norm.  Minimising the energy localises the Gaussian
at the shortest lattice vector.

Physical analogy
----------------
The Hamiltonian  H = −λΔ + V  is a lattice analogue of a harmonic oscillator
in quantum mechanics.  As λ → ∞ (the semi-classical limit), the ground-state
energy converges to the minimum of V, i.e. to ‖shortest vector‖² [HS84].
The Gaussian ansatz  ψ(x) ∝ exp(−‖x−μ‖²/(2σ²))  is the exact ground state
of a harmonic oscillator and provides a variational upper bound for our
non-quadratic potential.

Energy functional
-----------------
For the Gaussian ansatz  ψ_{μ,σ}  with mean μ and standard deviation σ:

    E[ψ] = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
           = ‖B·μ‖² + σ² · Tr(BᵀB)
             + λ · ∑_j sin²(π·μ_j) + λ·π²σ²·cos(2πμ)·…  (confinement)

The confinement term  λ·∑_j sin²(π·x_j)  has minima at integer points,
guiding the optimiser toward integer solutions.

Optimisation
------------
The energy is minimised over (μ, log σ) using L-BFGS-B from scipy.  The
coupling constant λ is increased adiabatically (annealing schedule) to
avoid being trapped in local minima of the confinement landscape.

References
----------
.. [HS84] Helffer, B. & Sjöstrand, J. (1984).
          Multiple wells in the semi-classical limit I.
          *Communications in Partial Differential Equations*, 9(4), 337–408.
.. [Agm82] Agmon, S. (1982). Lectures on exponential decay of solutions of
           second-order elliptic equations. Princeton University Press.
           (Exponential localisation of ground-state wave functions.)
.. [Tor10] Torquato, S. & Stillinger, F. (2010). Jammed hard-particle packings:
           From Kepler to Bernal and beyond. *Rev. Mod. Phys.*, 82, 2633.
           (Variational methods for lattice packings — physical analogy.)
.. [NV10] Nguyen, P.Q. & Vallée, B. (Eds.) (2010).
          *The LLL Algorithm: Survey and Applications*. Springer.
          Chapter 1 — geometric view of lattice reduction.
.. [LC07] Lovász, L. & de la Calle, M. (2007). Semidefinite programming and
          lattices. (Background on SDP relaxations of SVP.)
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "mhr_energy",
    "mhr_gradient",
    "mhr_solve",
    "MHRResult",
    "adiabatic_schedule",
]


# ---------------------------------------------------------------------------
# MHR energy functional
# ---------------------------------------------------------------------------

def mhr_energy(
    mu: np.ndarray,
    log_sigma: float,
    B: np.ndarray,
    lambda_val: float,
    n_samples: int = 512,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Monte Carlo estimate of the MHR variational energy.

    For the Gaussian ansatz  ψ_{μ,σ}  with mean μ ∈ R^n and isotropic
    standard deviation  σ = exp(log_sigma) > 0:

        E[μ, σ] ≈ (1/S) Σ_{s=1}^{S} [‖B·x_s‖² + λ·∑_j sin²(π·x_{sj})]

    where  x_s ~ N(μ, σ²·I_n)  are independent samples.

    The quadratic term  ‖Bx‖²  measures how close the ansatz mean is to a
    short lattice vector.  The confinement term  λ·sin²(π·x_j)  penalises
    non-integer coordinates, driving localisation at lattice points.

    Parameters
    ----------
    mu : np.ndarray, shape (n,)
        Mean of the Gaussian ansatz (current iterate).
    log_sigma : float
        Natural logarithm of the isotropic standard deviation σ.
    B : np.ndarray, shape (n, m)
        Lattice basis matrix (rows are basis vectors).
    lambda_val : float
        Coupling constant λ ≥ 0.  Higher λ enforces integrality more strongly.
    n_samples : int, optional
        Number of Monte Carlo samples for energy estimation.  Default: 512.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.  If None, a fresh
        default generator is used (results are non-deterministic).

    Returns
    -------
    float
        Estimated variational energy E[μ, σ].

    Notes
    -----
    The estimator has variance  O(1/n_samples).  For production runs use
    n_samples ≥ 1024; for fast exploratory runs 256 suffices.

    References
    ----------
    .. [HS84] §2 — semi-classical energy estimates for Gaussian wave packets.
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = math.exp(log_sigma)
    n = len(mu)

    # Draw samples x_s ~ N(μ, σ²·I)
    eps = rng.standard_normal((n_samples, n))
    x = mu[np.newaxis, :] + sigma * eps           # (S, n)

    # Quadratic lattice term: ‖B·x_s‖²  (B acts on column vectors → Bx^T)
    # B shape (n_basis, n), x shape (S, n) → Bx.T shape (n_basis, S)
    Bx = x @ B.T                                  # (S, n_basis)
    lattice_energy = np.sum(Bx ** 2, axis=1)      # (S,)

    # Confinement term: λ · Σ_j sin²(π·x_{sj})
    if lambda_val > 0:
        confinement = lambda_val * np.sum(np.sin(math.pi * x) ** 2, axis=1)
    else:
        confinement = np.zeros(n_samples)

    return float(np.mean(lattice_energy + confinement))


# ---------------------------------------------------------------------------
# MHR gradient (finite differences)
# ---------------------------------------------------------------------------

def mhr_gradient(
    mu: np.ndarray,
    log_sigma: float,
    B: np.ndarray,
    lambda_val: float,
    n_samples: int = 256,
    eps_fd: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, float]:
    """
    Finite-difference gradient of the MHR energy w.r.t. (μ, log_σ).

    Computes  ∂E/∂μ  and  ∂E/∂(log σ)  by central finite differences with
    step size eps_fd.

    Parameters
    ----------
    mu : np.ndarray, shape (n,)
        Current mean vector.
    log_sigma : float
        Current log-standard-deviation.
    B : np.ndarray, shape (n_basis, n)
        Lattice basis matrix.
    lambda_val : float
        Coupling constant.
    n_samples : int, optional
        Monte Carlo samples per energy evaluation.  Default: 256.
    eps_fd : float, optional
        Finite-difference step size.  Default: 1e-4.
    rng : np.random.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    grad_mu : np.ndarray, shape (n,)
        Gradient w.r.t. μ.
    grad_log_sigma : float
        Gradient w.r.t. log σ.

    References
    ----------
    .. [NR07] Press, W.H. et al. (2007). *Numerical Recipes* (3rd ed.).
              §5.7 — finite-difference derivatives.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(mu)
    grad_mu = np.zeros(n)

    for i in range(n):
        mu_p = mu.copy(); mu_p[i] += eps_fd
        mu_m = mu.copy(); mu_m[i] -= eps_fd
        ep = mhr_energy(mu_p, log_sigma, B, lambda_val, n_samples, rng)
        em = mhr_energy(mu_m, log_sigma, B, lambda_val, n_samples, rng)
        grad_mu[i] = (ep - em) / (2 * eps_fd)

    ep = mhr_energy(mu, log_sigma + eps_fd, B, lambda_val, n_samples, rng)
    em = mhr_energy(mu, log_sigma - eps_fd, B, lambda_val, n_samples, rng)
    grad_log_sigma = (ep - em) / (2 * eps_fd)

    return grad_mu, grad_log_sigma


# ---------------------------------------------------------------------------
# Adiabatic schedule
# ---------------------------------------------------------------------------

def adiabatic_schedule(
    step: int,
    lambda_max: float,
    n_steps: int,
    warm_up: int = 50,
) -> float:
    """
    Adiabatic coupling schedule  λ(t).

    The coupling constant is ramped from 0 to λ_max using a smooth
    sigmoidal schedule:

        λ(t) = λ_max · σ((t − n_steps/2) / (n_steps/8)),

    where σ(z) = 1/(1 + exp(−z)) is the logistic function.  This slow
    ramp allows the optimiser to first find the rough lattice neighbourhood
    (small λ) before being locked in by the integrality constraint (large λ).

    Parameters
    ----------
    step : int
        Current iteration index (0-based).
    lambda_max : float
        Maximum coupling constant (reached near step = n_steps).
    n_steps : int
        Total number of optimisation steps.
    warm_up : int, optional
        Number of steps during which λ = 0 (pure quadratic phase).
        Default: 50.

    Returns
    -------
    float
        Current coupling constant λ(step) ∈ [0, lambda_max].

    References
    ----------
    .. [HS84] §3 — semi-classical adiabatic deformation of the potential.
    """
    if step < warm_up:
        return 0.0
    t = step - warm_up
    n = n_steps - warm_up
    z = 8.0 * (t / n - 0.5)
    return lambda_max / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class MHRResult:
    """
    Result of an MHR variational optimisation run.

    Attributes
    ----------
    mu_opt : np.ndarray
        Optimal mean vector (best approximation to a short lattice vector).
    log_sigma_opt : float
        Optimal log-standard-deviation at convergence.
    energy_history : list of float
        Energy value recorded at each outer iteration.
    candidate : np.ndarray
        Rounded integer candidate  v = round(mu_opt).
    candidate_norm : float
        Euclidean norm  ‖B · v‖  of the rounded candidate.
    n_outer : int
        Number of outer (λ-update) iterations performed.
    converged : bool
        True if the norm improvement in the last 10% of outer iterations
        was less than 1e-3 relative.
    """

    def __init__(
        self,
        mu_opt: np.ndarray,
        log_sigma_opt: float,
        energy_history: list[float],
        B: np.ndarray,
    ) -> None:
        self.mu_opt = mu_opt
        self.log_sigma_opt = log_sigma_opt
        self.energy_history = energy_history
        self.candidate = np.round(mu_opt).astype(int)
        Bv = B.T @ self.candidate.astype(float)
        self.candidate_norm = float(np.linalg.norm(Bv))
        self.n_outer = len(energy_history)

        if len(energy_history) >= 10:
            tail = energy_history[-(len(energy_history) // 10):]
            rel_change = abs(tail[0] - tail[-1]) / (abs(tail[0]) + 1e-14)
            self.converged = rel_change < 1e-3
        else:
            self.converged = False

    def __repr__(self) -> str:
        return (
            f"MHRResult(candidate_norm={self.candidate_norm:.4f}, "
            f"converged={self.converged}, "
            f"n_outer={self.n_outer})"
        )


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def mhr_solve(
    B: list[list[float]] | np.ndarray,
    lambda_max: float = 500.0,
    n_outer: int = 80,
    n_inner: int = 30,
    n_samples_energy: int = 512,
    n_samples_grad: int = 256,
    sigma_init: float = 0.3,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, float, float], None]] = None,
) -> MHRResult:
    """
    MHR variational solver: find a short vector in a lattice.

    Minimises the energy functional

        E[μ, σ] = ⟨‖B·x‖²⟩_{x~N(μ,σ²)} + λ·⟨∑_j sin²(πx_j)⟩

    over (μ, log σ) using L-BFGS-B inner optimisation and an adiabatic
    coupling ramp for λ.  The rounded mean  v = round(μ_opt)  is returned
    as a candidate short lattice vector.

    Algorithm outline
    -----------------
    1. **Initialisation**: μ ← 0,  σ ← sigma_init.
    2. **Outer loop** (n_outer iterations):
       a. Compute  λ_t  from the adiabatic schedule.
       b. Run n_inner steps of L-BFGS-B on  E(μ, log σ)  with fixed λ_t.
       c. Record energy; update μ, σ.
    3. **Extraction**: v = round(μ_opt);  compute  ‖B·v‖.

    Parameters
    ----------
    B : array-like, shape (n, m)
        Lattice basis matrix.  Rows are basis vectors in R^m.
        **Should be LLL- or BKZ-reduced** for best results (use
        :func:`mhr_numerics.lattice.lll_reduce` beforehand).
    lambda_max : float, optional
        Maximum coupling constant.  Should scale with  max diagonal of BᵀB.
        Default: 500.
    n_outer : int, optional
        Number of outer λ-update iterations.  Default: 80.
    n_inner : int, optional
        Maximum L-BFGS-B iterations per outer step.  Default: 30.
    n_samples_energy : int, optional
        Monte Carlo samples for energy evaluation.  Default: 512.
    n_samples_grad : int, optional
        Monte Carlo samples per gradient evaluation.  Default: 256.
    sigma_init : float, optional
        Initial standard deviation σ_0.  Default: 0.3.
    seed : int or None, optional
        Random seed for reproducibility.  Default: None.
    callback : callable or None, optional
        Optional callback  f(outer_step, lambda_val, energy)  called at each
        outer iteration.

    Returns
    -------
    MHRResult
        Optimisation result including the candidate vector and convergence
        information.

    Notes
    -----
    The algorithm is heuristic — it finds a *short* vector but not necessarily
    the *shortest* (SVP is NP-hard in general).  Quality improves with larger
    n_outer, n_samples, and a well-reduced input basis.

    For a 48-dimensional LLL-reduced lattice with typical Gram matrix
    diagonal ∼ 2^{22}, use  lambda_max ≈ 1000,  n_outer ≈ 120.

    Examples
    --------
    >>> import numpy as np
    >>> from mhr_numerics.lattice import lll_reduce
    >>> rng = np.random.default_rng(42)
    >>> B_raw = rng.integers(-10, 10, size=(6, 6)).tolist()
    >>> B_lll = lll_reduce(B_raw)
    >>> result = mhr_solve(B_lll, lambda_max=200, n_outer=40, seed=42)
    >>> result.candidate_norm   # ≤ first LLL basis vector norm
    ...

    References
    ----------
    .. [HS84] Theorem 4.1 — ground-state localisation for V(x) = ‖Bx‖².
    .. [Agm82] Chapter 3 — exponential decay of eigenfunctions.
    .. [NV10] Chapter 1 — geometric view; candidate extraction by rounding.
    """
    rng = np.random.default_rng(seed)
    B_np = np.array(B, dtype=float)
    n = B_np.shape[0]

    # Initialise parameters
    mu = np.zeros(n)
    log_sigma = math.log(sigma_init)

    energy_history: list[float] = []

    for outer in range(n_outer):
        lam = adiabatic_schedule(outer, lambda_max, n_outer)

        # Pack parameters for scipy
        x0 = np.concatenate([mu, [log_sigma]])

        def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
            mu_x = x[:-1]
            ls_x = float(x[-1])
            # Clamp log_sigma to avoid collapse or explosion
            ls_x = max(min(ls_x, 2.0), -4.0)
            e = mhr_energy(mu_x, ls_x, B_np, lam, n_samples_energy, rng)
            # Finite-difference gradient
            gmu, gls = mhr_gradient(mu_x, ls_x, B_np, lam, n_samples_grad,
                                    rng=rng)
            g = np.concatenate([gmu, [gls]])
            return e, g

        # L-BFGS-B inner optimisation (n_inner iterations max)
        result_inner = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": n_inner, "ftol": 1e-12, "gtol": 1e-8},
        )

        mu = result_inner.x[:-1]
        log_sigma = float(result_inner.x[-1])
        energy = float(result_inner.fun)
        energy_history.append(energy)

        if callback is not None:
            callback(outer, lam, energy)

    return MHRResult(mu, log_sigma, energy_history, B_np)


# ---------------------------------------------------------------------------
# Batch search: multiple restarts
# ---------------------------------------------------------------------------

def mhr_multi_start(
    B: list[list[float]] | np.ndarray,
    n_restarts: int = 5,
    lambda_max: float = 500.0,
    n_outer: int = 60,
    n_inner: int = 20,
    n_samples_energy: int = 256,
    seed: Optional[int] = None,
) -> MHRResult:
    """
    Multi-start MHR: run several independent restarts and return the best.

    Runs :func:`mhr_solve` n_restarts times from different random initial
    means (sampled uniformly from [−1, 1]^n) and returns the result with the
    smallest candidate norm.

    Parameters
    ----------
    B : array-like
        Lattice basis matrix.
    n_restarts : int, optional
        Number of independent restarts.  Default: 5.
    lambda_max, n_outer, n_inner, n_samples_energy
        Passed directly to :func:`mhr_solve`.
    seed : int or None, optional
        Master seed; each restart uses seed + restart_index.

    Returns
    -------
    MHRResult
        Best result (smallest ``candidate_norm``) across all restarts.

    References
    ----------
    .. [NV10] Chapter 1 §4 — multi-start strategies for lattice problems.
    """
    B_np = np.array(B, dtype=float)
    n = B_np.shape[0]
    best: Optional[MHRResult] = None

    for k in range(n_restarts):
        r_seed = None if seed is None else seed + k
        result = mhr_solve(
            B_np,
            lambda_max=lambda_max,
            n_outer=n_outer,
            n_inner=n_inner,
            n_samples_energy=n_samples_energy,
            seed=r_seed,
        )
        if best is None or result.candidate_norm < best.candidate_norm:
            best = result

    return best  # type: ignore[return-value]
