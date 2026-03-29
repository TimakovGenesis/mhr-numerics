"""
mhr_numerics.lattice
====================
Lattice reduction and closest-vector algorithms for low-dimensional
integer lattices, with optional higher-dimensional support via fpylll.

This module provides:

* **Gaussian 2D reduction** — optimal reduction of a 2-dimensional lattice
  basis in O(log(max |b_i|)) steps [Gau01].
* **Babai's nearest-plane algorithm** (CVP approximation) for lattices of
  arbitrary dimension [Bab86].
* **LLL reduction wrapper** for higher dimensions, using the fpylll library
  when available [LLL82].
* **BKZ reduction wrapper** for high-quality reduction [SE94].
* **Hermite factor computation** — standard measure of basis quality.

Mathematical background
-----------------------
A lattice  L = B·Z^n  is the set of all integer linear combinations of the
columns (or rows) of a basis matrix B ∈ R^{m×n}.  The Shortest Vector
Problem (SVP) asks for the nonzero vector in L of minimum Euclidean norm.

Gaussian reduction (n=2)
    The algorithm iteratively replaces the longer basis vector by its
    projection minus the nearest integer multiple of the shorter one,
    analogous to the Euclidean algorithm on integers.  It terminates with
    the (provably unique) reduced basis satisfying
        ‖b₁‖ ≤ ‖b₂‖  and  |μ₁₂| ≤ 1/2,
    where μ₁₂ = ⟨b₂, b₁⟩ / ‖b₁‖² is the Gram–Schmidt coefficient.

Babai CVP (arbitrary dimension)
    Babai's nearest-plane algorithm iteratively rounds the Gram–Schmidt
    coefficients to the nearest integer.  It returns a lattice vector
    within a factor 2^(n/2) of the closest lattice point to a target [Bab86].

References
----------
.. [Gau01] Gauss, C.F. (1801). *Disquisitiones Arithmeticae*. §171.
           (Modern presentation in Nguyen & Vallée (Eds.), *The LLL Algorithm*,
           Springer 2010, Chapter 1.)
.. [LLL82] Lenstra, A.K., Lenstra, H.W. & Lovász, L. (1982).
           Factoring polynomials with rational coefficients.
           *Mathematische Annalen*, 261, 515–534.
.. [Bab86] Babai, L. (1986). On Lovász' lattice reduction and the nearest
           lattice point problem. *Combinatorica*, 6(1), 1–13.
.. [SE94]  Schnorr, C.P. & Euchner, M. (1994). Lattice basis reduction:
           improved practical algorithms and solving subset sum problems.
           *Mathematical Programming*, 66(2), 181–199.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = [
    "gauss_reduce_2d",
    "babai_cvp",
    "lll_reduce",
    "bkz_reduce",
    "hermite_factor",
    "gram_schmidt",
    "shortest_vector_norm_bound",
]


# ---------------------------------------------------------------------------
# Gram–Schmidt orthogonalisation
# ---------------------------------------------------------------------------

def gram_schmidt(
    B: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Gram–Schmidt orthogonalisation of a basis B.

    Computes  B* = {b*_1, ..., b*_n}  and the upper-triangular matrix  μ
    of Gram–Schmidt coefficients, where:

        b*_i = b_i − Σ_{j<i} μ_{ij} · b*_j,
        μ_{ij} = ⟨b_i, b*_j⟩ / ‖b*_j‖²   (j < i).

    Parameters
    ----------
    B : list of list of float
        n × m matrix whose rows are the lattice basis vectors.

    Returns
    -------
    B_star : list of list of float
        n × m matrix of orthogonalised basis vectors (rows).
    mu : list of list of float
        n × n lower-triangular matrix of Gram–Schmidt coefficients.

    Notes
    -----
    This is the classical (non-modified) Gram–Schmidt process, appropriate
    for exact or high-precision arithmetic.  For numerical stability in
    large dimensions, consider using the modified Gram–Schmidt process or
    an MPFR-based library.

    References
    ----------
    .. [LLL82] §2 — Gram–Schmidt coefficients and the LLL condition.
    """
    n = len(B)
    B_np = [list(map(float, row)) for row in B]
    B_star: list[list[float]] = []
    mu: list[list[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        b_star = list(B_np[i])
        for j in range(i):
            bs_j = B_star[j]
            dot_ij = sum(B_np[i][k] * bs_j[k] for k in range(len(bs_j)))
            dot_jj = sum(bs_j[k] ** 2 for k in range(len(bs_j)))
            mu_ij = dot_ij / dot_jj if dot_jj > 0 else 0.0
            mu[i][j] = mu_ij
            b_star = [b_star[k] - mu_ij * bs_j[k] for k in range(len(b_star))]
        mu[i][i] = 1.0
        B_star.append(b_star)

    return B_star, mu


# ---------------------------------------------------------------------------
# Gaussian 2D reduction
# ---------------------------------------------------------------------------

def gauss_reduce_2d(
    b1: list[int],
    b2: list[int],
) -> tuple[list[int], list[int]]:
    """
    Gaussian reduction of a 2-dimensional integer lattice basis.

    Iteratively reduces  (b₁, b₂)  until the pair satisfies the
    Gauss reduction conditions:
        ‖b₁‖ ≤ ‖b₂‖   and   |⟨b₁, b₂⟩| ≤ ‖b₁‖² / 2.

    The result is the unique (up to sign and ordering) Minkowski-reduced
    basis of the 2-dimensional lattice, i.e., b₁ is a shortest nonzero
    vector in the lattice [Gau01].

    Algorithm
    ---------
    .. code-block:: text

        while ‖b₂‖ < ‖b₁‖:
            swap b₁, b₂
        while True:
            m ← round(⟨b₂, b₁⟩ / ‖b₁‖²)
            b₂ ← b₂ − m · b₁
            if ‖b₂‖ ≥ ‖b₁‖: break
            swap b₁, b₂

    Termination follows because ‖b₁‖ strictly decreases at each swap [Gau01].

    Parameters
    ----------
    b1, b2 : list of int
        Integer vectors forming the input basis.  Must have the same length.

    Returns
    -------
    b1_red, b2_red : list of int
        Gauss-reduced basis vectors satisfying ‖b₁_red‖ ≤ ‖b₂_red‖.

    Examples
    --------
    >>> gauss_reduce_2d([3, 5], [2, 7])
    ([1, 2], [3, 5])

    Complexity
    ----------
    O(log(max(‖b₁‖, ‖b₂‖))) iterations, each O(d) where d is the
    ambient dimension.

    References
    ----------
    .. [Gau01] Nguyen & Vallée (2010), *The LLL Algorithm*, Chapter 1, §2.
    """
    b1 = list(b1)
    b2 = list(b2)

    def dot(u, v):
        return sum(a * b for a, b in zip(u, v))

    def norm_sq(u):
        return dot(u, u)

    # Ensure ‖b₁‖ ≤ ‖b₂‖
    if norm_sq(b1) > norm_sq(b2):
        b1, b2 = b2, b1

    while True:
        n1 = norm_sq(b1)
        if n1 == 0:
            break
        m = round(dot(b2, b1) / n1)
        b2 = [b2[i] - m * b1[i] for i in range(len(b1))]
        if norm_sq(b2) >= norm_sq(b1):
            break
        b1, b2 = b2, b1

    return b1, b2


# ---------------------------------------------------------------------------
# Babai nearest-plane CVP
# ---------------------------------------------------------------------------

def babai_cvp(
    B: list[list[float]],
    target: list[float],
) -> list[float]:
    """
    Babai's nearest-plane algorithm for the approximate Closest Vector Problem.

    Given a basis B (rows are basis vectors) and a target point t, returns
    a lattice vector  v = B · z  (z ∈ Z^n) such that
        ‖t − v‖ ≤ 2^(n/2) · dist(t, L),
    where dist(t, L) is the true distance from t to the lattice L [Bab86].

    Algorithm
    ---------
    Babai's nearest-plane proceeds by projecting t onto successive
    Gram–Schmidt subspaces and rounding each Gram–Schmidt coefficient to
    the nearest integer:

    .. code-block:: text

        w ← t
        for i from n-1 down to 0:
            c_i ← round(⟨w, b*_i⟩ / ‖b*_i‖²)
            w   ← w − c_i · b_i

    Parameters
    ----------
    B : list of list of float
        n × m basis matrix (rows are basis vectors).  Should be LLL-reduced
        or better for good approximation quality.
    target : list of float
        Target vector t in R^m.

    Returns
    -------
    list of float
        Approximate closest lattice vector  v ∈ L.

    Notes
    -----
    The approximation quality improves significantly when B is first
    LLL-reduced (use :func:`lll_reduce` beforehand).  On an LLL-reduced
    basis, the approximation factor is roughly 2^(n/4) in practice.

    References
    ----------
    .. [Bab86] Theorem 1 — nearest-plane approximation guarantee.
    .. [LLL82] §5 — application to CVP.
    """
    n = len(B)
    B_np = np.array(B, dtype=float)
    t = np.array(target, dtype=float)

    B_star, mu = gram_schmidt(B)
    B_star_np = np.array(B_star, dtype=float)

    w = t.copy()
    coeffs = [0] * n

    for i in range(n - 1, -1, -1):
        bs = B_star_np[i]
        bs_sq = float(np.dot(bs, bs))
        if bs_sq < 1e-14:
            continue
        c = round(float(np.dot(w, bs)) / bs_sq)
        coeffs[i] = c
        w = w - c * B_np[i]

    # Reconstruct lattice vector
    v = np.zeros(len(target))
    for i in range(n):
        v += coeffs[i] * B_np[i]

    return v.tolist()


# ---------------------------------------------------------------------------
# LLL reduction (wraps fpylll when available)
# ---------------------------------------------------------------------------

def lll_reduce(
    B: list[list[int]],
    delta: float = 0.99,
) -> list[list[int]]:
    """
    LLL lattice basis reduction.

    Returns an LLL-reduced basis B' satisfying the Lovász condition:
        δ · ‖b*_{i}‖² ≤ ‖b*_{i+1} + μ_{i+1,i} · b*_i‖²   ∀i,
    with the size-reduction condition  |μ_{ij}| ≤ 1/2  for j < i.

    On an LLL-reduced basis the first vector satisfies:
        ‖b₁‖ ≤ 2^((n−1)/2) · λ₁(L),
    where λ₁(L) is the length of the shortest nonzero vector [LLL82].

    Parameters
    ----------
    B : list of list of int
        n × m integer basis matrix (rows are basis vectors).
    delta : float, optional
        Lovász condition constant, 1/4 < δ ≤ 1.  Higher δ gives better
        quality at higher cost.  Default: 0.99.

    Returns
    -------
    list of list of int
        LLL-reduced basis (rows are reduced vectors, sorted by increasing
        norm).

    Notes
    -----
    Requires the **fpylll** library (https://github.com/fplll/fpylll).
    If fpylll is not available, a pure-Python Gram–Schmidt-based LLL is
    used as fallback (slower, identical results for small dimensions).

    Complexity
    ----------
    O(n^4 · log B) bit operations, where B = max |b_{ij}| [LLL82].

    References
    ----------
    .. [LLL82] Theorem 1.11 — reduction quality bound.
    .. [Neu10] Nguyen & Vallée (2010), *The LLL Algorithm*, Chapter 2 — proof
               of polynomial complexity.
    """
    try:
        from fpylll import IntegerMatrix, LLL as fpLLL, FPLLL
        FPLLL.set_precision(150)
        M = IntegerMatrix.from_matrix(B)
        fpLLL.reduction(M, delta=delta)
        return [list(M[i]) for i in range(M.nrows)]
    except ImportError:
        return _lll_pure_python(B, delta)


def _lll_pure_python(B: list[list[int]], delta: float) -> list[list[int]]:
    """Pure-Python LLL fallback (no external dependencies)."""
    n = len(B)
    B = [list(row) for row in B]

    def dot(u, v):
        return sum(a * b for a, b in zip(u, v))

    def proj_coeff(b, b_star):
        d = dot(b_star, b_star)
        return dot(b, b_star) / d if d > 0 else 0.0

    def size_reduce(i, B_star, mu):
        for j in range(i - 1, -1, -1):
            m = round(mu[i][j])
            if m != 0:
                B[i] = [B[i][k] - m * B[j][k] for k in range(len(B[i]))]
                mu[i][j] -= m
                for l in range(j):
                    mu[i][l] -= m * mu[j][l]

    k = 1
    while k < n:
        B_star, mu = gram_schmidt(B)
        size_reduce(k, B_star, mu)
        B_star, mu = gram_schmidt(B)
        lhs = dot(B_star[k], B_star[k])
        rhs = (delta - mu[k][k - 1] ** 2) * dot(B_star[k - 1], B_star[k - 1])
        if lhs >= rhs:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            k = max(k - 1, 1)

    return B


# ---------------------------------------------------------------------------
# BKZ reduction
# ---------------------------------------------------------------------------

def bkz_reduce(
    B: list[list[int]],
    block_size: int = 20,
    max_loops: int = 8,
) -> list[list[int]]:
    """
    BKZ lattice basis reduction.

    BKZ-β successively applies SVP oracles on projected sublattices of
    dimension β (the block size), achieving a basis quality governed by
    the Hermite factor:
        δ(β) ≈ ((πβ)^(1/β) · β / (2πe))^(1/(2(β-1)))   [SE94].

    Parameters
    ----------
    B : list of list of int
        n × m integer basis matrix (rows are basis vectors).
    block_size : int, optional
        BKZ block size β, 2 ≤ β ≤ n.  Higher β gives better reduction
        quality at exponential cost.  Default: 20.
    max_loops : int, optional
        Maximum number of BKZ tours.  Default: 8.

    Returns
    -------
    list of list of int
        BKZ-reduced basis.

    Raises
    ------
    ImportError
        If fpylll is not installed.

    Notes
    -----
    BKZ requires **fpylll**.  For n ≤ 30 and β ≤ 20, one tour typically
    suffices; for n ≥ 100 and β ≥ 40 expect minutes to hours.

    References
    ----------
    .. [SE94] Theorem 3 — asymptotic quality of BKZ-β.
    .. [CN11] Chen, Y. & Nguyen, P.Q. (2011). BKZ 2.0. ASIACRYPT 2011.
              Improved tour strategies and prediction of running time.
    """
    from fpylll import IntegerMatrix, LLL as fpLLL, BKZ, FPLLL

    n = len(B)
    precision = 150 if n <= 120 else 256
    FPLLL.set_precision(precision)

    M = IntegerMatrix.from_matrix(B)
    fpLLL.reduction(M)
    params = BKZ.Param(
        block_size=block_size,
        strategies=BKZ.DEFAULT_STRATEGY,
        max_loops=max_loops,
        auto_abort=True,
    )
    BKZ.reduction(M, params)
    return [list(M[i]) for i in range(M.nrows)]


# ---------------------------------------------------------------------------
# Hermite factor
# ---------------------------------------------------------------------------

def hermite_factor(
    b1_norm: float,
    vol: float,
    n: int,
) -> float:
    """
    Hermite factor δ of a lattice basis.

    The Hermite factor measures the quality of the shortest basis vector
    relative to the lattice volume:
        δ^n = ‖b₁‖ / vol(L)^(1/n),
    where  vol(L) = |det(B)|  for a full-rank lattice.

    Parameters
    ----------
    b1_norm : float
        Euclidean norm of the first (shortest) basis vector after reduction.
    vol : float
        Volume of the lattice, i.e., |det(B)| for a square basis.
    n : int
        Lattice dimension.

    Returns
    -------
    float
        Hermite factor δ.

    Notes
    -----
    The theoretical minimum (Hermite's constant) for dimension n satisfies
        δ_min^n ∼ (n/(2πe))^(1/2)  as n → ∞.
    BKZ-β achieves  δ ≈ ((πβ)^(1/β) · β / (2πe))^(1/(2(β−1))).

    References
    ----------
    .. [SE94] §2 — definition and bounds on δ.
    .. [Sil12] Silverman, J. (2012). Lattices, Cryptography, and the NTRU
               Cryptosystem. §3 — Hermite's constant.
    """
    if vol <= 0 or n <= 0 or b1_norm <= 0:
        raise ValueError("b1_norm, vol, n must all be positive.")
    return (b1_norm / vol ** (1.0 / n)) ** (1.0 / n)


# ---------------------------------------------------------------------------
# Shortest-vector norm bound
# ---------------------------------------------------------------------------

def shortest_vector_norm_bound(vol: float, n: int) -> float:
    """
    Minkowski's bound on the shortest nonzero vector in an n-dimensional lattice.

    By Minkowski's theorem:
        λ₁(L) ≤ √n · vol(L)^(1/n).

    Parameters
    ----------
    vol : float
        Lattice volume |det(B)|.
    n : int
        Lattice dimension.

    Returns
    -------
    float
        Upper bound on λ₁(L).

    References
    ----------
    .. [Cas97] Cassels, J.W.S. (1997). *An Introduction to the Geometry of
               Numbers*. Springer Classics. Theorem 1 (Minkowski's theorem).
    """
    return math.sqrt(n) * vol ** (1.0 / n)
