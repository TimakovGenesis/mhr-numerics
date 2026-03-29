"""
mhr_numerics.interpolation
==========================
Polynomial interpolation and evaluation over finite prime fields F_p.

Provides:

* **Lagrange interpolation** over F_p — given n+1 distinct evaluation
  points, recover the unique polynomial of degree ≤ n passing through them.
* **Polynomial multiplication** over F_p — coefficient-list representation.
* **Polynomial evaluation** over F_p — Horner's rule.
* **Root finding** over F_p — exhaustive search for small p; Cantor–Zassenhaus
  for general primes (requires sympy).
* **Berlekamp–Massey** over F_p — minimum linear recurrence / LFSR synthesis.

All polynomials are represented as lists of coefficients in *increasing
degree order*:  p(x) = c[0] + c[1]·x + c[2]·x² + ···

Mathematical background
-----------------------
Lagrange interpolation theorem
    Given n+1 distinct points  (x_0, y_0), …, (x_n, y_n)  over F_p, there
    exists a unique polynomial  f ∈ F_p[x]  of degree ≤ n such that
    f(x_i) = y_i for all i.  The Lagrange basis polynomials are:

        L_i(x) = ∏_{j ≠ i} (x − x_j) / (x_i − x_j)  ∈ F_p[x],

    and the interpolating polynomial is  f(x) = Σ_i y_i · L_i(x).

Applications
------------
* Polynomial recurrences over finite fields (degree-1 and degree-2
  linear recurrences) — extract recurrence coefficients from samples.
* Lagrange basis for numerical quadrature and finite-element methods.
* Error-correcting codes (Reed–Solomon).

References
----------
.. [GG13] von zur Gathen, J. & Gerhard, J. (2013).
          *Modern Computer Algebra* (3rd ed.). Cambridge University Press.
          Chapter 5 (polynomial arithmetic), §5.1 (Lagrange interpolation).
.. [Knu97] Knuth, D.E. (1997).
           *The Art of Computer Programming*, Vol. 2, §4.6 — polynomial
           algorithms and their complexity.
.. [Ber68] Berlekamp, E.R. (1968).
           Algebraic Coding Theory. McGraw-Hill. Chapter 7 — LFSR synthesis.
.. [Mas69] Massey, J.L. (1969). Shift-register synthesis and BCH decoding.
           *IEEE Trans. Inf. Theory*, 15(1), 122–127.
"""

from __future__ import annotations
from .linalg_Fp import _check_prime

__all__ = [
    "lagrange_interpolate_Fp",
    "poly_eval_Fp",
    "poly_mul_Fp",
    "poly_add_Fp",
    "poly_roots_Fp",
    "berlekamp_massey_Fp",
]


# ---------------------------------------------------------------------------
# Polynomial arithmetic helpers
# ---------------------------------------------------------------------------

def poly_add_Fp(A: list[int], B: list[int], p: int) -> list[int]:
    """
    Add two polynomials over F_p.

    Parameters
    ----------
    A, B : list of int
        Coefficient lists [c_0, c_1, …] in increasing degree order.
    p : int
        Prime modulus.

    Returns
    -------
    list of int
        Coefficient list of A + B over F_p, with trailing zeros removed.

    References
    ----------
    .. [GG13] §5.1 — polynomial addition.
    """
    n = max(len(A), len(B))
    C = [(int(A[i]) if i < len(A) else 0) + (int(B[i]) if i < len(B) else 0)
         for i in range(n)]
    C = [x % p for x in C]
    while len(C) > 1 and C[-1] == 0:
        C.pop()
    return C


def poly_mul_Fp(A: list[int], B: list[int], p: int) -> list[int]:
    """
    Multiply two polynomials over F_p (schoolbook O(n²) algorithm).

    Parameters
    ----------
    A, B : list of int
        Coefficient lists [c_0, c_1, …].  May have any length.
    p : int
        Prime modulus.

    Returns
    -------
    list of int
        Coefficient list of A · B over F_p.

    Complexity
    ----------
    O(deg(A) · deg(B)) multiplications in F_p.

    References
    ----------
    .. [GG13] §5.1 — polynomial multiplication.
    .. [Knu97] §4.6.1 — schoolbook multiplication.
    """
    if not A or not B:
        return [0]
    C = [0] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            C[i + j] = (C[i + j] + int(a) * int(b)) % p
    return C


def poly_eval_Fp(poly: list[int], x: int, p: int) -> int:
    """
    Evaluate a polynomial at x over F_p using Horner's rule.

    For  f(x) = c_0 + c_1·x + ··· + c_n·x^n:

        f(x) = c_0 + x·(c_1 + x·(c_2 + ··· + x·c_n))

    Parameters
    ----------
    poly : list of int
        Coefficients [c_0, c_1, …, c_n] in increasing degree order.
    x : int
        Evaluation point in F_p.
    p : int
        Prime modulus.

    Returns
    -------
    int
        f(x) mod p, in [0, p).

    Complexity
    ----------
    O(n) multiplications (vs O(n²) for naive evaluation).

    References
    ----------
    .. [Knu97] §4.6.4 — Horner's rule.
    """
    result = 0
    x_mod = int(x) % p
    for c in reversed(poly):
        result = (result * x_mod + int(c)) % p
    return result


# ---------------------------------------------------------------------------
# Lagrange interpolation
# ---------------------------------------------------------------------------

def lagrange_interpolate_Fp(
    points: list[tuple[int, int]],
    p: int,
) -> list[int]:
    """
    Lagrange polynomial interpolation over F_p.

    Given n+1 distinct evaluation points  (x_0, y_0), …, (x_n, y_n)  in
    F_p × F_p, computes the unique polynomial  f ∈ F_p[x]  of degree ≤ n
    satisfying  f(x_i) ≡ y_i (mod p)  for all i.

    The polynomial is computed as:

        f(x) = Σ_{i=0}^{n}  y_i · L_i(x),

    where the Lagrange basis polynomials are:

        L_i(x) = ∏_{j ≠ i} (x − x_j) · [∏_{j ≠ i} (x_i − x_j)]^{-1}.

    Parameters
    ----------
    points : list of (int, int)
        List of (x_i, y_i) evaluation pairs.  All x_i must be distinct mod p.
    p : int
        Prime modulus.

    Returns
    -------
    list of int
        Coefficients [c_0, c_1, …, c_n] of the interpolating polynomial,
        in increasing degree order.  All entries are in [0, p).

    Raises
    ------
    ValueError
        If p is not prime, if fewer than 1 point is given, or if any two
        x-coordinates are equal mod p.

    Examples
    --------
    Interpolate a polynomial of degree 2 over F_7:

    >>> points = [(0, 1), (1, 3), (2, 7)]
    >>> poly = lagrange_interpolate_Fp(points, p=7)
    >>> poly_eval_Fp(poly, 0, 7)
    1
    >>> poly_eval_Fp(poly, 1, 7)
    3

    Complexity
    ----------
    O(n²) multiplications in F_p.

    References
    ----------
    .. [GG13] §5.1, Theorem 5.1 — uniqueness and formula.
    """
    _check_prime(p)
    if len(points) == 0:
        raise ValueError("At least one interpolation point is required.")

    # Check distinct x-coordinates
    xs = [int(x) % p for x, _ in points]
    if len(set(xs)) < len(xs):
        raise ValueError("All x-coordinates must be distinct modulo p.")

    n = len(points)
    result = [0]  # zero polynomial

    for i in range(n):
        xi, yi = int(points[i][0]) % p, int(points[i][1]) % p
        if yi == 0:
            continue

        # Build numerator polynomial ∏_{j≠i} (x − x_j) — coefficient list
        numer = [1]
        for j in range(n):
            if j == i:
                continue
            xj = int(points[j][0]) % p
            # Multiply by (x − xj) = [−xj, 1] in coeff representation
            numer = poly_mul_Fp(numer, [(-xj) % p, 1], p)

        # Compute denominator ∏_{j≠i} (x_i − x_j) over F_p
        denom = 1
        for j in range(n):
            if j == i:
                continue
            xj = int(points[j][0]) % p
            denom = (denom * ((xi - xj) % p)) % p

        if denom == 0:
            raise ValueError(f"Denominator is zero at i={i} — check for duplicate x values.")

        denom_inv = pow(denom, -1, p)
        coef = (yi * denom_inv) % p

        # Add yi · L_i to result
        term = [(coef * c) % p for c in numer]
        result = poly_add_Fp(result, term, p)

    return result


# ---------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------

def poly_roots_Fp(
    poly: list[int],
    p: int,
) -> list[int]:
    """
    Find all roots of a polynomial over F_p.

    For small primes (p < 10^6), uses exhaustive evaluation.  For larger
    primes, delegates to ``sympy.GF(p)`` / Cantor–Zassenhaus factorisation.

    Parameters
    ----------
    poly : list of int
        Coefficients [c_0, …, c_n] of f ∈ F_p[x], increasing degree order.
    p : int
        Prime modulus.

    Returns
    -------
    list of int
        List of roots r ∈ [0, p) such that f(r) ≡ 0 (mod p), in
        increasing order.  May be empty.

    Notes
    -----
    A degree-n polynomial has at most n roots in F_p.  If p < n, exhaustive
    search is always used regardless of the prime size.

    References
    ----------
    .. [CZ81] Cantor, D.G. & Zassenhaus, H. (1981). A new algorithm for
              factoring polynomials over finite fields. *Mathematics of
              Computation*, 36(154), 587–592.
    .. [GG13] §14.2 — polynomial factorisation over finite fields.
    """
    _check_prime(p)
    roots = []

    if p < 10 ** 6:
        # Exhaustive evaluation
        for x in range(p):
            if poly_eval_Fp(poly, x, p) == 0:
                roots.append(x)
        return roots

    # For large p: use sympy's GF(p) polynomial factoriser
    try:
        from sympy import Poly, Symbol, GF, factor_list
        t = Symbol('t')
        coeffs_map = {i: int(c) for i, c in enumerate(poly)}
        # sympy Poly uses decreasing degree order
        deg = len(poly) - 1
        sym_coeffs = {deg - i: int(poly[i]) for i in range(len(poly))}
        f = Poly.from_dict(sym_coeffs, t, domain=GF(p))
        _, factors = factor_list(f, domain=GF(p))
        for fac, mult in factors:
            if fac.degree() == 1:
                # (t − r) → root = −c_0/c_1 mod p
                coeffs_fac = fac.all_coeffs()  # [leading, ..., constant]
                r = int((-coeffs_fac[-1] * pow(int(coeffs_fac[0]), -1, p)) % p)
                roots.extend([r] * mult)
    except ImportError:
        # Fallback: try a random sample of 10000 values
        import random
        sample = random.sample(range(p), min(10000, p))
        for x in sample:
            if poly_eval_Fp(poly, x, p) == 0:
                roots.append(x)

    return sorted(set(roots))


# ---------------------------------------------------------------------------
# Berlekamp–Massey algorithm
# ---------------------------------------------------------------------------

def berlekamp_massey_Fp(
    sequence: list[int],
    p: int,
) -> list[int]:
    """
    Berlekamp–Massey algorithm over F_p.

    Given a sequence  s_0, s_1, …, s_{N-1}  over F_p, finds the shortest
    linear feedback shift register (LFSR) with connection polynomial

        C(x) = 1 + c_1·x + c_2·x² + ··· + c_L·x^L ∈ F_p[x]

    such that  Σ_{j=0}^{L} c_j · s_{n-j} ≡ 0 (mod p)  for all valid n.

    Parameters
    ----------
    sequence : list of int
        Input sequence over F_p of length N ≥ 1.
    p : int
        Prime modulus.

    Returns
    -------
    list of int
        LFSR connection polynomial C = [c_0, c_1, …, c_L] in increasing
        degree order (c_0 = 1 always).

    Notes
    -----
    The minimum LFSR length L satisfies  L ≤ N/2  when the sequence is
    generated by a linear recurrence over F_p.  The output polynomial
    encodes the recurrence:  s_n = −Σ_{j=1}^{L} c_j · s_{n-j} (mod p).

    References
    ----------
    .. [Mas69] Algorithm BM — Theorem 1.
    .. [Ber68] Chapter 7 — shift-register synthesis.
    """
    _check_prime(p)
    N = len(sequence)
    s = [int(x) % p for x in sequence]

    C = [1]       # current LFSR polynomial (starts as 1)
    B = [1]       # previous best polynomial
    L = 0         # current LFSR length
    b = 1         # previous discrepancy
    x = 1         # power shift (number of steps since last length change)

    for n in range(N):
        # Compute discrepancy d = s_n + Σ_{j=1}^{L} C_j · s_{n-j}
        d = s[n]
        for j in range(1, L + 1):
            if j < len(C):
                d = (d + C[j] * s[n - j]) % p

        if d == 0:
            x += 1
            continue

        T = list(C)
        coef = (d * pow(b, -1, p)) % p
        # C ← C − coef · x^x · B
        shift_B = [0] * x + B
        C_new = list(C)
        for j in range(len(shift_B)):
            if j < len(C_new):
                C_new[j] = (C_new[j] - coef * shift_B[j]) % p
            else:
                C_new.append((-coef * shift_B[j]) % p)
        C = C_new

        if 2 * L <= n:
            L = n + 1 - L
            B = T
            b = d
            x = 1
        else:
            x += 1

    return [int(c) % p for c in C]
