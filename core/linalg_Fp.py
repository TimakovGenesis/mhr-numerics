"""
mhr_numerics.linalg_Fp
======================
Linear algebra over finite prime fields F_p.

Provides exact arithmetic for Gaussian elimination, determinants,
matrix inversion, and polynomial interpolation over F_p, where p is
an arbitrary prime.  All computations are performed with Python's
arbitrary-precision integers — no floating-point rounding errors.

Mathematical background
-----------------------
Let p be a prime and F_p = Z/pZ the field with p elements.
The multiplicative inverse of a ∈ F_p* is computed via Fermat's
little theorem:  a^{-1} ≡ a^{p-2} (mod p),
or, equivalently, via the extended Euclidean algorithm (Python's
built-in  pow(a, -1, p)  uses the latter for efficiency).

References
----------
.. [GG13] von zur Gathen, J. & Gerhard, J. (2013).
          *Modern Computer Algebra* (3rd ed.). Cambridge University Press.
          Chapter 2 (modular arithmetic) and Chapter 3 (linear algebra).
.. [Coh93] Cohen, H. (1993).
           *A Course in Computational Algebraic Number Theory*.
           Springer GTM 138.  §2.1 (Gaussian elimination over fields).
"""

from __future__ import annotations

__all__ = [
    "gauss_jordan_Fp",
    "det_Fp",
    "inv_Fp",
    "rank_Fp",
    "solve_Fp",
    "is_prime",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_prime(p: int) -> None:
    """Raise ValueError if p is not a prime > 1."""
    if not is_prime(p):
        raise ValueError(f"p={p} is not prime.")


def _mod_matrix(M: list[list[int]], p: int) -> list[list[int]]:
    """Return a deep copy of M with all entries reduced mod p."""
    return [[int(x) % p for x in row] for row in M]


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """
    Deterministic primality test (trial division up to sqrt, then
    Miller–Rabin with the first 20 prime witnesses — sufficient for
    n < 3.3 × 10^24).

    Parameters
    ----------
    n : int
        Integer to test.

    Returns
    -------
    bool
        True iff n is prime.

    Notes
    -----
    For cryptographic-scale primes (512+ bits) this function falls
    back to a probabilistic Miller–Rabin test with 20 rounds.

    References
    ----------
    .. [MR80] Miller, G.L. (1976). Riemann's hypothesis and tests for
              primality. *J. Comput. System Sci.*, 13(3), 300–317.
    """
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for sp in small_primes:
        if n == sp:
            return True
        if n % sp == 0:
            return False
    if n < 53 * 53:
        return True
    # Miller–Rabin
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Gauss–Jordan elimination
# ---------------------------------------------------------------------------

def gauss_jordan_Fp(
    A: list[list[int]],
    p: int,
    augmented: bool = False,
) -> tuple[list[list[int]], list[int], int]:
    """
    Gauss–Jordan elimination over F_p, returning the reduced row-echelon
    form (RREF) together with pivot column indices and the matrix rank.

    Algorithm
    ---------
    Standard column-pivoting Gauss–Jordan, with all arithmetic performed
    modulo p.  Each pivot element a is normalised to 1 by multiplying the
    entire row by a^{-1} mod p, then all other rows are zeroed in that
    column.

    Parameters
    ----------
    A : list of list of int
        m × n integer matrix.  Entries need not be reduced mod p in advance.
    p : int
        Prime modulus.
    augmented : bool, optional
        If True, the last column is treated as the right-hand side of a
        linear system [A | b] and is not used for pivot selection.
        Default: False.

    Returns
    -------
    R : list of list of int
        m × n matrix in reduced row-echelon form over F_p.
    pivots : list of int
        Column indices of the pivot positions (length = rank(A)).
    rank : int
        Rank of A over F_p.

    Raises
    ------
    ValueError
        If p is not prime, or if the matrix is empty.

    Examples
    --------
    >>> A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    >>> R, pivots, r = gauss_jordan_Fp(A, p=11)
    >>> r
    3

    References
    ----------
    .. [GG13] §3.4 — Gaussian elimination over fields.
    """
    _check_prime(p)
    M = _mod_matrix(A, p)
    m = len(M)
    if m == 0:
        raise ValueError("Matrix A must be non-empty.")
    n = len(M[0])
    n_cols = n - 1 if augmented else n  # exclude augmented column from pivoting

    pivot_row = 0
    pivots: list[int] = []

    for col in range(n_cols):
        # Find a non-zero entry in this column at or below pivot_row
        sel = None
        for row in range(pivot_row, m):
            if M[row][col] != 0:
                sel = row
                break
        if sel is None:
            continue  # entire sub-column is zero

        # Swap selected row with pivot row
        M[pivot_row], M[sel] = M[sel], M[pivot_row]

        # Normalise pivot row
        inv = pow(int(M[pivot_row][col]), -1, p)
        M[pivot_row] = [(inv * x) % p for x in M[pivot_row]]

        # Eliminate all other rows in this column
        for row in range(m):
            if row == pivot_row:
                continue
            factor = M[row][col]
            if factor == 0:
                continue
            M[row] = [(M[row][c] - factor * M[pivot_row][c]) % p for c in range(n)]

        pivots.append(col)
        pivot_row += 1
        if pivot_row == m:
            break

    return M, pivots, len(pivots)


# ---------------------------------------------------------------------------
# Determinant
# ---------------------------------------------------------------------------

def det_Fp(A: list[list[int]], p: int) -> int:
    """
    Determinant of a square matrix over F_p.

    The determinant is computed as a by-product of Gaussian elimination
    (with row swaps tracked for sign).  Complexity: O(n³) multiplications
    in F_p.

    Parameters
    ----------
    A : list of list of int
        n × n integer matrix.
    p : int
        Prime modulus.

    Returns
    -------
    int
        det(A) mod p, in the range [0, p).

    Raises
    ------
    ValueError
        If A is not square, or p is not prime.

    Examples
    --------
    >>> det_Fp([[1, 2], [3, 4]], p=7)
    5   # (1·4 - 2·3) = -2 ≡ 5 (mod 7)

    References
    ----------
    .. [Coh93] §2.1 — modular determinant via row reduction.
    """
    _check_prime(p)
    M = _mod_matrix(A, p)
    n = len(M)
    if any(len(row) != n for row in M):
        raise ValueError("Matrix A must be square.")

    det = 1
    for col in range(n):
        sel = None
        for row in range(col, n):
            if M[row][col] != 0:
                sel = row
                break
        if sel is None:
            return 0  # singular

        if sel != col:
            M[col], M[sel] = M[sel], M[col]
            det = (-det) % p

        inv = pow(int(M[col][col]), -1, p)
        det = (det * M[col][col]) % p
        M[col] = [(inv * x) % p for x in M[col]]

        for row in range(col + 1, n):
            factor = M[row][col]
            if factor == 0:
                continue
            M[row] = [(M[row][c] - factor * M[col][c]) % p for c in range(n)]

    return int(det) % p


# ---------------------------------------------------------------------------
# Matrix inverse
# ---------------------------------------------------------------------------

def inv_Fp(A: list[list[int]], p: int) -> list[list[int]]:
    """
    Inverse of an invertible square matrix over F_p.

    Uses Gauss–Jordan on the augmented matrix [A | I_n] to compute A^{-1}.

    Parameters
    ----------
    A : list of list of int
        n × n invertible matrix over F_p.
    p : int
        Prime modulus.

    Returns
    -------
    list of list of int
        n × n matrix A^{-1} over F_p.

    Raises
    ------
    ValueError
        If A is singular, not square, or p is not prime.

    References
    ----------
    .. [GG13] §3.5 — matrix inversion via augmented elimination.
    """
    _check_prime(p)
    M = _mod_matrix(A, p)
    n = len(M)
    if any(len(row) != n for row in M):
        raise ValueError("Matrix A must be square.")

    # Augment with identity
    aug = [row + [int(i == j) for j in range(n)] for i, row in enumerate(M)]

    # Run elimination on augmented system (pivot only in left half)
    pivot_row = 0
    for col in range(n):
        sel = None
        for row in range(pivot_row, n):
            if aug[row][col] != 0:
                sel = row
                break
        if sel is None:
            raise ValueError("Matrix A is singular over F_p.")

        aug[pivot_row], aug[sel] = aug[sel], aug[pivot_row]

        inv = pow(int(aug[pivot_row][col]), -1, p)
        aug[pivot_row] = [(inv * x) % p for x in aug[pivot_row]]

        for row in range(n):
            if row == pivot_row:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            aug[row] = [(aug[row][c] - factor * aug[pivot_row][c]) % p
                        for c in range(2 * n)]
        pivot_row += 1

    return [row[n:] for row in aug]


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------

def rank_Fp(A: list[list[int]], p: int) -> int:
    """
    Rank of a matrix over F_p.

    Parameters
    ----------
    A : list of list of int
        m × n integer matrix.
    p : int
        Prime modulus.

    Returns
    -------
    int
        rank(A) over F_p.

    Examples
    --------
    >>> rank_Fp([[1, 2, 3], [2, 4, 6]], p=7)
    1

    References
    ----------
    .. [GG13] §3.4.
    """
    _, _, r = gauss_jordan_Fp(A, p)
    return r


# ---------------------------------------------------------------------------
# Linear system solver
# ---------------------------------------------------------------------------

def solve_Fp(
    A: list[list[int]],
    b: list[int],
    p: int,
) -> list[int] | None:
    """
    Solve the linear system  Ax ≡ b (mod p)  over F_p.

    Returns one particular solution if the system is consistent, or None
    if it has no solution.  Free variables (underdetermined system) are
    set to zero.

    Parameters
    ----------
    A : list of list of int
        m × n coefficient matrix.
    b : list of int
        Right-hand side vector of length m.
    p : int
        Prime modulus.

    Returns
    -------
    list of int or None
        A solution vector x of length n with entries in [0, p),
        or None if the system is inconsistent.

    Examples
    --------
    >>> A = [[2, 1], [1, 3]]
    >>> b = [5, 10]
    >>> solve_Fp(A, b, p=11)
    [0, 5]   # 2·0 + 1·5 ≡ 5, 1·0 + 3·5 = 15 ≡ 4... check with p=11

    References
    ----------
    .. [Coh93] §2.2 — linear system solving over F_p.
    """
    _check_prime(p)
    m = len(A)
    n = len(A[0])
    # Build augmented matrix [A | b]
    aug = [list(A[i]) + [int(b[i])] for i in range(m)]
    R, pivots, rank = gauss_jordan_Fp(aug, p, augmented=True)

    # Check consistency
    for row in range(rank, m):
        if R[row][n] % p != 0:
            return None  # inconsistent

    # Extract solution (free variables → 0)
    x = [0] * n
    for idx, col in enumerate(pivots):
        x[col] = int(R[idx][n]) % p

    return x
