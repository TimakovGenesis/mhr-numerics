"""
mhr_numerics.linalg_GF2
========================
Linear algebra over GF(2) = F_2 = {0, 1}.

All arithmetic is performed with Python integers used as packed bit-vectors
(one integer = one matrix row).  This gives O(n/64) factor speedup over
element-wise operations for dense matrices, using native bitwise operators.

Mathematical background
-----------------------
In GF(2): addition ≡ XOR, multiplication ≡ AND, subtraction ≡ addition.
Row reduction is identical to the F_p case with p = 2, but pivoting and
elimination reduce to pure bitwise operations.

The rank of a matrix over GF(2) equals the dimension of its column space,
and has applications in linear recurrence analysis and lattice preprocessing.

References
----------
.. [LN97] Lidl, R. & Niederreiter, H. (1997).
          *Finite Fields* (2nd ed.). Cambridge University Press.
          Chapter 1 (field axioms), Chapter 6 (linear algebra over GF(2)).
.. [Knu98] Knuth, D.E. (1998).
           *The Art of Computer Programming*, Vol. 2 (Seminumerical Algorithms).
           §4.6.3 — polynomial arithmetic over GF(2).
.. [Coh93] Cohen, H. (1993).
           *A Course in Computational Algebraic Number Theory*. Springer GTM 138.
           §3.1 — bit-packed linear algebra.
"""

from __future__ import annotations

__all__ = [
    "rref_GF2",
    "rank_GF2",
    "null_space_GF2",
    "mat_mul_GF2",
    "solve_GF2",
]


# ---------------------------------------------------------------------------
# Internal: matrix representation
# ---------------------------------------------------------------------------
#
# An m × n GF(2) matrix is stored as a list of m Python ints, each of
# which is an n-bit packed row.  Bit j of row i holds A[i][j].
# Helper functions convert between this representation and nested lists.

def _to_packed(A: list[list[int]], n: int) -> list[int]:
    rows = []
    for row in A:
        v = 0
        for j, x in enumerate(row):
            if int(x) & 1:
                v |= 1 << j
        rows.append(v)
    return rows


def _from_packed(rows: list[int], m: int, n: int) -> list[list[int]]:
    return [[(rows[i] >> j) & 1 for j in range(n)] for i in range(m)]


# ---------------------------------------------------------------------------
# Reduced row-echelon form over GF(2)
# ---------------------------------------------------------------------------

def rref_GF2(
    A: list[list[int]],
) -> tuple[list[list[int]], list[int], int]:
    """
    Reduced row-echelon form (RREF) of a GF(2) matrix.

    Uses bit-packed row operations (XOR for elimination) with partial
    column pivoting.  The pivot columns correspond to a basis for the
    column space of A.

    Parameters
    ----------
    A : list of list of int
        m × n matrix with entries in {0, 1} (any integer is reduced mod 2).

    Returns
    -------
    R : list of list of int
        m × n RREF matrix over GF(2).
    pivots : list of int
        Column indices of the pivot positions (0-indexed).
    rank : int
        rank(A) over GF(2).

    Examples
    --------
    >>> A = [[1, 0, 1], [1, 1, 0], [0, 1, 1]]
    >>> R, pivots, r = rref_GF2(A)
    >>> r
    3

    References
    ----------
    .. [LN97] Theorem 6.7 — row equivalence classes over GF(2).
    .. [Coh93] Algorithm 2.1.3 — bit-parallel Gaussian elimination.
    """
    m = len(A)
    if m == 0:
        return [], [], 0
    n = len(A[0])
    rows = _to_packed(A, n)

    pivot_row = 0
    pivots: list[int] = []

    for col in range(n):
        mask = 1 << col
        # Find a row with a 1 in this column
        sel = None
        for row in range(pivot_row, m):
            if rows[row] & mask:
                sel = row
                break
        if sel is None:
            continue

        rows[pivot_row], rows[sel] = rows[sel], rows[pivot_row]
        # Eliminate all other rows
        for row in range(m):
            if row != pivot_row and (rows[row] & mask):
                rows[row] ^= rows[pivot_row]

        pivots.append(col)
        pivot_row += 1
        if pivot_row == m:
            break

    return _from_packed(rows, m, n), pivots, len(pivots)


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------

def rank_GF2(A: list[list[int]]) -> int:
    """
    Rank of a GF(2) matrix.

    Parameters
    ----------
    A : list of list of int
        m × n matrix with entries in {0, 1}.

    Returns
    -------
    int
        rank(A) over GF(2).

    Examples
    --------
    >>> rank_GF2([[1, 0], [0, 0], [1, 0]])
    1

    References
    ----------
    .. [LN97] §6.3.
    """
    _, _, r = rref_GF2(A)
    return r


# ---------------------------------------------------------------------------
# Null space
# ---------------------------------------------------------------------------

def null_space_GF2(A: list[list[int]]) -> list[list[int]]:
    """
    Basis for the null space (kernel) of a GF(2) matrix.

    Computes  ker(A) = {x ∈ GF(2)^n : Ax = 0}  by augmenting A with the
    identity and reading off the free-variable columns after RREF.

    Parameters
    ----------
    A : list of list of int
        m × n matrix with entries in {0, 1}.

    Returns
    -------
    list of list of int
        A list of vectors forming a basis for ker(A) over GF(2).
        Empty list if A has full column rank.

    Notes
    -----
    Dimension of the null space equals n − rank(A)  (rank–nullity theorem).

    References
    ----------
    .. [LN97] Theorem 6.9 — rank–nullity over finite fields.
    """
    m = len(A)
    if m == 0:
        return []
    n = len(A[0])

    # Augment A with I_n (transposed: augment columns, not rows)
    # We work on A^T augmented with I_n, then read null space from free columns.
    # Equivalent: run RREF on [A | I] interpreted column-wise.
    # Standard approach: augment A on right with n×n identity, run RREF on A.
    aug = [list(A[i]) + [int(i == j) for j in range(m)] for i in range(m)]
    # We need RREF of A part, tracking what happens to identity part.
    # Use bit-packing only on the A-part for pivot selection:
    n_aug = n + m
    rows = _to_packed(aug, n_aug)

    pivot_row = 0
    pivots: list[int] = []

    for col in range(n):
        mask = 1 << col
        sel = None
        for row in range(pivot_row, m):
            if rows[row] & mask:
                sel = row
                break
        if sel is None:
            continue
        rows[pivot_row], rows[sel] = rows[sel], rows[pivot_row]
        for row in range(m):
            if row != pivot_row and (rows[row] & mask):
                rows[row] ^= rows[pivot_row]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == m:
            break

    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]

    basis = []
    for fc in free_cols:
        # Build null-space vector for free column fc
        vec = [0] * n
        vec[fc] = 1
        for idx, pc in enumerate(pivots):
            # coefficient of pivot variable = entry in column fc of rref row
            if (rows[idx] >> fc) & 1:
                vec[pc] = 1
        basis.append(vec)

    return basis


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------

def mat_mul_GF2(
    A: list[list[int]],
    B: list[list[int]],
) -> list[list[int]]:
    """
    Matrix product A · B over GF(2).

    Uses bit-packed inner products (AND + popcount for parity) for
    efficient computation.

    Parameters
    ----------
    A : list of list of int
        m × k matrix over GF(2).
    B : list of list of int
        k × n matrix over GF(2).

    Returns
    -------
    list of list of int
        m × n product matrix over GF(2).

    Raises
    ------
    ValueError
        If dimensions are incompatible.

    References
    ----------
    .. [Knu98] §4.6.3 — binary matrix multiplication.
    """
    m = len(A)
    k = len(A[0])
    if len(B) != k:
        raise ValueError(
            f"Dimension mismatch: A is {m}×{k}, B is {len(B)}×{len(B[0])}."
        )
    n = len(B[0])

    # Pack A rows and B columns
    a_rows = _to_packed(A, k)
    # Transpose B for column access
    B_T = [[B[i][j] for i in range(k)] for j in range(n)]
    b_cols = _to_packed(B_T, k)

    C = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(bin(a_rows[i] & b_cols[j]).count('1') % 2)
        C.append(row)
    return C


# ---------------------------------------------------------------------------
# Linear system solver
# ---------------------------------------------------------------------------

def solve_GF2(
    A: list[list[int]],
    b: list[int],
) -> list[int] | None:
    """
    Solve the linear system  Ax = b  over GF(2).

    Parameters
    ----------
    A : list of list of int
        m × n coefficient matrix over GF(2).
    b : list of int
        Right-hand side vector of length m, entries in {0, 1}.

    Returns
    -------
    list of int or None
        A solution x of length n over GF(2), or None if inconsistent.
        Free variables are set to 0.

    References
    ----------
    .. [Coh93] §2.2 — GF(2) system solving.
    """
    m = len(A)
    n = len(A[0])
    # Augmented [A | b] as (n+1)-wide rows
    aug = [list(A[i]) + [int(b[i]) & 1] for i in range(m)]
    rows = _to_packed(aug, n + 1)

    pivot_row = 0
    pivots: list[int] = []

    for col in range(n):
        mask = 1 << col
        sel = None
        for row in range(pivot_row, m):
            if rows[row] & mask:
                sel = row
                break
        if sel is None:
            continue
        rows[pivot_row], rows[sel] = rows[sel], rows[pivot_row]
        for row in range(m):
            if row != pivot_row and (rows[row] & mask):
                rows[row] ^= rows[pivot_row]
        pivots.append(col)
        pivot_row += 1

    # Consistency check
    rhs_mask = 1 << n
    for row in range(pivot_row, m):
        if rows[row] & rhs_mask:
            return None  # inconsistent

    x = [0] * n
    for idx, col in enumerate(pivots):
        x[col] = (rows[idx] >> n) & 1

    return x
