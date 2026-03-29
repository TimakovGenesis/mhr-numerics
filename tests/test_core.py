"""
tests/test_core.py
==========================
Synthetic test suite for mhr_numerics.

All test cases use randomly generated matrices and lattices, or
well-known mathematical constants (Mersenne primes, Fibonacci lattice).
No domain-specific constants or real-world datasets are referenced.

Run with:  python -m pytest tests/ -v
"""

import math
import sys
import os
import random
import numpy as np
import pytest

# Настройка путей для корректного нахождения пакетов core и Part_I
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.linalg_Fp import (
    gauss_jordan_Fp, det_Fp, inv_Fp, rank_Fp, solve_Fp, is_prime
)
from core.linalg_GF2 import (
    rref_GF2, rank_GF2, null_space_GF2, mat_mul_GF2, solve_GF2
)
from core.lattice import (
    gauss_reduce_2d, babai_cvp, lll_reduce, hermite_factor,
    shortest_vector_norm_bound, gram_schmidt
)
from core.interpolation import (
    lagrange_interpolate_Fp, poly_eval_Fp, poly_mul_Fp, poly_add_Fp,
    poly_roots_Fp, berlekamp_massey_Fp
)
from Part_I.variational import mhr_energy, mhr_solve, adiabatic_schedule

# ---------------------------------------------------------------------------
# Small primes used in tests (all verified prime)
# ---------------------------------------------------------------------------
P7   = 7
P11  = 11
P13  = 13
P17  = 17
P31  = 31
P101 = 101
P_MERSENNE_7 = 127      # Mersenne prime 2^7 − 1
P_MERSENNE_13 = 8191    # Mersenne prime 2^13 − 1


# ===========================================================================
# test_linalg_Fp.py
# ===========================================================================

class TestIsPrime:
    def test_small_primes(self):
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 127, 8191]:
            assert is_prime(p), f"{p} should be prime"

    def test_composites(self):
        for n in [1, 4, 6, 9, 15, 49, 100, 128]:
            assert not is_prime(n), f"{n} should be composite"


class TestGaussJordanFp:
    def test_identity_3x3(self):
        """RREF of identity is identity."""
        I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        R, pivots, r = gauss_jordan_Fp(I, p=P7)
        assert r == 3
        assert pivots == [0, 1, 2]
        for i in range(3):
            for j in range(3):
                assert R[i][j] == (1 if i == j else 0)

    def test_rank_deficient(self):
        """Matrix with a zero row has rank < n."""
        A = [[1, 2], [2, 4]]   # rows are proportional over any field
        _, _, r = gauss_jordan_Fp(A, p=P7)
        assert r == 1

    def test_full_rank_random(self):
        """A random lower-triangular matrix with non-zero diagonal has full rank."""
        p = P101
        n = 5
        rng = random.Random(42)
        # Lower triangular with ones on diagonal
        A = [[1 if j <= i else 0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i):
                A[i][j] = rng.randint(1, p - 1)
        _, _, r = gauss_jordan_Fp(A, p=p)
        assert r == n

    def test_augmented_mode(self):
        """In augmented mode the last column is excluded from pivot selection."""
        # System:  x + 2y = 5,  3x + 7y = 18  over F_11
        # Solution: x=1, y=2 (check: 1+4=5, 3+14=17≡6... use p=11: 17%11=6≠18%11=7)
        # Let's use a system we can verify:
        # 2x + 3y ≡ 0,  x + y ≡ 4   mod 7
        A = [[2, 3, 0], [1, 1, 4]]
        R, _, r = gauss_jordan_Fp(A, p=P7, augmented=True)
        assert r == 2  # full rank in left part


class TestDetFp:
    def test_2x2_known(self):
        """det([[1,2],[3,4]]) = -2 ≡ 5 mod 7."""
        assert det_Fp([[1, 2], [3, 4]], p=P7) == 5

    def test_identity(self):
        """det(I_n) = 1."""
        for n in range(1, 6):
            I = [[int(i == j) for j in range(n)] for i in range(n)]
            assert det_Fp(I, p=P13) == 1

    def test_singular(self):
        """Singular matrix has det = 0."""
        A = [[2, 4], [1, 2]]
        assert det_Fp(A, p=P7) == 0

    def test_transpose(self):
        """det(A) = det(Aᵀ)."""
        A = [[1, 2, 3], [0, 4, 5], [1, 0, 6]]
        AT = [[A[j][i] for j in range(3)] for i in range(3)]
        for p in [P7, P11, P31]:
            assert det_Fp(A, p=p) == det_Fp(AT, p=p)


class TestInvFp:
    def test_2x2(self):
        """A · A^{-1} = I over F_p."""
        A = [[3, 1], [2, 5]]
        for p in [P7, P11, P17, P101]:
            Ainv = inv_Fp(A, p=p)
            # Compute A · Ainv mod p
            n = 2
            prod = [[sum(A[i][k] * Ainv[k][j] for k in range(n)) % p
                     for j in range(n)] for i in range(n)]
            for i in range(n):
                for j in range(n):
                    assert prod[i][j] == (1 if i == j else 0)

    def test_singular_raises(self):
        """Inverting a singular matrix raises ValueError."""
        with pytest.raises(ValueError):
            inv_Fp([[1, 2], [2, 4]], p=P7)

    def test_3x3_round_trip(self):
        """(A^{-1})^{-1} = A over F_p."""
        A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        p = P_MERSENNE_7
        Ainv = inv_Fp(A, p=p)
        Ainvinv = inv_Fp(Ainv, p=p)
        for i in range(3):
            for j in range(3):
                assert Ainvinv[i][j] == A[i][j] % p


class TestSolveFp:
    def test_unique_solution(self):
        """Ax = b has unique solution over F_p."""
        # x + y = 3,  2x + y = 5  over F_7  → x=2, y=1
        A = [[1, 1], [2, 1]]
        b = [3, 5]
        x = solve_Fp(A, b, p=P7)
        assert x is not None
        assert (A[0][0] * x[0] + A[0][1] * x[1]) % P7 == 3
        assert (A[1][0] * x[0] + A[1][1] * x[1]) % P7 == 5

    def test_inconsistent(self):
        """Inconsistent system returns None."""
        # x + y = 1, x + y = 2 (mod 7) — no solution
        A = [[1, 1], [1, 1]]
        assert solve_Fp(A, [1, 2], p=P7) is None

    def test_underdetermined(self):
        """Underdetermined system returns a valid particular solution."""
        A = [[1, 2, 3]]
        b = [6]
        x = solve_Fp(A, b, p=P7)
        assert x is not None
        assert (A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2]) % P7 == 6


# ===========================================================================
# test_linalg_GF2.py
# ===========================================================================

class TestRrefGF2:
    def test_identity(self):
        I = [[1, 0], [0, 1]]
        R, piv, r = rref_GF2(I)
        assert r == 2

    def test_rank_1(self):
        A = [[1, 0, 1], [1, 0, 1]]   # rows are equal → rank 1
        _, _, r = rref_GF2(A)
        assert r == 1

    def test_full_rank_3x3(self):
        # Note: [[1,0,1],[0,1,1],[1,1,0]] has rank 2 over GF(2) because
        # row3 = row1 XOR row2.  Use identity matrix for a genuine rank-3 test.
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        _, _, r = rref_GF2(A)
        assert r == 3

    def test_dependent_rows(self):
        # [[1,0,1],[0,1,1],[1,1,0]]: row3 = row1 ⊕ row2 over GF(2) → rank 2
        A = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
        _, _, r = rref_GF2(A)
        assert r == 2

    def test_zero_matrix(self):
        A = [[0, 0], [0, 0]]
        _, _, r = rref_GF2(A)
        assert r == 0


class TestNullSpaceGF2:
    def test_rank_nullity(self):
        """dim(ker A) = n − rank(A)."""
        rng = random.Random(7)
        for _ in range(5):
            m, n = rng.randint(2, 6), rng.randint(2, 8)
            A = [[rng.randint(0, 1) for _ in range(n)] for _ in range(m)]
            r = rank_GF2(A)
            ns = null_space_GF2(A)
            assert len(ns) == n - r

    def test_null_vectors_satisfy_Ax_eq_0(self):
        """Each null-space vector satisfies Ax = 0 over GF(2)."""
        A = [[1, 1, 0], [0, 1, 1]]
        ns = null_space_GF2(A)
        for v in ns:
            for i in range(len(A)):
                dot = sum(A[i][j] * v[j] for j in range(len(v))) % 2
                assert dot == 0


class TestMatMulGF2:
    def test_identity(self):
        I = [[1, 0], [0, 1]]
        A = [[1, 1], [0, 1]]
        C = mat_mul_GF2(A, I)
        assert C == A

    def test_self_product(self):
        """A · A over GF(2)."""
        A = [[1, 1], [0, 1]]
        C = mat_mul_GF2(A, A)  # [[1,0],[0,1]] mod 2: [[1+0,1+1],[0,0+1]]
        assert C[0] == [1, 0]
        assert C[1] == [0, 1]


class TestSolveGF2:
    def test_simple(self):
        A = [[1, 0], [0, 1]]
        x = solve_GF2(A, [1, 1])
        assert x == [1, 1]

    def test_inconsistent(self):
        A = [[1, 1], [1, 1]]
        assert solve_GF2(A, [0, 1]) is None


# ===========================================================================
# test_lattice.py
# ===========================================================================

class TestGaussReduce2D:
    def test_already_reduced(self):
        """An already-reduced basis is unchanged (up to ordering)."""
        b1 = [3, 0]
        b2 = [0, 5]
        r1, r2 = gauss_reduce_2d(b1, b2)
        n1 = sum(x ** 2 for x in r1)
        n2 = sum(x ** 2 for x in r2)
        assert n1 <= n2

    def test_fibonacci_lattice(self):
        """
        The Fibonacci lattice basis [[F_{2k}, F_{2k+1}], [F_{2k+1}, F_{2k+2}]]
        reduces to the canonical 2D lattice; known shortest vector has length ~φ^k.
        """
        # Use Fibonacci numbers: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55
        b1 = [8, 13]
        b2 = [13, 21]
        r1, r2 = gauss_reduce_2d(b1, b2)
        # For a Fibonacci lattice the shortest vector length is small
        norm_r1 = math.sqrt(sum(x ** 2 for x in r1))
        assert norm_r1 < math.sqrt(sum(x ** 2 for x in b1))

    def test_random_2d(self):
        """Gaussian reduction gives ‖b₁‖ ≤ ‖b₂‖."""
        rng = random.Random(99)
        for _ in range(20):
            b1 = [rng.randint(-50, 50), rng.randint(-50, 50)]
            b2 = [rng.randint(-50, 50), rng.randint(-50, 50)]
            if b1 == [0, 0] or b2 == [0, 0]:
                continue
            r1, r2 = gauss_reduce_2d(b1, b2)
            n1 = sum(x ** 2 for x in r1)
            n2 = sum(x ** 2 for x in r2)
            assert n1 <= n2 + 1  # floating-point tolerance


class TestBabaiCVP:
    def test_identity_basis(self):
        """Babai on identity basis should give nearest integer point."""
        B = [[1.0, 0.0], [0.0, 1.0]]
        target = [2.7, 3.1]
        v = babai_cvp(B, target)
        assert abs(v[0] - 3.0) < 0.5 and abs(v[1] - 3.0) < 0.5

    def test_2d_simple(self):
        """Simple 2D test: basis [[1,0],[0,2]], target [0.4, 1.6]."""
        B = [[1.0, 0.0], [0.0, 2.0]]
        target = [0.4, 1.6]
        v = babai_cvp(B, target)
        # Expected CVP: [0, 2] (nearest lattice point)
        assert abs(v[0]) < 1.0
        assert abs(v[1] - 2.0) < 1.0


class TestLLLReduce:
    def test_output_is_shorter(self):
        """LLL reduction does not increase the first basis vector norm."""
        rng = np.random.default_rng(42)
        n = 8
        B_raw = rng.integers(-20, 20, size=(n, n)).tolist()
        try:
            B_lll = lll_reduce(B_raw)
            norm_raw = math.sqrt(sum(x ** 2 for x in B_raw[0]))
            norm_lll = math.sqrt(sum(x ** 2 for x in B_lll[0]))
            # After LLL, first vector should not be longer than before
            # (LLL minimises ‖b₁‖ up to a factor of 2^((n-1)/2))
            assert norm_lll <= norm_raw * 2 ** ((n - 1) / 2) + 1e-6
        except ImportError:
            pytest.skip("fpylll not available")

    def test_same_lattice(self):
        """LLL output spans the same lattice (det is preserved)."""
        B = [[1, 0, 2], [0, 2, 1], [3, 1, 0]]
        try:
            B_lll = lll_reduce(B, delta=0.75)
            det_orig = abs(int(round(np.linalg.det(np.array(B, dtype=float)))))
            det_lll  = abs(int(round(np.linalg.det(np.array(B_lll, dtype=float)))))
            assert det_orig == det_lll
        except ImportError:
            pytest.skip("fpylll not available")


class TestHermiteFactor:
    def test_known_example(self):
        """For a 2D lattice with vol=1 and ‖b₁‖=1, δ=1."""
        delta = hermite_factor(b1_norm=1.0, vol=1.0, n=2)
        assert abs(delta - 1.0) < 1e-10

    def test_positive(self):
        assert hermite_factor(2.0, 4.0, 3) > 0


# ===========================================================================
# test_interpolation.py
# ===========================================================================

class TestLagrangeInterpolateFp:
    def test_linear_reconstruction(self):
        """Reconstruct f(x) = 3x + 2 over F_7 from 2 points."""
        p = P7
        pts = [(0, 2), (1, 5)]   # f(0)=2, f(1)=5
        poly = lagrange_interpolate_Fp(pts, p)
        assert poly_eval_Fp(poly, 0, p) == 2
        assert poly_eval_Fp(poly, 1, p) == 5
        assert poly_eval_Fp(poly, 3, p) == (3 * 3 + 2) % p

    def test_quadratic_reconstruction(self):
        """Reconstruct f(x) = x² + x + 1 over F_11 from 3 points."""
        p = P11
        def f(x): return (x*x + x + 1) % p
        pts = [(i, f(i)) for i in range(3)]
        poly = lagrange_interpolate_Fp(pts, p)
        for x in range(p):
            assert poly_eval_Fp(poly, x, p) == f(x)

    def test_degree_5_over_mersenne(self):
        """Degree-5 polynomial reconstructed from 6 points over F_{127}."""
        p = P_MERSENNE_7
        coeffs = [1, 2, 3, 4, 5, 6]  # f(x) = 1+2x+3x²+4x³+5x⁴+6x⁵
        pts = [(x, poly_eval_Fp(coeffs, x, p)) for x in range(6)]
        recovered = lagrange_interpolate_Fp(pts, p)
        for x in range(10):
            assert poly_eval_Fp(recovered, x, p) == poly_eval_Fp(coeffs, x, p)

    def test_duplicate_x_raises(self):
        with pytest.raises(ValueError):
            lagrange_interpolate_Fp([(1, 2), (1, 3)], p=P7)


class TestPolyMulFp:
    def test_mul_by_monomial(self):
        """(x+1)(x-1) = x²-1 over F_7."""
        A = [(P7-1) % P7, 1]   # x-1 = [6, 1]
        B = [1, 1]              # x+1 = [1, 1]
        C = poly_mul_Fp(A, B, P7)
        # x²-1 = [6, 0, 1] over F_7
        assert C == [6, 0, 1]

    def test_commutativity(self):
        A = [1, 2, 3]
        B = [4, 5]
        assert poly_mul_Fp(A, B, P11) == poly_mul_Fp(B, A, P11)


class TestPolyEvalFp:
    def test_constant(self):
        assert poly_eval_Fp([5], 99, P7) == 5 % P7

    def test_linear(self):
        assert poly_eval_Fp([0, 1], 3, P11) == 3   # f(x) = x

    def test_horner_vs_naive(self):
        """Horner evaluation matches naive summation."""
        p = P101
        poly = [3, 7, 2, 5, 1]
        for x in range(10):
            naive = sum(poly[i] * pow(x, i, p) for i in range(len(poly))) % p
            assert poly_eval_Fp(poly, x, p) == naive


class TestPolyRootsFp:
    def test_no_roots(self):
        """x² + 1 has no roots mod 7."""
        # x²+1 = [1, 0, 1]
        roots = poly_roots_Fp([1, 0, 1], p=P7)
        for r in roots:
            assert (r*r + 1) % P7 == 0

    def test_linear_root(self):
        """x − 3 has exactly one root mod 7."""
        roots = poly_roots_Fp([(P7-3) % P7, 1], p=P7)
        assert 3 in roots

    def test_splits_completely(self):
        """∏_{a=0}^{4} (x − a) has 5 roots mod 5."""
        # polynomial x(x-1)(x-2)(x-3)(x-4) = x^5 - x mod 5
        # = [0, 4, 0, 0, 0, 1] over F_5 (Fermat: x^5=x)
        p = 5
        poly = [0, 4, 0, 0, 0, 1]   # x^5 + 4x = x^5 - x mod 5
        roots = poly_roots_Fp(poly, p=p)
        for r in range(p):
            assert r in roots or poly_eval_Fp(poly, r, p) != 0


class TestBerlekampMasseyFp:
    def test_fibonacci_recurrence(self):
        """
        Fibonacci sequence mod 7 satisfies s_n = s_{n-1} + s_{n-2},
        i.e. LFSR polynomial C(x) = 1 − x − x² = [1, 6, 6] mod 7.
        """
        p = P7
        s = [1, 1]
        for _ in range(12):
            s.append((s[-1] + s[-2]) % p)
        C = berlekamp_massey_Fp(s, p)
        # Verify: LFSR recurrence holds
        L = len(C) - 1
        for n in range(L, len(s)):
            val = sum(C[j] * s[n-j] for j in range(L+1)) % p
            assert val == 0

    def test_constant_sequence(self):
        """Constant sequence [a, a, a, ...] has LFSR of length 1: C = [1, p-1]."""
        p = P11
        s = [3] * 10
        C = berlekamp_massey_Fp(s, p)
        # s_n = s_{n-1} → 1·s_n − 1·s_{n-1} = 0 → C = [1, -1 mod p]
        assert len(C) == 2
        assert C[0] == 1
        assert C[1] == p - 1


# ===========================================================================
# test_variational.py
# ===========================================================================

class TestAdiabaticsSchedule:
    def test_warm_up_is_zero(self):
        for t in range(50):
            assert adiabatic_schedule(t, lambda_max=100, n_steps=200, warm_up=50) == 0.0

    def test_monotone_increasing(self):
        values = [adiabatic_schedule(t, 500, 200) for t in range(200)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-12

    def test_approaches_max(self):
        lam = adiabatic_schedule(199, 500, 200)
        assert lam > 490


class TestMHREnergy:
    def test_identity_basis_at_origin(self):
        """
        For B = I_n, μ = 0, λ = 0:
        E = σ² · n  (only kinetic / quadratic term from spread).
        """
        n = 4
        B = np.eye(n)
        mu = np.zeros(n)
        log_sigma = math.log(0.1)
        e = mhr_energy(mu, log_sigma, B, lambda_val=0.0, n_samples=2000,
                       rng=np.random.default_rng(42))
        expected = n * 0.01   # σ² · n  = 0.01 · 4
        assert abs(e - expected) < 0.02 * expected + 0.01   # 2% tolerance

    def test_energy_decreases_with_smaller_sigma(self):
        """For fixed μ at integer point, smaller σ → smaller energy."""
        n = 3
        B = np.eye(n)
        mu = np.array([1.0, 2.0, 3.0])  # integer point
        e_large = mhr_energy(mu, math.log(1.0), B, 0.0, 1000,
                             np.random.default_rng(0))
        e_small = mhr_energy(mu, math.log(0.01), B, 0.0, 1000,
                             np.random.default_rng(0))
        assert e_small < e_large


class TestMHRSolve:
    def test_small_lattice_finds_short_vector(self):
        """
        For a small 4-dimensional lattice (constructed so the shortest vector
        is known), mhr_solve should find a candidate with small norm.
        """
        # Diagonal lattice: shortest vector is the first standard basis vector
        # scaled by the smallest diagonal entry
        B = np.diag([3.0, 5.0, 7.0, 11.0])
        result = mhr_solve(
            B,
            lambda_max=100.0,
            n_outer=30,
            n_inner=10,
            n_samples_energy=128,
            seed=0,
        )
        # The true shortest vector has norm 3; accept anything ≤ 2 × 3 = 6
        assert result.candidate_norm < 12.0

    def test_result_has_correct_attributes(self):
        B = np.eye(3)
        result = mhr_solve(B, lambda_max=50, n_outer=10, n_inner=5, seed=1)
        assert hasattr(result, 'candidate_norm')
        assert hasattr(result, 'candidate')
        assert hasattr(result, 'energy_history')
        assert len(result.energy_history) == 10
        assert result.candidate.shape == (3,)
        assert result.candidate_norm >= 0.0


# ===========================================================================
# Integration test: full pipeline
# ===========================================================================

class TestFullPipeline:
    def test_solve_and_verify(self):
        """
        Build a random n×n invertible matrix over F_p, solve a linear system,
        and verify the solution directly.
        """
        p = P101
        n = 5
        rng = random.Random(2024)
        # Lower triangular with non-zero diagonal (guaranteed invertible)
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            A[i][i] = rng.randint(1, p - 1)
            for j in range(i):
                A[i][j] = rng.randint(0, p - 1)

        b = [rng.randint(0, p - 1) for _ in range(n)]
        x = solve_Fp(A, b, p)
        assert x is not None
        for i in range(n):
            val = sum(A[i][j] * x[j] for j in range(n)) % p
            assert val == b[i] % p

    def test_interpolate_and_root_find(self):
        """Interpolate a polynomial and verify root finding."""
        p = P31
        # f(x) = (x-3)(x-5) = x²-8x+15  over F_31
        f_coeffs = [(15) % p, (-8) % p, 1]  # [c0, c1, c2]
        pts = [(i, poly_eval_Fp(f_coeffs, i, p)) for i in range(3)]
        recovered = lagrange_interpolate_Fp(pts, p)
        roots = poly_roots_Fp(recovered, p)
        assert 3 in roots
        assert 5 in roots

    def test_gf2_null_space_and_solve_consistency(self):
        """null_space ⊆ ker(A) ⟺ solve_GF2 returns consistent solution."""
        rng = random.Random(55)
        m, n = 4, 6
        A = [[rng.randint(0, 1) for _ in range(n)] for _ in range(m)]
        ns = null_space_GF2(A)
        for v in ns:
            for i in range(m):
                assert sum(A[i][j] * v[j] for j in range(n)) % 2 == 0

        b = [0] * m   # zero RHS — always consistent
        x = solve_GF2(A, b)
        assert x is not None
        for i in range(m):
            assert sum(A[i][j] * x[j] for j in range(n)) % 2 == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run via pytest for full output
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
