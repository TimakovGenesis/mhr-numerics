# Part of mhr-numerics library
# Companion to: "Spectral gap and modular degree of elliptic curves, Part II:
#                The Spectral Rank Criterion and Selmer Theory"
# DOI: 10.5281/zenodo.19165246
# https://github.com/TimakovGenesis/mhr-numerics
"""
mhr_real_solver.py
Real MHR variational solver для deformation.py
Интерфейс совместим с DeformationEngine(solver=mhr_real_solver)

Theory (Part I, §4):
    H_λ = -½ d²/dx² + D·x² + λ·sin²(πx)
    Гауссовский анзатц:  ψ(μ, σ; x) = (2πσ²)^{-1/4} · exp(-(x-μ)²/(4σ²))
    Аналитический функционал (eq. 4.2):
        E = 1/(4σ²) + D(μ² + σ²) + λ[sin²(πμ)·exp(-2π²σ²) + ½(1 - exp(-2π²σ²))]

Протокол раздельной оптимизации:
    Вакуумный солвер  (μ₀ ≈ 0): E_vac(λ)
    Топологический солвер (μ₀ ≈ 1): E_1(λ)
    ΔE(λ) = E_1(λ) - E_vac(λ)

Для rank≥1: Bloch band width оператора h_y = -½∂²_y + λsin²(πy) через
матрицу Фурье — proxy для спектрального коллапса.

[F: контрольные числа Part I]:
    37.a1 (D=2): λ=100 → ΔE≈1.9941, λ=1000 → ΔE≈1.9996, λ=5000 → ΔE≈1.9999
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Аналитический функционал
# ─────────────────────────────────────────────────────────────────────────────

def energy_functional(mu: float, sigma: float, D: float, lmbda: float) -> float:
    """
    Аналитический функционал <ψ|H_λ|ψ> для гауссовского анзатца.

    H_λ = -½ d²/dx² + D·x² + λ·sin²(πx)
    E(μ,σ) = T + V_metric + V_osc

    Parameters
    ----------
    mu    : float — центр гауссиана
    sigma : float — ширина (положительная)
    D     : float — deg φ_E (модулярный степень)
    lmbda : float — параметр связи λ

    Returns
    -------
    float — значение энергии
    """
    sigma = abs(sigma) + 1e-8  # гарантия положительности

    # Кинетическая энергия: T = 1/(4σ²)
    T = 1.0 / (4.0 * sigma ** 2)

    # Метрический потенциал: V_met = D(μ² + σ²)
    V_met = D * (mu ** 2 + sigma ** 2)

    # Осцилляторный потенциал с guard при больших λ (exp может → 0)
    pi2s2 = 2.0 * np.pi ** 2 * sigma ** 2
    # При pi2s2 > 500 численно exp ≈ 0, обрабатываем явно
    if pi2s2 > 500.0:
        exp_term = 0.0
    else:
        exp_term = np.exp(-pi2s2)

    V_osc = lmbda * (
        np.sin(np.pi * mu) ** 2 * exp_term
        + 0.5 * (1.0 - exp_term)
    )

    return T + V_met + V_osc


def _grad_energy(mu: float, sigma: float, D: float, lmbda: float) -> Tuple[float, float]:
    """Градиент функционала (аналитически) для оценки residual."""
    sigma = abs(sigma) + 1e-8
    pi2s2 = 2.0 * np.pi ** 2 * sigma ** 2
    exp_term = 0.0 if pi2s2 > 500.0 else np.exp(-pi2s2)

    # dE/dμ
    dE_dmu = (2.0 * D * mu
              + lmbda * 2.0 * np.pi * np.sin(np.pi * mu) * np.cos(np.pi * mu) * exp_term)

    # dE/dσ (через σ = |σ_raw| + 1e-8, здесь σ уже положительный)
    dE_dsigma = (- 1.0 / (2.0 * sigma ** 3)
                 + 2.0 * D * sigma
                 + lmbda * (
                     np.sin(np.pi * mu) ** 2 * exp_term * (-2.0 * np.pi ** 2) * 2.0 * sigma
                     + 0.5 * exp_term * 2.0 * np.pi ** 2 * 2.0 * sigma
                 ))
    return dE_dmu, dE_dsigma


# ─────────────────────────────────────────────────────────────────────────────
# Вакуумный солвер (E_vac, μ₀ ≈ 0)
# ─────────────────────────────────────────────────────────────────────────────

def vacuum_solver(
    D: float,
    lmbda: float,
    n_restarts: int = 7,
    seed: int = 42,
) -> Tuple[float, bool, int, float]:
    """
    Найти E_vac(λ) — основное состояние у x=0.

    При малых λ (< ~200) осциллятор слаб, минимум у μ=0, σ большой.
    При больших λ минимум у μ=0 с малым σ.
    Перезапуски покрывают оба режима.

    Returns
    -------
    (E_vac, converged, total_iterations, residual)
    """
    rng = np.random.default_rng(seed)
    best_E = np.inf
    best_iters = 0
    best_res = np.inf
    any_converged = False

    # Фиксированные стартовые точки + случайные возмущения
    # μ₀ строго вблизи 0
    fixed_starts = [
        (0.0,    0.5),
        (0.0,    1.0 / (1.0 + lmbda ** 0.25)),  # адаптивный σ
        (0.0,    0.1),
        (0.01,   0.5),
        (-0.01,  0.5),
    ]
    random_starts = [(rng.normal(0.0, 0.02), 0.5) for _ in range(n_restarts - len(fixed_starts))]
    all_starts = fixed_starts + random_starts

    for mu0, sigma0 in all_starts:
        result = minimize(
            fun     = lambda p: energy_functional(p[0], p[1], D, lmbda),
            x0      = [mu0, sigma0],
            method  = 'L-BFGS-B',
            bounds  = [(-0.5, 0.5), (1e-6, 10.0)],   # μ строго < 0.5 — вблизи 0
            options = {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10},
        )

        if result.fun < best_E:
            best_E = float(result.fun)
            best_iters = int(result.nit)
            g = _grad_energy(result.x[0], result.x[1], D, lmbda)
            best_res = float(max(abs(g[0]), abs(g[1])))
            if result.success:
                any_converged = True

    return best_E, any_converged, best_iters, best_res


# ─────────────────────────────────────────────────────────────────────────────
# Топологический солвер (E_1, μ₀ ≈ 1)
# ─────────────────────────────────────────────────────────────────────────────

def topological_solver(
    D: float,
    lmbda: float,
    n_restarts: int = 7,
    seed: int = 137,
) -> Tuple[float, bool, int, float]:
    """
    Найти E_1(λ) — первое решёточное возбуждение у x=1.

    ВАЖНО: μ ограничено в (0.5, 1.5) — строго в лунке x=1.
    Это предотвращает соскальзывание в вакуумный минимум у x=0.

    Returns
    -------
    (E_1, converged, total_iterations, residual)
    """
    rng = np.random.default_rng(seed)
    best_E = np.inf
    best_iters = 0
    best_res = np.inf
    any_converged = False

    fixed_starts = [
        (1.0,    0.5),
        (1.0,    1.0 / (1.0 + lmbda ** 0.25)),
        (1.0,    0.1),
        (1.01,   0.5),
        (0.99,   0.5),
    ]
    random_starts = [(1.0 + rng.normal(0.0, 0.02), 0.5) for _ in range(n_restarts - len(fixed_starts))]
    all_starts = fixed_starts + random_starts

    for mu0, sigma0 in all_starts:
        result = minimize(
            fun     = lambda p: energy_functional(p[0], p[1], D, lmbda),
            x0      = [mu0, sigma0],
            method  = 'L-BFGS-B',
            bounds  = [(0.5, 1.5), (1e-6, 10.0)],   # μ строго в лунке x=1
            options = {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10},
        )

        if result.fun < best_E:
            best_E = float(result.fun)
            best_iters = int(result.nit)
            g = _grad_energy(result.x[0], result.x[1], D, lmbda)
            best_res = float(max(abs(g[0]), abs(g[1])))
            if result.success:
                any_converged = True

    return best_E, any_converged, best_iters, best_res


# ─────────────────────────────────────────────────────────────────────────────
# Bloch band width (proxy для rank≥1)
# ─────────────────────────────────────────────────────────────────────────────

def bloch_band_width(lmbda: float, n_fourier: int = 50) -> float:
    """
    Ширина нижней зоны Блоха оператора h = -½∂²_y + λsin²(πy).

    Метод: матрица Хилла для двух квазиимпульсов:
      q=0  → зонный минимум E_0(0)    (нижний край зоны)
      q=π  → зонный максимум E_0(π)   (верхний край зоны)
    Band width W = E_0(π) - E_0(0)

    Матрица в базисе e^{i(q+2πk)y}, k=-n..n:
      H_{k,k}   = ½(q + 2πk)²  +  λ/2
      H_{k,k±1} = -λ/4

    При λ→∞: W ~ exp(-c√λ) → 0  (спектральный коллапс)
    При λ=0:  W = 2π²  (свободный электрон)

    Parameters
    ----------
    lmbda    : float — параметр связи λ
    n_fourier: int   — порядок (обычно 50 достаточно)

    Returns
    -------
    float — ширина нижней зоны Блоха W = E_0(π) - E_0(0)
    """
    dim = 2 * n_fourier + 1
    ks  = np.arange(-n_fourier, n_fourier + 1, dtype=float)

    # Потенциальные матричные элементы (одинаковы для всех q)
    off_diag = -0.25 * lmbda

    def lowest_eigenvalue(q: float) -> float:
        """Наименьшее собственное значение H(q)."""
        # Кинетика: ½(q + 2πk)²
        kinetic = 0.5 * (q + 2.0 * np.pi * ks) ** 2
        # Полная диагональ: кинетика + λ/2 (от <sin²>)
        diag = kinetic + 0.5 * lmbda
        H = np.diag(diag)
        # Потенциальные связи k↔k±1
        for i in range(dim - 1):
            H[i, i + 1] = off_diag
            H[i + 1, i] = off_diag
        eigvals = np.linalg.eigvalsh(H)
        return float(eigvals[0])

    E_min = lowest_eigenvalue(q=0.0)          # нижний край зоны
    E_max = lowest_eigenvalue(q=np.pi)        # верхний край зоны (q = π/a, a=1)

    width = E_max - E_min
    # Ширина зоны всегда ≥ 0; при численных ошибках клэмпим
    return max(width, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция-интерфейс
# ─────────────────────────────────────────────────────────────────────────────

def mhr_real_solver(lmbda: float, params: Dict) -> Tuple[float, bool, int, float]:
    """
    Real MHR variational solver.

    Интерфейс совместим с DeformationEngine(solver=mhr_real_solver).

    Parameters
    ----------
    lmbda  : float — параметр связи λ
    params : dict  — curve_params с ключами:
        'deg_phi' : int   — модулярная степень D = deg φ_E
        'rank'    : int   — ранг кривой

    Returns
    -------
    (delta_E, converged, iterations, residual) : tuple
        delta_E   : float — спектральный зазор ΔE(λ) = E_1 - E_vac
        converged : bool  — оба решателя сошлись
        iterations: int   — суммарное число итераций
        residual  : float — max(|grad E_vac|, |grad E_1|)

    Notes
    -----
    rank=0 : двухточечная оптимизация (вакуум + топологический минимум)
    rank≥1 : Bloch band width как proxy для спектрального коллапса
    """
    D    = float(params.get('deg_phi', 2))
    rank = int(params.get('rank', 0))

    if rank >= 1:
        # ── Режим rank≥1: Bloch band width ──────────────────────────────────
        # При λ→∞ ΔE → 0 (спектральный коллапс)
        # Используем Bloch band width как физически корректный proxy
        delta_E = bloch_band_width(lmbda, n_fourier=60)
        # Сходимость: всегда (диагонализация точная)
        converged  = True
        iterations = 60 * 2 + 1  # размер матрицы
        residual   = 0.0
        return delta_E, converged, iterations, residual

    # ── Режим rank=0: двухточечная оптимизация ──────────────────────────────
    E_vac, ok_vac, iters_vac, res_vac = vacuum_solver(D, lmbda)
    E_1,   ok_1,   iters_1,   res_1   = topological_solver(D, lmbda)

    delta_E   = E_1 - E_vac
    converged = ok_vac and ok_1
    iterations = iters_vac + iters_1
    residual   = max(res_vac, res_1)

    return delta_E, converged, iterations, residual
