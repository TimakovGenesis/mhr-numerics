"""
deformation.py  —  Asymptotic Deformation Engine  v2.0
mhr-numerics  |  Part II: Rank Detection via Spectral Collapse

════════════════════════════════════════════════════════════════════════════════
Theory (Part II)
════════════════════════════════════════════════════════════════════════════════

Part I established [F: mhr-numerics Zenodo]:

    lim_{λ→∞}  ΔE(λ)  =  deg φ_E

where  ΔE(λ)  is the spectral gap of the MHR variational problem at coupling
strength  λ,  and  deg φ_E  is the modular degree of the elliptic curve  E.

Part II studies the *approach* to this limit.  We fit the asymptotic series

    ΔE(λ)  =  deg φ_E  +  a₁ λ⁻¹  +  a₂ λ⁻²  +  a₃ λ⁻³  +  …          (*)

The leading correction coefficient  a₁  encodes the Mordell–Weil rank:

    Spectral Collapse Theorem [H→F: target of Part II]:
        rank(E/ℚ) = 0  ⟺  a₁ → deg φ_E     (no collapse)
        rank(E/ℚ) ≥ 1  ⟺  a₁ → 0            (spectral collapse)

Numerical evidence threshold:
    Primary:   |a₁| / σ(a₁) < 2.0           (sigma-test from cov matrix)
    Fallback:  |a₁| / max(1, deg φ_E) < ε_rel = 1e-4  (relative threshold)

    [FIX v2.0] Fallback threshold нормирован на deg φ_E — устраняет
    ложные срабатывания на кривых с большим deg_phi (~40 для 389.a1).

Adiabatic condition [F: Part I, §4]:
    λ_min = 28  (below this, quantum delocalisation dominates → exclude)

════════════════════════════════════════════════════════════════════════════════
Новое в v2.0
════════════════════════════════════════════════════════════════════════════════

  [NEW] bootstrap_uncertainty()    — 95% CI на a₁ через параметрический bootstrap
  [NEW] cross_validate_n_terms()   — k-fold CV для выбора оптимального числа слагаемых
  [NEW] ascii_spectral_plot()      — ASCII-визуализация кривой ΔE(λ) прямо в терминале
  [NEW] strain_collapse_bridge()   — интеграция с crystal_input_v3.TcDeformationAdapter
  [FIX] FitResult.spectral_collapse: относительный порог ε_rel вместо абсолютного
  [FIX] DeformationEngine.analyse: возвращает bootstrap_ci в результирующем dict
  [FIX] Убрана зависимость scipy.stats из верхнего уровня (lazy import)

Зависимости: numpy, scipy.  Optional: crystal_input_v3 (для BRIDGE).
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_ADIABATIC_MIN: float = 28.0
EPSILON_COLLAPSE_REL: float = 1e-4   # |a₁| / max(1, deg_phi) < ε → collapse
SIGMA_COLLAPSE:       float = 2.0    # z-threshold для sigma-test
MAX_SERIES_TERMS:     int   = 5
DEFAULT_LAMBDA_RANGE: np.ndarray = np.logspace(2, 5, 30)


# ─────────────────────────────────────────────────────────────────────────────
# Структуры данных
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeformationResult:
    """Одно измерение (λ, ΔE) с метаданными сходимости."""
    lmbda:      float
    delta_E:    float
    converged:  bool  = True
    iterations: int   = 0
    residual:   float = 0.0
    note:       str   = ""

    def is_valid(self) -> bool:
        return self.converged and self.lmbda >= LAMBDA_ADIABATIC_MIN


@dataclass
class FitResult:
    """
    Результат подгонки асимптотического ряда.

    Поля
    ----
    deg_phi    : float  — извлечённый модулярный степень (ΔE при λ→∞)
    coeffs     : list   — [a₁, a₂, …] коэффициенты коррекций
    cov        : ndarray — ковариационная матрица (NaN при fallback)
    r_squared  : float  — коэффициент детерминации
    n_terms    : int    — число слагаемых ряда
    n_points   : int    — число точек данных
    collapse_signal : float — |a₁| / deg_phi (безразмерный)
    bootstrap_ci    : tuple или None — (a1_mean, a1_std, lower95, upper95)
    """
    deg_phi:         float
    coeffs:          List[float]
    cov:             np.ndarray
    r_squared:       float
    n_terms:         int
    n_points:        int
    collapse_signal: float
    bootstrap_ci:    Optional[Tuple[float, float, float, float]] = None

    @property
    def a1(self) -> float:
        return self.coeffs[0] if self.coeffs else math.nan

    @property
    def spectral_collapse(self) -> bool:
        """
        True если a₁ статистически совместим с нулём (rank ≥ 1).

        Приоритеты:
          1. Bootstrap CI: 0 ∈ [lower95, upper95]
          2. Sigma-test из cov: |a₁|/σ(a₁) < SIGMA_COLLAPSE
          3. Relative fallback: |a₁| / max(1, deg_phi) < EPSILON_COLLAPSE_REL

        [FIX v2.0] Fallback теперь нормирован на deg_phi.
        """
        a1 = self.a1
        if math.isnan(a1):
            return False

        # Bootstrap CI (наиболее надёжный)
        if self.bootstrap_ci is not None:
            _, _, lower, upper = self.bootstrap_ci
            return lower <= 0.0 <= upper

        # Sigma-test из ковариационной матрицы
        if (self.cov is not None
                and self.cov.shape[0] > 1
                and not math.isnan(float(self.cov[1, 1]))
                and float(self.cov[1, 1]) >= 0):
            sigma_a1 = math.sqrt(float(self.cov[1, 1]))
            if sigma_a1 > 0:
                return (abs(a1) / sigma_a1) < SIGMA_COLLAPSE

        # Relative fallback [FIX v2.0]
        return abs(a1) / max(1.0, abs(self.deg_phi)) < EPSILON_COLLAPSE_REL

    def summary(self) -> str:
        lines = [
            f"  deg φ_E  = {self.deg_phi:.6f}",
            f"  a₁       = {self.a1:.6e}  {'← COLLAPSE' if self.spectral_collapse else ''}",
        ]
        for i, c in enumerate(self.coeffs[1:], start=2):
            lines.append(f"  a{i}       = {c:.6e}")
        lines.append(f"  R²       = {self.r_squared:.6f}")
        if self.bootstrap_ci is not None:
            mean, std, lo, hi = self.bootstrap_ci
            lines.append(f"  a₁ bootstrap: {mean:.4e} ± {std:.2e}  [95% CI: {lo:.3e}, {hi:.3e}]")
        lines += [
            f"  n_points = {self.n_points}  n_terms = {self.n_terms}",
            f"  Spectral Collapse: {'YES  →  rank ≥ 1' if self.spectral_collapse else 'NO   →  rank = 0'}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции фитирования
# ─────────────────────────────────────────────────────────────────────────────

def _make_model(n_terms: int) -> Callable:
    """f(λ, c₀, a₁, …, aₙ) = c₀ + a₁/λ + a₂/λ² + …"""
    def model(lmbda, *params):
        lmbda = np.asarray(lmbda, dtype=float)
        val   = np.full_like(lmbda, params[0])
        for k, ak in enumerate(params[1:], start=1):
            val = val + ak * lmbda ** (-k)
        return val
    return model


def fit_asymptotic_series(
    lambdas:  np.ndarray,
    delta_Es: np.ndarray,
    n_terms:  int = 3,
    p0:       Optional[List[float]] = None,
) -> FitResult:
    """
    Подгонка  ΔE(λ) = c₀ + a₁/λ + … + aₙ/λⁿ  методом наименьших квадратов.

    При нестабильности curve_fit автоматически переключается на polyfit в 1/λ.

    Parameters
    ----------
    lambdas  : 1-D array, все λ > 0
    delta_Es : 1-D array, измерения ΔE(λ)
    n_terms  : число коррекционных слагаемых (≥ 1, ≤ MAX_SERIES_TERMS)
    p0       : начальное приближение [c₀, a₁, …].  Auto если None.

    Returns
    -------
    FitResult (без bootstrap_ci — заполнить через bootstrap_uncertainty)
    """
    n_terms = max(1, min(n_terms, MAX_SERIES_TERMS))
    if len(lambdas) < n_terms + 2:
        raise ValueError(
            f"Нужно ≥ {n_terms + 2} точек для {n_terms}-членного ряда, "
            f"получено {len(lambdas)}."
        )

    model    = _make_model(n_terms)
    n_params = n_terms + 1

    if p0 is None:
        c0_guess = float(np.mean(delta_Es[lambdas >= np.percentile(lambdas, 75)]))
        p0 = [c0_guess] + [0.0] * n_terms

    use_fallback = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(
                model, lambdas, delta_Es,
                p0=p0, maxfev=50_000, method="trf",
            )
    except (RuntimeError, OptimizeWarning, ValueError):
        logger.warning("curve_fit нестабилен — polyfit в 1/λ")
        use_fallback = True

    if use_fallback:
        u    = 1.0 / lambdas
        poly = np.polyfit(u, delta_Es, deg=n_terms)
        popt = poly[::-1]  # [c₀, a₁, …]
        pcov = np.full((n_params, n_params), np.nan)

    deg_phi = float(popt[0])
    coeffs  = list(popt[1:])

    residuals = delta_Es - model(lambdas, *popt)
    ss_res    = float(np.sum(residuals ** 2))
    ss_tot    = float(np.sum((delta_Es - np.mean(delta_Es)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    collapse_signal = abs(coeffs[0]) / max(1.0, abs(deg_phi))

    return FitResult(
        deg_phi         = deg_phi,
        coeffs          = coeffs,
        cov             = pcov,
        r_squared       = r_squared,
        n_terms         = n_terms,
        n_points        = len(lambdas),
        collapse_signal = collapse_signal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI  [NEW v2.0]
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_uncertainty(
    lambdas:     np.ndarray,
    delta_Es:    np.ndarray,
    n_terms:     int = 3,
    n_bootstrap: int = 400,
    ci_level:    float = 0.95,
    seed:        Optional[int] = 42,
) -> Tuple[float, float, float, float]:
    """
    Параметрический bootstrap для оценки неопределённости a₁.

    Перевыборка с возвратом по точкам (λ_i, ΔE_i), каждый раз подгоняет ряд.
    Возвращает (mean, std, lower_ci, upper_ci) для a₁.

    Parameters
    ----------
    lambdas, delta_Es : исходные данные
    n_terms           : число слагаемых ряда
    n_bootstrap       : число реплик (default 400)
    ci_level          : уровень доверия (default 0.95 → 95% CI)
    seed              : random seed для воспроизводимости

    Returns
    -------
    (a1_mean, a1_std, lower_ci, upper_ci)

    Notes
    -----
    При нестабильных репликах они отбрасываются (try/except).
    Если < 20 реплик успешны — предупреждение.
    """
    rng = np.random.default_rng(seed)
    n   = len(lambdas)
    a1_samples: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        lam_b = lambdas[idx]
        dE_b  = delta_Es[idx]
        try:
            fit_b = fit_asymptotic_series(lam_b, dE_b, n_terms=n_terms)
            if not math.isnan(fit_b.a1):
                a1_samples.append(fit_b.a1)
        except Exception:
            pass

    if len(a1_samples) < 20:
        logger.warning(
            "Bootstrap: только %d/%d реплик успешны. CI ненадёжен.",
            len(a1_samples), n_bootstrap,
        )

    a1_arr = np.array(a1_samples)
    alpha  = (1.0 - ci_level) / 2.0
    lower  = float(np.percentile(a1_arr, 100 * alpha))
    upper  = float(np.percentile(a1_arr, 100 * (1 - alpha)))

    return float(np.mean(a1_arr)), float(np.std(a1_arr)), lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# Выбор n_terms через k-fold CV  [NEW v2.0]
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_n_terms(
    lambdas:  np.ndarray,
    delta_Es: np.ndarray,
    max_terms: int = 4,
    k_folds:   int = 5,
) -> int:
    """
    k-fold кросс-валидация для выбора оптимального числа слагаемых ряда.

    Минимизирует средний MSE на тестовых фолдах.

    Parameters
    ----------
    lambdas, delta_Es : данные
    max_terms : максимально рассматриваемое число членов (1 … max_terms)
    k_folds   : число фолдов

    Returns
    -------
    int — оптимальное n_terms по CV

    Notes
    -----
    При недостаточном числе точек возвращает 2 как консервативный default.
    """
    n = len(lambdas)
    if n < k_folds + max_terms + 2:
        logger.warning("Слишком мало точек для CV, возвращаю n_terms=2.")
        return 2

    idx   = np.arange(n)
    folds = np.array_split(idx, k_folds)
    best_terms, best_mse = 2, math.inf

    for n_t in range(1, max_terms + 1):
        mse_list = []
        for fold in folds:
            train = np.setdiff1d(idx, fold)
            if len(train) < n_t + 2:
                continue
            try:
                fit = fit_asymptotic_series(
                    lambdas[train], delta_Es[train], n_terms=n_t,
                )
                model = _make_model(n_t)
                pred  = model(lambdas[fold],
                              fit.deg_phi, *fit.coeffs)
                mse_list.append(float(np.mean((delta_Es[fold] - pred) ** 2)))
            except Exception:
                pass
        if mse_list:
            mean_mse = float(np.mean(mse_list))
            logger.debug("n_terms=%d  CV-MSE=%.4e", n_t, mean_mse)
            if mean_mse < best_mse:
                best_mse, best_terms = mean_mse, n_t

    logger.info("CV выбрал n_terms=%d  (MSE=%.4e)", best_terms, best_mse)
    return best_terms


# ─────────────────────────────────────────────────────────────────────────────
# ASCII визуализация  [NEW v2.0]
# ─────────────────────────────────────────────────────────────────────────────

def ascii_spectral_plot(
    lambdas:  np.ndarray,
    delta_Es: np.ndarray,
    fit:      Optional[FitResult] = None,
    width:    int = 60,
    height:   int = 14,
) -> str:
    """
    ASCII-арт кривой ΔE(λ) с опциональной подогнанной кривой.

    Parameters
    ----------
    lambdas, delta_Es : данные (измерения)
    fit               : FitResult — если задан, рисуется подогнанная кривая
    width, height     : размер символьного поля

    Returns
    -------
    str — многострочный ASCII-арт

    Example output::

        ΔE
      2.1 ┤ · ·
      2.0 ┤     · ·
      1.9 ┤         · ·──
              λ →
    """
    lam_log = np.log10(lambdas)
    dE_min, dE_max = delta_Es.min(), delta_Es.max()
    lam_min, lam_max = lam_log.min(), lam_log.max()

    dE_range = dE_max - dE_min if dE_max > dE_min else 1.0
    lam_range = lam_max - lam_min if lam_max > lam_min else 1.0

    def to_col(lam): return int((np.log10(lam) - lam_min) / lam_range * (width - 1))
    def to_row(dE):  return int((dE_max - dE) / dE_range * (height - 1))

    grid = [[' '] * width for _ in range(height)]

    # Подогнанная кривая
    if fit is not None:
        model   = _make_model(fit.n_terms)
        lam_fine = np.logspace(lam_min, lam_max, width * 4)
        dE_fine  = model(lam_fine, fit.deg_phi, *fit.coeffs)
        for lf, df in zip(lam_fine, dE_fine):
            c = to_col(lf)
            r = to_row(df)
            if 0 <= r < height and 0 <= c < width:
                if grid[r][c] == ' ':
                    grid[r][c] = '─'

    # Точки данных
    for lv, dv in zip(lambdas, delta_Es):
        c = to_col(lv)
        r = to_row(dv)
        if 0 <= r < height and 0 <= c < width:
            grid[r][c] = '●'

    lines = []
    for i, row in enumerate(grid):
        dE_label = dE_max - i * dE_range / (height - 1)
        prefix = f"{dE_label:6.3f} │"
        lines.append(prefix + ''.join(row))

    lines.append(f"       └{'─'*width}")
    lines.append(f"         log₁₀λ: {lam_min:.1f} → {lam_max:.1f}")
    collapse_str = (
        "  ╔═ SPECTRAL COLLAPSE ═╗" if (fit and fit.spectral_collapse) else ""
    )
    if collapse_str:
        lines.append(collapse_str)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis test
# ─────────────────────────────────────────────────────────────────────────────

def spectral_collapse_test(
    fit:             FitResult,
    sigma_threshold: float = SIGMA_COLLAPSE,
) -> Dict:
    """
    Тест гипотезы  H₀: a₁ = 0 (rank ≥ 1)  vs  H₁: a₁ ≠ 0 (rank = 0).

    Если bootstrap_ci задан в fit, использует его как основной критерий.
    Иначе — sigma-test по ковариационной матрице.

    Returns dict: decision, z_score, a1, sigma_a1, p_value, method
    """
    import scipy.stats as _stats

    a1 = fit.a1

    # Bootstrap CI (приоритет)
    if fit.bootstrap_ci is not None:
        mean, std, lower, upper = fit.bootstrap_ci
        collapse = (lower <= 0.0 <= upper)
        decision = ("RANK_GE1 (a₁ ≈ 0, spectral collapse)"
                    if collapse else
                    "RANK_0  (a₁ ≠ 0, no collapse)")
        z_score  = abs(mean) / std if std > 0 else math.inf
        p_value  = 2.0 * (1.0 - _stats.norm.cdf(z_score))
        return {
            "decision": decision,
            "z_score":  z_score,
            "a1":       mean,
            "sigma_a1": std,
            "p_value":  p_value,
            "method":   "bootstrap",
        }

    # Sigma-test из cov
    if (fit.cov is not None
            and fit.cov.shape[0] > 1
            and not math.isnan(float(fit.cov[1, 1]))
            and float(fit.cov[1, 1]) >= 0):
        sigma_a1 = math.sqrt(float(fit.cov[1, 1]))
        if sigma_a1 > 0:
            z_score = abs(a1) / sigma_a1
            p_value = 2.0 * (1.0 - _stats.norm.cdf(z_score))
            collapse = z_score < sigma_threshold
            decision = ("RANK_GE1 (a₁ ≈ 0, spectral collapse)"
                        if collapse else
                        "RANK_0  (a₁ ≠ 0, no collapse)")
            return {
                "decision": decision, "z_score": z_score,
                "a1": a1, "sigma_a1": sigma_a1,
                "p_value": p_value, "method": "sigma_test",
            }

    return {
        "decision":  "INCONCLUSIVE (fallback fit)",
        "z_score":   math.nan, "a1": a1,
        "sigma_a1":  math.nan, "p_value": math.nan,
        "method":    "relative_threshold",
    }


# ─────────────────────────────────────────────────────────────────────────────
# BRIDGE: crystal_input_v3  [NEW v2.0]
# ─────────────────────────────────────────────────────────────────────────────

def strain_collapse_bridge(
    p_holes:         float = 0.16,
    interface_boost: float = 1.0,
    n_terms:         int   = 3,
    verbose:         bool  = True,
) -> Optional[Dict]:
    """
    [BRIDGE: crystal_input_v3 ↔ deformation.py]

    Запускает TcDeformationAdapter из crystal_input_v3, подгоняет
    асимптотический ряд Tc(ε)/Tc_bulk через полный DeformationEngine
    (включая bootstrap), возвращает результат.

    Интерпретация:
      a₁ ≠ 0 → Tc линейно чувствительна к деформации (Harrison актив)
      a₁ ≈ 0 → Strain Collapse: интерфейсный boost доминирует над strain

    Parameters
    ----------
    p_holes, interface_boost : параметры кристалла
    n_terms                  : члены асимптотического ряда
    verbose                  : печатать ли результат

    Returns
    -------
    dict с ключами: fit, test, lambdas, delta_Es  или None при ImportError
    """
    try:
        from crystal_input_v3 import TcDeformationAdapter
    except ImportError:
        logger.warning("crystal_input_v3 не найден — BRIDGE недоступен.")
        return None

    adapter = TcDeformationAdapter(p_holes=p_holes,
                                   interface_boost=interface_boost)
    lambdas, delta_Es = adapter.build_sweep()

    # Выбор n_terms через CV
    n_cv = cross_validate_n_terms(lambdas, delta_Es, max_terms=4)
    logger.info("BRIDGE CV выбрал n_terms=%d", n_cv)
    n_terms = n_cv

    fit  = fit_asymptotic_series(lambdas, delta_Es, n_terms=n_terms)
    bsci = bootstrap_uncertainty(lambdas, delta_Es, n_terms=n_terms)
    fit.bootstrap_ci = bsci
    test = spectral_collapse_test(fit)

    if verbose:
        print(f"\n{'═'*62}")
        print(f"  BRIDGE: Tc(ε) Deformation | p={p_holes:.3f}  boost={interface_boost:.2f}")
        print(f"{'═'*62}")
        print(fit.summary())
        print(f"\n  Hypothesis test: {test['decision']}")
        print(f"  z-score = {test['z_score']:.2f}  p-value = {test['p_value']:.4f}")
        print(f"  Method:  {test['method']}")
        print(f"{'═'*62}\n")
        print(ascii_spectral_plot(lambdas, delta_Es, fit))
        print()

    return {"fit": fit, "test": test,
            "lambdas": lambdas, "delta_Es": delta_Es}


# ─────────────────────────────────────────────────────────────────────────────
# DeformationEngine — главный класс
# ─────────────────────────────────────────────────────────────────────────────

class DeformationEngine:
    """
    Asymptotic Deformation Engine для эллиптических кривых.

    Pipeline:
      1. Lambda sweep  через MHR variational solver
      2. Фильтрация (адиабатический порог, сходимость)
      3. CV-выбор n_terms
      4. Подгонка ΔE(λ) = deg φ + a₁/λ + …
      5. Bootstrap CI на a₁
      6. Spectral Collapse detection → rank signal

    Parameters
    ----------
    curve_params : dict с ключами:
        conductor  (int)  — кондуктор E
        deg_phi    (int)  — ожидаемый модулярный степень (prior, optional)
        rank       (int)  — известный rank (для валидации, optional)
        label      (str)  — метка, e.g. \"11a1\"
    solver : callable(λ, params) → (delta_E, converged, iters, residual)
        Если None — используется mock solver.
    """

    def __init__(
        self,
        curve_params: Dict,
        solver:       Optional[Callable] = None,
    ):
        self.params   = dict(curve_params)
        self._solver  = solver or self._mock_solver
        self.raw_data: List[DeformationResult] = []
        logger.info(
            "DeformationEngine: curve=%s  conductor=%s  deg_phi=%s  rank=%s",
            self.params.get("label",     "?"),
            self.params.get("conductor", "?"),
            self.params.get("deg_phi",   "?"),
            self.params.get("rank",      "?"),
        )

    # ── Sweep ────────────────────────────────────────────────────────────────

    def run_sweep(
        self,
        lambda_range: Optional[np.ndarray] = None,
    ) -> List[DeformationResult]:
        """
        Запустить MHR solver для каждого λ в lambda_range.

        Точки с λ < LAMBDA_ADIABATIC_MIN сохраняются в raw_data, но
        помечаются как невалидные и исключаются из фитирования.
        """
        if lambda_range is None:
            lambda_range = DEFAULT_LAMBDA_RANGE

        self.raw_data = []
        for lmbda in lambda_range:
            result = self._run_single(float(lmbda))
            self.raw_data.append(result)
            status = ("OK"        if result.is_valid() else
                      "ADIABATIC" if lmbda < LAMBDA_ADIABATIC_MIN else
                      "FAILED")
            logger.debug("λ=%10.2f  ΔE=%12.6f  [%s]",
                         lmbda, result.delta_E, status)

        n_valid = sum(r.is_valid() for r in self.raw_data)
        logger.info("Sweep: %d / %d точек валидны.", n_valid, len(self.raw_data))
        return self.raw_data

    def _run_single(self, lmbda: float) -> DeformationResult:
        try:
            out = self._solver(lmbda, self.params)
            if isinstance(out, (int, float)):
                return DeformationResult(lmbda=lmbda, delta_E=float(out))
            delta_E, converged, iters, residual = out
            return DeformationResult(
                lmbda      = lmbda,
                delta_E    = float(delta_E),
                converged  = bool(converged),
                iterations = int(iters),
                residual   = float(residual),
            )
        except Exception as exc:
            logger.error("Solver failed at λ=%.2f: %s", lmbda, exc)
            return DeformationResult(
                lmbda=lmbda, delta_E=math.nan,
                converged=False, note=str(exc),
            )

    # ── Фильтрация ───────────────────────────────────────────────────────────

    def valid_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Валидные точки: сошлись и λ ≥ LAMBDA_ADIABATIC_MIN."""
        pts = [r for r in self.raw_data
               if r.is_valid() and not math.isnan(r.delta_E)]
        if not pts:
            raise RuntimeError("Нет валидных точек после фильтрации.")
        return (np.array([r.lmbda   for r in pts]),
                np.array([r.delta_E for r in pts]))

    # ── Фитирование ──────────────────────────────────────────────────────────

    def fit(
        self,
        n_terms:     Optional[int] = None,
        auto_cv:     bool = True,
        do_bootstrap: bool = True,
        n_bootstrap:  int  = 400,
    ) -> FitResult:
        """
        Подогнать асимптотический ряд к валидным данным.

        Parameters
        ----------
        n_terms      : число слагаемых.  Если None и auto_cv=True — выбирается CV.
        auto_cv      : автоматически выбрать n_terms через k-fold CV
        do_bootstrap : вычислять ли bootstrap CI (рекомендуется)
        n_bootstrap  : число bootstrap реплик
        """
        lambdas, delta_Es = self.valid_data()
        prior_deg = self.params.get("deg_phi")

        if n_terms is None and auto_cv:
            n_terms = cross_validate_n_terms(lambdas, delta_Es)
        elif n_terms is None:
            n_terms = 3

        p0 = ([float(prior_deg)] + [0.0] * n_terms
              if prior_deg is not None else None)

        fit = fit_asymptotic_series(lambdas, delta_Es,
                                    n_terms=n_terms, p0=p0)

        if do_bootstrap:
            bsci = bootstrap_uncertainty(
                lambdas, delta_Es, n_terms=n_terms, n_bootstrap=n_bootstrap,
            )
            fit.bootstrap_ci = bsci

        logger.info("Fit:\n%s", fit.summary())
        return fit

    # ── Полный пайплайн ──────────────────────────────────────────────────────

    def analyse(
        self,
        lambda_range:  Optional[np.ndarray] = None,
        n_terms:       Optional[int] = None,
        auto_cv:       bool  = True,
        do_bootstrap:  bool  = True,
        n_bootstrap:   int   = 400,
        sigma:         float = SIGMA_COLLAPSE,
        verbose:       bool  = True,
        show_plot:     bool  = True,
    ) -> Dict:
        """
        Полный пайплайн: sweep → filter → CV → fit → bootstrap → rank test.

        Returns dict: fit, test, rank_signal, lambdas, delta_Es
        """
        self.run_sweep(lambda_range)
        lambdas, delta_Es = self.valid_data()

        fit  = self.fit(n_terms=n_terms, auto_cv=auto_cv,
                        do_bootstrap=do_bootstrap, n_bootstrap=n_bootstrap)
        test = spectral_collapse_test(fit, sigma_threshold=sigma)

        if verbose:
            label = self.params.get("label", "E")
            print(f"\n{'═'*60}")
            print(f"  Deformation Analysis — {label}")
            print(f"{'═'*60}")
            print(fit.summary())
            print(f"\n  Hypothesis test (H₀: rank ≥ 1):")
            print(f"    decision  = {test['decision']}")
            print(f"    z-score   = {test['z_score']:.2f}")
            print(f"    p-value   = {test['p_value']:.4f}")
            print(f"    method    = {test['method']}")
            known_rank = self.params.get("rank")
            if known_rank is not None:
                expected = "RANK_GE1" if known_rank >= 1 else "RANK_0"
                match    = "✅" if expected in test["decision"] else "❌"
                print(f"    known rank = {known_rank}  {match}")
            print(f"{'═'*60}\n")

        if show_plot:
            print(ascii_spectral_plot(lambdas, delta_Es, fit))
            print()

        return {
            "fit":         fit,
            "test":        test,
            "rank_signal": test["decision"],
            "lambdas":     lambdas,
            "delta_Es":    delta_Es,
        }

    # ── Mock solver ──────────────────────────────────────────────────────────

    @staticmethod
    def _mock_solver(lmbda: float, params: Dict):
        """
        Mock MHR solver для тестов без mhr-numerics.

        Симулирует  ΔE(λ) = deg_phi + a1/λ + 0.5/λ²  + N(0, 1e-4).
          rank = 0  →  a1 = deg_phi  (без коллапса)
          rank ≥ 1  →  a1 = 0        (спектральный коллапс)

        Реальный интерфейс: variational.mhr_solver(lmbda, params)
        → (delta_E, converged, iterations, residual)
        """
        deg_phi = params.get("deg_phi", 2)
        rank    = params.get("rank",    0)
        a1_true = 0.0 if rank >= 1 else float(deg_phi)

        delta_E = (deg_phi
                   + a1_true / lmbda
                   + 0.5    / lmbda ** 2
                   + np.random.normal(0, 1e-4))
        return delta_E, True, 42, 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Быстрый smoke-тест
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== deformation.py v2.0 — smoke test ===\n")

    # Кривая rank=0 (11a1, deg=2)
    print("── Кривая 11a1 (rank=0, deg_phi=2) ──")
    eng0 = DeformationEngine(
        {"label": "11a1", "conductor": 11, "deg_phi": 2, "rank": 0}
    )
    eng0.analyse(do_bootstrap=True, n_bootstrap=200, show_plot=True)

    # Кривая rank=1 (37a, deg=2)
    print("── Кривая 37a (rank=1, deg_phi=2) ──")
    eng1 = DeformationEngine(
        {"label": "37a", "conductor": 37, "deg_phi": 2, "rank": 1}
    )
    eng1.analyse(do_bootstrap=True, n_bootstrap=200, show_plot=True)

    # BRIDGE с crystal_input_v3
    print("── BRIDGE: strain_collapse_bridge() ──")
    strain_collapse_bridge(p_holes=0.16, interface_boost=1.0)
