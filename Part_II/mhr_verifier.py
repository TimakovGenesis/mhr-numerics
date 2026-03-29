"""
mhr_verifier.py  —  MHR Asymptotic Coefficient Verifier  v1.0
════════════════════════════════════════════════════════════════════════════════
Part of mhr-numerics library.
Companion to: "Spectral gap and modular degree of elliptic curves, Part II"
DOI: 10.5281/zenodo.19165246

Validates the rank signal coefficient a₁ against the analytic prediction
a₁ = −D²/π² (Part II, Proposition 5.1).

════════════════════════════════════════════════════════════════════════════════
Математическое основание [F: Part II, Proposition 5.1]
════════════════════════════════════════════════════════════════════════════════

  rank = 0  →  a₁ = -D²/π²  (строго отрицательно)
               ΔE(λ) → deg(φ) монотонно снизу
               z-score > 10  (с реальным solver > 20)

  rank ≥ 1  →  a₁ ≡ 0  (тождественно, Bloch band width → 0)
               ΔE(λ) → 0  экспоненциально
               z-score < 2  (обычно < 1)

Граница предсказания из Part II (Table a1_bootstrap):
  rank=0: z > 89  (с полным solver, λ до 1e5)
  rank≥1: z < 1.7

С реальным mhr_real_solver [F: verify_real_solver.py]:
  rank=0: z ≈ 25-35  (λ ∈ [316, 1e5], 40 точек)
  rank≥1: z → 0      (Bloch band width → 0)

════════════════════════════════════════════════════════════════════════════════
Режимы верификации
════════════════════════════════════════════════════════════════════════════════

  РЕЖИМ 1: verify_a1_prediction(fit, deg_phi)
    Быстрый — нужен только готовый FitResult.
    Сравнивает a₁_measured с a₁_theory = -D²/π².

  РЕЖИМ 2: quick_rank_signal(fit)
    Только сигнал по знаку и z-score — без теоретического значения.
    Для случаев когда deg_phi неизвестен.

════════════════════════════════════════════════════════════════════════════════
Вердикт
════════════════════════════════════════════════════════════════════════════════

  "RANK_GE1"     — collapse подтверждён, a₁ ≈ 0, rank ≥ 1
  "RANK_ZERO"    — нет collapse, a₁ << 0, rank = 0
  "WEAK"         — слабый сигнал, нужно больше данных
  "INCONCLUSIVE" — недостаточно информации

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Константы [F: Part II, Prop.5.1 + verify_real_solver.py]
# ─────────────────────────────────────────────────────────────────────────────

# Теоретическое значение a₁ для rank=0
def _a1_theory(deg_phi: float) -> float:
    """a₁ = -D²/π²  [F: Part II, Proposition 5.1(i)]"""
    return -(deg_phi ** 2) / (math.pi ** 2)

# Пороги z-score [F: verify_real_solver.py + Part II]
Z_ATTACK_THRESHOLD = 2.0    # z < этого → collapse (rank≥1)
Z_SAFE_THRESHOLD   = 10.0   # z > этого → безопасно (rank=0)

# Допустимое отклонение a₁ от теории (поправки O(λ⁻²) дают ~10%)
A1_DEVIATION_MAX = 0.20     # 20% — с запасом на конечный lambda_range
A1_DEVIATION_WARN = 0.10    # 10% — предупреждение

# Минимальный z-score для уверенного "RANK_ZERO"
Z_SAFE_MIN = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Структура результата
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VerifierResult:
    """
    Результат верификации — количественный критерий ранга.

    verdict:        "RANK_GE1" / "RANK_ZERO" / "WEAK" / "INCONCLUSIVE"
    confidence:     "HIGH" / "MEDIUM" / "LOW"
    a1_measured:    float — измеренный коэффициент
    a1_theory:      float — теоретический (-D²/π² для rank=0, 0 для rank≥1)
    a1_deviation_pct: float — отклонение от теории в %
    z_score:        float — z-score из bootstrap
    spectral_collapse: bool — флаг коллапса
    bootstrap_ci:   tuple — (lower95, upper95)
    action:         str — конкретная рекомендация
    note:           str — дополнительные детали
    """
    verdict:           str
    confidence:        str
    a1_measured:       float
    a1_theory:         float
    a1_deviation_pct:  float
    z_score:           float
    spectral_collapse: bool
    bootstrap_ci:      Tuple[float, float]
    action:            str
    note:              str = ""

    def is_rank_ge1(self) -> bool:
        return self.verdict == "RANK_GE1"

    def summary(self) -> str:
        lines = [
            f"  Вердикт:      {self.verdict}  [{self.confidence}]",
            f"  a₁ измерен:   {self.a1_measured:.6f}",
            f"  a₁ теория:    {self.a1_theory:.6f}",
            f"  Отклонение:   {self.a1_deviation_pct:.1f}%",
            f"  z-score:      {self.z_score:.1f}",
            f"  Collapse:     {self.spectral_collapse}",
            f"  95% CI:       [{self.bootstrap_ci[0]:.3e}, {self.bootstrap_ci[1]:.3e}]",
            f"  Действие:     {self.action}",
        ]
        if self.note:
            lines.append(f"  Примечание:   {self.note}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# РЕЖИМ 1: verify_a1_prediction — быстрый, по готовому FitResult
# ─────────────────────────────────────────────────────────────────────────────

def verify_a1_prediction(
    fit,
    deg_phi:    float,
    z_attack:   float = Z_ATTACK_THRESHOLD,
    z_safe:     float = Z_SAFE_THRESHOLD,
    verbose:    bool  = False,
) -> VerifierResult:
    """
    Быстрая верификация по готовому FitResult.

    Сравнивает измеренный a₁ с теоретическим -D²/π².
    Не требует повторного запуска solver.

    Parameters
    ----------
    fit      : FitResult из deformation.fit_asymptotic_series()
    deg_phi  : модулярная степень кривой D
    z_attack : z-score ниже которого → collapse (ATTACK)
    z_safe   : z-score выше которого → безопасно (SAFE)
    verbose  : печатать отчёт

    Returns
    -------
    VerifierResult
    """
    a1_meas = fit.a1
    a1_th   = _a1_theory(deg_phi)

    # z-score из bootstrap CI (приоритет) или cov матрицы
    z_score = _extract_z_score(fit)

    # Отклонение от теории
    if abs(a1_th) > 1e-10:
        dev_pct = abs(a1_meas - a1_th) / abs(a1_th) * 100.0
    else:
        dev_pct = abs(a1_meas) * 100.0

    # Bootstrap CI
    if fit.bootstrap_ci is not None:
        ci = (fit.bootstrap_ci[2], fit.bootstrap_ci[3])
    else:
        ci = (float('nan'), float('nan'))

    collapse = fit.spectral_collapse

    # Вердикт
    verdict, confidence, action, note = _decide_verdict(
        a1_meas, a1_th, dev_pct, z_score, collapse, ci,
        z_attack, z_safe,
    )

    result = VerifierResult(
        verdict           = verdict,
        confidence        = confidence,
        a1_measured       = a1_meas,
        a1_theory         = a1_th,
        a1_deviation_pct  = dev_pct,
        z_score           = z_score,
        spectral_collapse = collapse,
        bootstrap_ci      = ci,
        action            = action,
        note              = note,
    )

    if verbose:
        _print_result(result, "verify_a1_prediction")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# РЕЖИМ 2b: direct_collapse_check — по сырым dE данным (без fit)
# ─────────────────────────────────────────────────────────────────────────────

# Порог прямого collapse: если max(dEs) < этого → экспоненциальный коллапс
DIRECT_COLLAPSE_THRESHOLD = 0.01

def direct_collapse_check(
    delta_Es:  np.ndarray,
    deg_phi:   float        = 0.0,
    threshold: float        = DIRECT_COLLAPSE_THRESHOLD,
    verbose:   bool         = False,
) -> VerifierResult:
    """
    Прямой детектор collapse по сырым значениям ΔE_proxy.

    Используется когда ΔE настолько близко к 0, что fit_asymptotic_series
    фитирует шум (rank≥1 с Bloch band width → 0).

    Логика [F: mhr_real_solver.py + verify_real_solver.py]:
      rank=0: dEs ∈ [1.99, 2.00]  → max(dEs) >> threshold
      rank≥1: dEs ∈ [0, 1e-4]    → max(dEs) << threshold

    Parameters
    ----------
    delta_Es  : numpy array значений ΔE_proxy из sweep
    deg_phi   : модулярная степень D (для вычисления a1_theory)
    threshold : порог (default 0.01, разрыв между rank=0 и rank≥1 огромный)
    verbose   : печатать

    Returns
    -------
    VerifierResult
    """
    max_dE = float(np.max(np.abs(delta_Es)))
    mean_dE = float(np.mean(np.abs(delta_Es)))
    a1_th   = _a1_theory(deg_phi) if deg_phi > 0 else 0.0

    if max_dE < threshold:
        # Прямой коллапс — rank≥1
        verdict    = "RANK_GE1"
        confidence = "HIGH"
        note       = (f"Прямой collapse: max(|ΔE|)={max_dE:.2e} "
                      f"< {threshold} — Bloch band width → 0")
        action     = "Запустить audit_engine_v2.run_hnp_attack() → извлечь ключ"
        # z-score для прямого collapse → ∞ (нет fit uncertainty)
        z_score = float('inf')
        ci = (float('-inf'), float('inf'))
    else:
        # Нет коллапса — rank=0
        verdict    = "RANK_ZERO"
        confidence = "HIGH"
        note       = (f"Нет collapse: max(|ΔE|)={max_dE:.4f} "
                      f"> {threshold}")
        action     = "Адрес безопасен"
        z_score    = float('inf')
        ci         = (float('nan'), float('nan'))

    result = VerifierResult(
        verdict           = verdict,
        confidence        = confidence,
        a1_measured       = mean_dE if max_dE < threshold else -mean_dE,
        a1_theory         = a1_th,
        a1_deviation_pct  = 0.0,
        z_score           = z_score,
        spectral_collapse = (max_dE < threshold),
        bootstrap_ci      = ci,
        action            = action,
        note              = note,
    )

    if verbose:
        _print_result(result, "direct_collapse_check")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# РЕЖИМ 3: quick_rank_signal — только знак и z-score
# ─────────────────────────────────────────────────────────────────────────────

def quick_rank_signal(
    fit,
    z_attack: float = Z_ATTACK_THRESHOLD,
    z_safe:   float = Z_SAFE_THRESHOLD,
    verbose:  bool  = False,
) -> VerifierResult:
    """
    Быстрый сигнал по знаку a₁ и z-score — без теоретического значения.

    Используется когда deg_phi неизвестен (нет данных о кривой).
    Менее точен чем verify_a1_prediction, но не требует deg_phi.

    Parameters
    ----------
    fit      : FitResult
    z_attack : порог для ATTACK
    z_safe   : порог для SAFE
    verbose  : печатать

    Returns
    -------
    VerifierResult
    """
    a1_meas  = fit.a1
    z_score  = _extract_z_score(fit)
    collapse = fit.spectral_collapse

    if fit.bootstrap_ci is not None:
        ci = (fit.bootstrap_ci[2], fit.bootstrap_ci[3])
    else:
        ci = (float('nan'), float('nan'))

    # Без теории используем 0 как ориентир
    a1_th   = 0.0
    dev_pct = 0.0  # неприменимо

    verdict, confidence, action, note = _decide_verdict(
        a1_meas, a1_th, dev_pct, z_score, collapse, ci,
        z_attack, z_safe, has_theory=False,
    )

    result = VerifierResult(
        verdict           = verdict,
        confidence        = confidence,
        a1_measured       = a1_meas,
        a1_theory         = a1_th,
        a1_deviation_pct  = dev_pct,
        z_score           = z_score,
        spectral_collapse = collapse,
        bootstrap_ci      = ci,
        action            = action,
        note              = "(quick mode: deg_phi неизвестен) " + note,
    )

    if verbose:
        _print_result(result, "quick_rank_signal")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _extract_z_score(fit) -> float:
    """Извлекает z-score из FitResult (bootstrap приоритет → cov → nan)."""
    if fit.bootstrap_ci is not None:
        mean, std = fit.bootstrap_ci[0], fit.bootstrap_ci[1]
        if std > 1e-15:
            return abs(mean) / std
    if (fit.cov is not None and fit.cov.shape[0] > 1
            and not math.isnan(float(fit.cov[1, 1]))
            and float(fit.cov[1, 1]) >= 0):
        sigma = math.sqrt(float(fit.cov[1, 1]))
        if sigma > 1e-15:
            return abs(fit.a1) / sigma
    return float('nan')


def _decide_verdict(
    a1_meas:    float,
    a1_th:      float,
    dev_pct:    float,
    z_score:    float,
    collapse:   bool,
    ci:         Tuple[float, float],
    z_attack:   float,
    z_safe:     float,
    has_theory: bool = True,
) -> Tuple[str, str, str, str]:
    """
    Логика принятия решения.

    Приоритет критериев:
      1. z_score и collapse (надёжнее всего)
      2. Знак a₁
      3. Отклонение от теории (только если has_theory)

    Returns: (verdict, confidence, action, note)
    """
    note = ""

    # ── Данных нет ────────────────────────────────────────────────────────────
    if math.isnan(a1_meas) or math.isnan(z_score):
        return ("INCONCLUSIVE", "LOW",
                "Собери больше подписей и повтори sweep",
                "Нет данных для z-score")

    # ── ATTACK: collapse подтверждён ──────────────────────────────────────────
    if collapse and z_score < z_attack:
        # Проверяем что оба конца CI охватывают 0
        ci_covers_zero = (not math.isnan(ci[0])
                          and ci[0] <= 0.0 <= ci[1])
        if ci_covers_zero:
            confidence = "HIGH"
        else:
            confidence = "MEDIUM"
            note = "CI не покрывает 0 — нетипично для collapse"

        return ("RANK_GE1", confidence,
                "Запустить audit_engine_v2.run_hnp_attack() → извлечь ключ",
                note)

    # ── SAFE: явный rank=0 сигнал ─────────────────────────────────────────────
    if not collapse and z_score > z_safe and a1_meas < 0:
        if has_theory and dev_pct < A1_DEVIATION_WARN:
            confidence = "HIGH"
            note = f"a₁ совпадает с -D²/π² с точностью {dev_pct:.1f}%"
        elif has_theory and dev_pct < A1_DEVIATION_MAX:
            confidence = "HIGH"
            note = (f"a₁ отклонение {dev_pct:.1f}% — норма "
                    f"(поправки O(λ⁻²) дают ~10%)")
        elif has_theory and dev_pct >= A1_DEVIATION_MAX:
            confidence = "MEDIUM"
            note = (f"a₁ отклонение {dev_pct:.1f}% > {A1_DEVIATION_MAX*100:.0f}% "
                    f"— увеличь lambda_range")
        else:
            confidence = "HIGH"
        return ("RANK_ZERO", confidence,
                "Кривая rank=0 — спектральный коллапс отсутствует",
                note)

    # ── WEAK: граничная зона ──────────────────────────────────────────────────
    if z_attack <= z_score <= z_safe:
        if collapse:
            action = "Расширь lambda_range или добавь точек для подтверждения"
            return ("WEAK", "LOW", action,
                    f"z={z_score:.1f} в граничной зоне [{z_attack}, {z_safe}]")
        else:
            action = "Расширь lambda_range — граничная зона"
            return ("WEAK", "LOW", action,
                    f"z={z_score:.1f} недостаточно для SAFE")

    # ── ATTACK без collapse флага (но z близко к 0) ───────────────────────────
    if z_score < z_attack and not collapse:
        # Редкий случай: collapse.spectral_collapse=False но z очень мал
        note = "z-score мал, но collapse=False — возможна ошибка порога"
        return ("WEAK", "LOW",
                "Проверить EPSILON_COLLAPSE_REL в deformation.py",
                note)

    # ── Fallback ──────────────────────────────────────────────────────────────
    return ("INCONCLUSIVE", "LOW",
            "Запусти калибровку: python a1_calibrator.py",
            f"z={z_score:.1f} collapse={collapse}")


def _inconclusive(reason: str) -> VerifierResult:
    """Возвращает INCONCLUSIVE при ошибке."""
    return VerifierResult(
        verdict           = "INCONCLUSIVE",
        confidence        = "LOW",
        a1_measured       = float('nan'),
        a1_theory         = float('nan'),
        a1_deviation_pct  = float('nan'),
        z_score           = float('nan'),
        spectral_collapse = False,
        bootstrap_ci      = (float('nan'), float('nan')),
        action            = "Проверь логи и входные данные",
        note              = reason,
    )


def _print_result(result: VerifierResult, mode: str):
    """Форматированный вывод результата."""
    icon = {"RANK_GE1": "🎯", "RANK_ZERO": "✅", "WEAK": "⚠️ ",
            "INCONCLUSIVE": "❓"}.get(result.verdict, "?")
    print(f"\n  {'═'*54}")
    print(f"  MHR Verifier [{mode}]")
    print(f"  {'═'*54}")
    print(f"  {icon} {result.verdict}  [{result.confidence}]")
    print(f"  {'─'*54}")
    print(result.summary())
    print(f"  {'═'*54}")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO_VERIFY — главный публичный API
# ─────────────────────────────────────────────────────────────────────────────

def auto_verify(
    fit          = None,
    delta_Es:    Optional[np.ndarray] = None,
    deg_phi:     float                = 0.0,
    verbose:     bool                 = True,
) -> VerifierResult:
    """
    Умная верификация — автоматически выбирает режим.

    Приоритет:
      1. Если delta_Es все < DIRECT_COLLAPSE_THRESHOLD → direct_collapse_check
         (rank≥1 с Bloch band width → 0, fit не нужен)
      2. Если fit доступен и deg_phi известен → verify_a1_prediction
      3. Если fit доступен без deg_phi → quick_rank_signal

    Parameters
    ----------
    fit       : FitResult (опционально)
    delta_Es  : numpy array сырых ΔE_proxy (опционально)
    deg_phi   : модулярная степень D (0 = неизвестен)
    verbose   : печатать результат

    Returns
    -------
    VerifierResult
    """
    # Режим 1: прямой collapse по сырым данным
    if delta_Es is not None and len(delta_Es) > 0:
        max_dE = float(np.max(np.abs(delta_Es)))
        if max_dE < DIRECT_COLLAPSE_THRESHOLD:
            return direct_collapse_check(delta_Es, deg_phi, verbose=verbose)

    # Режим 2: fit + теоретическое значение
    if fit is not None and deg_phi > 0:
        return verify_a1_prediction(fit, deg_phi, verbose=verbose)

    # Режим 3: fit без deg_phi
    if fit is not None:
        return quick_rank_signal(fit, verbose=verbose)

    return _inconclusive("Нет данных: нужен fit или delta_Es")


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная таблица границ
# ─────────────────────────────────────────────────────────────────────────────

def print_thresholds():
    """Печатает таблицу текущих порогов принятия решений."""
    print(f"\n  Пороги MHR Verifier [F: Part II + verify_real_solver.py]:")
    print(f"  {'Параметр':<28} {'Значение':>10}  Смысл")
    print(f"  {'─'*60}")
    print(f"  {'Z_ATTACK_THRESHOLD':<28} {Z_ATTACK_THRESHOLD:>10.1f}  "
          f"z < X → collapse → ATTACK")
    print(f"  {'Z_SAFE_THRESHOLD':<28} {Z_SAFE_THRESHOLD:>10.1f}  "
          f"z > X → rank=0 → SAFE")
    print(f"  {'A1_DEVIATION_MAX':<28} {A1_DEVIATION_MAX*100:>9.0f}%  "
          f"макс отклонение от -D²/π²")
    print(f"  {'A1_DEVIATION_WARN':<28} {A1_DEVIATION_WARN*100:>9.0f}%  "
          f"предупреждение")
    print()
    print(f"  Реальные z-score [F: verify_real_solver.py, MacBook]:")
    print(f"    rank=0 (37a1, D=2, λ∈[316,1e5]): z ≈ 25-35")
    print(f"    rank=0 (11a1, D=1, λ∈[316,1e5]): z ≈ 20-25")
    print(f"    rank≥1 (37b1):                    z ≈ 0")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI / smoke-тест
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    logging.basicConfig(level=logging.WARNING)

    print("═" * 60)
    print("  mhr_verifier.py v1.0 — smoke-тест")
    print("═" * 60)

    print_thresholds()

    # Импортируем нужные модули
    try:
        import numpy as np
        from deformation import (
            DeformationEngine, fit_asymptotic_series, bootstrap_uncertainty
        )
        try:
            from mhr_real_solver import mhr_real_solver
            USE_REAL = True
            print("  [OK] mhr_real_solver доступен — используем реальный solver")
        except ImportError:
            USE_REAL = False
            print("  [WARN] mhr_real_solver не найден — используем mock solver")

    except ImportError as e:
        print(f"  [ERR] {e}")
        sys.exit(1)

    lam_range = np.logspace(2.5, 5, 30)

    # ── Тест 1: 37a1 (rank=0, D=2) → ожидаем SAFE ────────────────────────────
    print("\n  ── Тест 1: 37a1 (rank=0, D=2) ──")
    solver1 = mhr_real_solver if USE_REAL else None

    eng1 = DeformationEngine(
        {"label": "37a1", "deg_phi": 2, "rank": 0},
        solver=solver1
    )
    eng1.run_sweep(lam_range)
    lams1, dEs1 = eng1.valid_data()
    fit1 = fit_asymptotic_series(lams1, dEs1, n_terms=3)
    bsci1 = bootstrap_uncertainty(lams1, dEs1, n_terms=3, n_bootstrap=150, seed=42)
    fit1.bootstrap_ci = bsci1

    res1 = verify_a1_prediction(fit1, deg_phi=2, verbose=True)
    t1_ok = res1.verdict == "RANK_ZERO"
    print(f"  Тест 1: {'✅ PASS' if t1_ok else '❌ FAIL'} (ожидалось SAFE)")

    # ── Тест 2: 37b1 (rank=1, D=2) → ожидаем ATTACK ──────────────────────────
    print("\n  ── Тест 2: 37b1 (rank=1, D=2) ──")
    solver2 = mhr_real_solver if USE_REAL else None

    eng2 = DeformationEngine(
        {"label": "37b1", "deg_phi": 2, "rank": 1},
        solver=solver2
    )
    eng2.run_sweep(lam_range)
    lams2, dEs2 = eng2.valid_data()
    fit2 = fit_asymptotic_series(lams2, dEs2, n_terms=2)
    bsci2 = bootstrap_uncertainty(lams2, dEs2, n_terms=2, n_bootstrap=150, seed=42)
    fit2.bootstrap_ci = bsci2

    res2 = verify_a1_prediction(fit2, deg_phi=2, verbose=True)
    t2_ok = res2.verdict == "RANK_GE1"
    print(f"  Тест 2: {'✅ PASS' if t2_ok else '❌ FAIL'} (ожидалось RANK_GE1)")

    # ── Тест 3: quick_rank_signal (без deg_phi) ───────────────────────────────
    print("\n  ── Тест 3: quick_rank_signal (без deg_phi) ──")
    res3a = quick_rank_signal(fit1, verbose=True)
    res3b = quick_rank_signal(fit2, verbose=True)
    t3_ok = (res3a.verdict == "RANK_ZERO") and (res3b.verdict == "RANK_GE1")
    print(f"  Тест 3: {'✅ PASS' if t3_ok else '❌ FAIL'}")

    # ── Итог ──────────────────────────────────────────────────────────────────
    print(f"\n  {'═'*54}")
    all_ok = t1_ok and t2_ok and t3_ok
    print(f"  ИТОГ: {'✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if all_ok else '❌ ЕСТЬ ОШИБКИ'}")
    print(f"  {'═'*54}")
    sys.exit(0 if all_ok else 1)
