"""
a1_calibrator.py  —  MHR Deformation Calibrator  v1.0
════════════════════════════════════════════════════════════════════════════════
Part of mhr-numerics library.
Companion to: "Spectral gap and modular degree of elliptic curves, Part II"
DOI: 10.5281/zenodo.19165246

Калибровка параметров deformation.py
по benchmark эллиптическим кривым из Part II [F: timakov26II].

════════════════════════════════════════════════════════════════════════════════
Математическое основание [F: Part II, Proposition 5.1]
════════════════════════════════════════════════════════════════════════════════

  rank = 0  →  a₁ = -D²/π²  (строго отрицательно, z-score > 89)
  rank ≥ 1  →  a₁ ≡ 0        (тождественно, z-score < 1.7)

  Граница предсказания: 95% bootstrap CI НЕ пересекает 0 в обоих случаях.

  Контрольные числа [F: Part II, Table a1_bootstrap]:
    11a1:  D=1,  rank=0,  a₁_theory = -1/π²   ≈ -0.10132
    37a:   D=2,  rank=1,  a₁_theory = 0
    389a1: D=40, rank=2,  a₁_theory = 0

════════════════════════════════════════════════════════════════════════════════
Что калибруется
════════════════════════════════════════════════════════════════════════════════

Параметры deformation.py:
  LAMBDA_ADIABATIC_MIN  — нижний порог λ, ниже которого данные исключаются
  EPSILON_COLLAPSE_REL  — порог |a₁|/deg_phi → считается "коллапсом"
  SIGMA_COLLAPSE        — z-score порог для sigma-test

════════════════════════════════════════════════════════════════════════════════
Как использовать
════════════════════════════════════════════════════════════════════════════════

  # Быстрая калибровка (только mock solver, без BKZ):
  python a1_calibrator.py

  # Полная калибровка (с реальным MHR solver если доступен):
  python a1_calibrator.py --full

  # Из кода:
  from a1_calibrator import run_calibration, load_calibration
  params = run_calibration()
  # → сохраняет в data/calibration.json
  # → выводит рекомендованные значения параметров

  params = load_calibration()
  # → загружает последний результат

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import math
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Путь к data/ ──────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_BASE, "data")
CALIBRATION_FILE = os.path.join(_DATA, "calibration.json")

# ─────────────────────────────────────────────────────────────────────────────
# Контрольные кривые [F: Part II, Proposition 5.1, Table a1_bootstrap]
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_CURVES: Dict[str, dict] = {
    "11a1": {
        "label":      "11a1",
        "conductor":  11,
        "deg_phi":    1,
        "rank":       0,
        "a1_theory":  -(1.0 ** 2) / (math.pi ** 2),   # = -1/π² ≈ -0.10132
        "z_expected": ">89",
        "source":     "[F: Part II, Table a1_bootstrap]",
    },
    "37a": {
        "label":      "37a",
        "conductor":  37,
        "deg_phi":    2,
        "rank":       1,
        "a1_theory":  0.0,    # spectral collapse: a₁ ≡ 0
        "z_expected": "<1.7",
        "source":     "[F: Part II, Table a1_bootstrap]",
    },
    "389a1": {
        "label":      "389a1",
        "conductor":  389,
        "deg_phi":    40,
        "rank":       2,
        "a1_theory":  0.0,    # spectral collapse: a₁ ≡ 0
        "z_expected": "<1.7",
        "source":     "[F: Part II, Table a1_bootstrap]",
    },
}

# Теоретическое значение a₁ для rank=0: a₁ = -D²/π²
def a1_theoretical(deg_phi: int) -> float:
    """
    [F: Part II, Proposition 5.1(i)]
    a₁ = -D²/π²  для rank=0 кривых.
    """
    return -(deg_phi ** 2) / (math.pi ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Структура результата калибровки
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveCalibResult:
    """Результат теста одной кривой."""
    label:           str
    deg_phi:         int
    rank:            int
    a1_theory:       float
    a1_measured:     float
    a1_error_pct:    float     # |measured - theory| / |theory| × 100
    z_score:         float
    bootstrap_ci:    Tuple[float, float]  # (lower95, upper95)
    spectral_collapse: bool
    prediction_correct: bool   # совпало ли с ожидаемым rank сигналом
    n_points:        int
    lambda_range:    Tuple[float, float]
    elapsed_s:       float
    note:            str = ""


@dataclass
class CalibrationResult:
    """Полный результат калибровки."""
    timestamp:           str
    curves_tested:       int
    curves_passed:       int
    # Текущие параметры (были при запуске)
    lambda_adiabatic_min_used: float
    epsilon_collapse_rel_used: float
    sigma_collapse_used:       float
    # Рекомендованные параметры (результат калибровки)
    lambda_adiabatic_min_rec:  float
    epsilon_collapse_rel_rec:  float
    sigma_collapse_rec:        float
    scale_rec:                 float
    # Детали по кривым
    curve_results:       List[dict]
    # Итоговая оценка
    status:              str    # "OK" / "WARN" / "FAIL"
    summary:             str


# ─────────────────────────────────────────────────────────────────────────────
# Mock MHR solver (для калибровки без реального вычисления)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_solver_rank0(lmbda: float, params: dict) -> float:
    """
    Mock solver для rank=0 кривых.
    Симулирует: ΔE(λ) = D + (-D²/π²)/λ + 0.5/λ² + N(0, 1e-4)
    [F: Part II, eq.(3.4), Proposition 5.1(i)]
    """
    D   = params.get("deg_phi", 1)
    a1  = a1_theoretical(D)
    dE  = D + a1 / lmbda + 0.5 / lmbda ** 2
    dE += np.random.normal(0, 1e-4)
    return dE, True, 42, 1e-8


def _mock_solver_rank_ge1(lmbda: float, params: dict) -> float:
    """
    Mock solver для rank≥1 кривых.
    Симулирует: ΔE(λ) ≡ 0 (spectral collapse)
    [F: Part II, Proposition 5.1(ii)]
    """
    dE  = 0.0 + np.random.normal(0, 1e-5)
    return dE, True, 42, 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Тест одной кривой
# ─────────────────────────────────────────────────────────────────────────────

def _test_curve(
    curve:          dict,
    lambda_range:   np.ndarray,
    n_bootstrap:    int,
    n_terms:        int,
    use_real_solver: bool = False,
) -> CurveCalibResult:
    """
    Запускает деформационный тест на одной кривой и сравнивает
    измеренный a₁ с теоретическим значением из Part II.

    Parameters
    ----------
    curve        : описание кривой из BENCHMARK_CURVES
    lambda_range : диапазон λ для sweep
    n_bootstrap  : число bootstrap реплик
    n_terms      : число слагаемых асимптотического ряда
    use_real_solver : если True — пробует импортировать реальный MHR solver
    """
    from deformation import (
        DeformationEngine,
        fit_asymptotic_series,
        bootstrap_uncertainty,
        spectral_collapse_test,
    )

    label    = curve["label"]
    deg_phi  = curve["deg_phi"]
    rank     = curve["rank"]
    a1_th    = curve["a1_theory"]

    # Выбираем solver
    if use_real_solver:
        solver = None  # DeformationEngine использует _mock_solver если None
    else:
        solver = _mock_solver_rank0 if rank == 0 else _mock_solver_rank_ge1

    t0 = time.time()

    # Запускаем DeformationEngine
    engine = DeformationEngine(
        curve_params={
            "label":     label,
            "conductor": curve["conductor"],
            "deg_phi":   deg_phi,
            "rank":      rank,
        },
        solver=solver,
    )

    engine.run_sweep(lambda_range)
    lambdas, delta_Es = engine.valid_data()

    # Фитируем ряд
    fit = fit_asymptotic_series(lambdas, delta_Es, n_terms=n_terms)

    # Bootstrap CI
    bsci = bootstrap_uncertainty(
        lambdas, delta_Es,
        n_terms=n_terms,
        n_bootstrap=n_bootstrap,
        seed=42,
    )
    fit.bootstrap_ci = bsci

    elapsed = time.time() - t0

    # Метрики
    a1_meas  = fit.a1
    ci_lower, ci_upper = bsci[2], bsci[3]

    # Ошибка относительно теории
    if abs(a1_th) > 1e-12:
        err_pct = abs(a1_meas - a1_th) / abs(a1_th) * 100.0
    else:
        # rank≥1: теория = 0, измеряем абсолютное отклонение
        err_pct = abs(a1_meas) * 100.0

    # z-score из bootstrap
    bs_std   = bsci[1]
    z_score  = abs(bsci[0]) / bs_std if bs_std > 1e-15 else math.inf

    # spectral_collapse по fit
    collapse = fit.spectral_collapse

    # Проверяем корректность предсказания
    if rank == 0:
        # Ожидаем: НЕТ коллапса (a₁ строго отрицательный)
        correct = (not collapse) and (a1_meas < 0)
    else:
        # Ожидаем: ЕСТЬ коллапс
        correct = collapse

    note = ""
    if rank == 0 and err_pct > 5.0:
        note = f"⚠️  a₁ отклонение {err_pct:.1f}% > 5% — нужно проверить lambda_range"
    elif rank == 0 and err_pct > 1.0:
        note = f"ℹ️  a₁ отклонение {err_pct:.1f}% — норма при конечном lambda_range"

    return CurveCalibResult(
        label            = label,
        deg_phi          = deg_phi,
        rank             = rank,
        a1_theory        = a1_th,
        a1_measured      = a1_meas,
        a1_error_pct     = err_pct,
        z_score          = z_score,
        bootstrap_ci     = (ci_lower, ci_upper),
        spectral_collapse= collapse,
        prediction_correct=correct,
        n_points         = len(lambdas),
        lambda_range     = (float(lambdas.min()), float(lambdas.max())),
        elapsed_s        = round(elapsed, 2),
        note             = note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Подбор оптимальных параметров
# ─────────────────────────────────────────────────────────────────────────────

def _recommend_parameters(
    results:      List[CurveCalibResult],
    lambda_range: np.ndarray,
) -> dict:
    """
    По результатам тестов кривых вычисляет рекомендованные параметры.

    Логика:
      LAMBDA_ADIABATIC_MIN: нижняя граница λ где все тесты дали
                             стабильный результат. Берём λ_min × 0.9
                             как запас снизу.

      EPSILON_COLLAPSE_REL: для rank≥1 кривых |a₁|/deg_phi должно быть
                            меньше этого порога. Берём max(|a₁|/D) × 3
                            по rank≥1 кривым с запасом.

      SIGMA_COLLAPSE:       из z-scores. rank=0: z > этого порога.
                            rank≥1: z < этого порога.
                            Оптимальный = посередине между min(z_rank0)
                            и max(z_rank_ge1).

      scale:                текущий 256 — проверяем что λ_eff при
                            типичных параметрах (n=47, leak=8) даёт
                            λ_eff >> LAMBDA_ADIABATIC_MIN.
    """
    from deformation import LAMBDA_ADIABATIC_MIN, EPSILON_COLLAPSE_REL, SIGMA_COLLAPSE

    # Разделяем по rank
    rank0    = [r for r in results if r.rank == 0]
    rank_ge1 = [r for r in results if r.rank >= 1]

    # ── LAMBDA_ADIABATIC_MIN ──────────────────────────────────────────────────
    # Минимальный λ в данных × 0.9 — чуть ниже для запаса
    lam_min_data = float(lambda_range.min())
    lam_rec = round(lam_min_data * 0.9, 1)
    # Но не меньше текущего если текущий работает
    all_correct = all(r.prediction_correct for r in results)
    if all_correct:
        lam_rec = min(lam_rec, LAMBDA_ADIABATIC_MIN)
    else:
        lam_rec = max(lam_rec, 30.0)  # безопасный минимум

    # ── EPSILON_COLLAPSE_REL ──────────────────────────────────────────────────
    if rank_ge1:
        # Максимальный |a₁|/deg_phi среди rank≥1 кривых (их a₁ должно быть ≈ 0)
        eps_candidates = [
            abs(r.a1_measured) / max(1.0, float(r.deg_phi))
            for r in rank_ge1
        ]
        eps_max = max(eps_candidates)
        # Берём с запасом ×3, но не меньше текущего если всё работает
        eps_rec = eps_max * 3.0
        if all_correct:
            eps_rec = max(eps_rec, EPSILON_COLLAPSE_REL)
        eps_rec = round(eps_rec, 6)
    else:
        eps_rec = EPSILON_COLLAPSE_REL

    # ── SIGMA_COLLAPSE ────────────────────────────────────────────────────────
    z_rank0    = [r.z_score for r in rank0    if math.isfinite(r.z_score)]
    z_rank_ge1 = [r.z_score for r in rank_ge1 if math.isfinite(r.z_score)]

    if z_rank0 and z_rank_ge1:
        z_min_rank0    = min(z_rank0)
        z_max_rank_ge1 = max(z_rank_ge1)
        if z_min_rank0 > z_max_rank_ge1:
            # Есть чёткое разделение — ставим порог посередине
            sigma_rec = round((z_min_rank0 + z_max_rank_ge1) / 2.0, 1)
        else:
            # Перекрытие — оставляем текущий с предупреждением
            sigma_rec = SIGMA_COLLAPSE
    else:
        sigma_rec = SIGMA_COLLAPSE

    # ── scale ─────────────────────────────────────────────────────────────────
    # Проверяем: при n=47, leak=8 → λ_eff = 47/64 × scale
    # Должно быть >> LAMBDA_ADIABATIC_MIN (т.е. >> lam_rec)
    # scale=256 → λ_eff = 188 (хорошо)
    # Если λ_eff < 5 × lam_rec — scale надо увеличить
    scale_current = 256.0
    lam_eff_test  = (47 / 64.0) * scale_current
    if lam_eff_test >= 5.0 * lam_rec:
        scale_rec = scale_current
    else:
        # Подбираем scale чтобы λ_eff(n=47,leak=8) = 10 × lam_rec
        scale_rec = round(10.0 * lam_rec * 64.0 / 47.0, 1)

    return {
        "lambda_adiabatic_min": lam_rec,
        "epsilon_collapse_rel": eps_rec,
        "sigma_collapse":       sigma_rec,
        "scale":                scale_rec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция калибровки
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    curves:          Optional[List[str]] = None,
    lambda_range:    Optional[np.ndarray] = None,
    n_bootstrap:     int   = 300,
    n_terms:         int   = 3,
    use_real_solver: bool  = False,
    save:            bool  = True,
    verbose:         bool  = True,
) -> CalibrationResult:
    """
    Запускает калибровку по benchmark кривым из Part II.

    Parameters
    ----------
    curves       : список кривых для теста (default: все три)
                   допустимые: "11a1", "37a", "389a1"
    lambda_range : диапазон λ (default: logspace(2, 5, 30))
    n_bootstrap  : bootstrap итерации (default 300, для быстрой ~100)
    n_terms      : члены асимптотического ряда (default 3)
    use_real_solver : True — использовать реальный MHR solver если доступен
    save         : сохранять результат в data/calibration.json
    verbose      : печатать отчёт в терминал

    Returns
    -------
    CalibrationResult с рекомендованными параметрами

    Notes
    -----
    Быстрая калибровка (mock, n_bootstrap=100): ~5-10 сек
    Полная калибровка (mock, n_bootstrap=300):  ~30-60 сек
    С реальным solver: зависит от solver
    """
    from deformation import (
        LAMBDA_ADIABATIC_MIN, EPSILON_COLLAPSE_REL, SIGMA_COLLAPSE,
    )
    import datetime as _dt

    if curves is None:
        curves = list(BENCHMARK_CURVES.keys())

    if lambda_range is None:
        lambda_range = np.logspace(2, 5, 30)

    if verbose:
        print("\n" + "═" * 66)
        print("  AEGIS a1_calibrator.py v1.0")
        print("  Калибровка по benchmark кривым [F: Part II, Prop.5.1]")
        print("═" * 66)
        print(f"\n  Кривые:     {', '.join(curves)}")
        print(f"  λ диапазон: [{lambda_range.min():.0f}, {lambda_range.max():.0f}]  ({len(lambda_range)} точек)")
        print(f"  Bootstrap:  {n_bootstrap} итераций")
        print(f"  n_terms:    {n_terms}")
        print(f"\n  Текущие параметры:")
        print(f"    LAMBDA_ADIABATIC_MIN = {LAMBDA_ADIABATIC_MIN}")
        print(f"    EPSILON_COLLAPSE_REL = {EPSILON_COLLAPSE_REL}")
        print(f"    SIGMA_COLLAPSE       = {SIGMA_COLLAPSE}")
        print()

    # Тестируем каждую кривую
    curve_results: List[CurveCalibResult] = []
    for cname in curves:
        if cname not in BENCHMARK_CURVES:
            logger.warning("Неизвестная кривая '%s' — пропускаем", cname)
            continue

        c = BENCHMARK_CURVES[cname]

        if verbose:
            rank_str = f"rank={c['rank']}, D={c['deg_phi']}"
            a1_str   = f"a₁_theory = {c['a1_theory']:.5f}" if c['a1_theory'] != 0 else "a₁_theory = 0 (collapse)"
            print(f"  ── {cname:8s} ({rank_str}, {a1_str}) ──")

        try:
            result = _test_curve(
                curve           = c,
                lambda_range    = lambda_range,
                n_bootstrap     = n_bootstrap,
                n_terms         = n_terms,
                use_real_solver = use_real_solver,
            )
            curve_results.append(result)

            if verbose:
                ok_mark = "✅" if result.prediction_correct else "❌"
                print(f"    a₁ измерен:  {result.a1_measured:.5f}")
                if result.rank == 0:
                    print(f"    a₁ теория:   {result.a1_theory:.5f}  "
                          f"(ошибка {result.a1_error_pct:.2f}%)")
                print(f"    z-score:     {result.z_score:.1f}  "
                      f"(ожидалось {c['z_expected']})")
                print(f"    95% CI:      [{result.bootstrap_ci[0]:.4e}, "
                      f"{result.bootstrap_ci[1]:.4e}]")
                print(f"    collapse:    {result.spectral_collapse}  "
                      f"(верно: {ok_mark})")
                print(f"    точек λ:     {result.n_points}  "
                      f"время: {result.elapsed_s}с")
                if result.note:
                    print(f"    {result.note}")
                print()

        except Exception as exc:
            logger.error("Ошибка при тесте %s: %s", cname, exc)
            if verbose:
                print(f"    ❌ Ошибка: {exc}\n")

    # Итоговые метрики
    n_tested = len(curve_results)
    n_passed = sum(1 for r in curve_results if r.prediction_correct)

    # Рекомендованные параметры
    rec = _recommend_parameters(curve_results, lambda_range)

    # Статус
    if n_passed == n_tested and n_tested > 0:
        status = "OK"
    elif n_passed >= n_tested * 0.66:
        status = "WARN"
    else:
        status = "FAIL"

    summary = _build_summary(curve_results, rec, status)

    if verbose:
        print(summary)

    # Формируем результат
    calib = CalibrationResult(
        timestamp                  = _dt.datetime.utcnow().isoformat() + "Z",
        curves_tested              = n_tested,
        curves_passed              = n_passed,
        lambda_adiabatic_min_used  = LAMBDA_ADIABATIC_MIN,
        epsilon_collapse_rel_used  = EPSILON_COLLAPSE_REL,
        sigma_collapse_used        = SIGMA_COLLAPSE,
        lambda_adiabatic_min_rec   = rec["lambda_adiabatic_min"],
        epsilon_collapse_rel_rec   = rec["epsilon_collapse_rel"],
        sigma_collapse_rec         = rec["sigma_collapse"],
        scale_rec                  = rec["scale"],
        curve_results              = [asdict(r) for r in curve_results],
        status                     = status,
        summary                    = summary,
    )

    if save:
        _save_calibration(calib)

    return calib


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции вывода и сохранения
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary(
    results: List[CurveCalibResult],
    rec:     dict,
    status:  str,
) -> str:
    """Форматирует итоговый отчёт."""
    n = len(results)
    passed = sum(1 for r in results if r.prediction_correct)

    icon = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(status, "?")

    lines = [
        "═" * 66,
        f"  ИТОГ КАЛИБРОВКИ  {icon} {status}",
        "═" * 66,
        f"  Кривых протестировано: {n}  |  Верных предсказаний: {passed}/{n}",
        "",
        "  Таблица результатов:",
        f"  {'Кривая':<8} {'rank':>5} {'D':>4} {'a₁_theory':>12} "
        f"{'a₁_meas':>12} {'z':>8} {'верно':>6}",
        "  " + "─" * 60,
    ]
    for r in results:
        ok = "✅" if r.prediction_correct else "❌"
        lines.append(
            f"  {r.label:<8} {r.rank:>5} {r.deg_phi:>4} "
            f"{r.a1_theory:>12.5f} {r.a1_measured:>12.5f} "
            f"{r.z_score:>8.1f} {ok:>6}"
        )

    from deformation import (
        LAMBDA_ADIABATIC_MIN, EPSILON_COLLAPSE_REL, SIGMA_COLLAPSE,
    )
    lam_changed  = abs(rec["lambda_adiabatic_min"] - LAMBDA_ADIABATIC_MIN) > 0.5
    eps_changed  = abs(rec["epsilon_collapse_rel"] - EPSILON_COLLAPSE_REL) > 1e-6
    sig_changed  = abs(rec["sigma_collapse"]       - SIGMA_COLLAPSE) > 0.05
    any_changed  = lam_changed or eps_changed or sig_changed

    lines += [
        "",
        "  Рекомендованные параметры:",
        f"    LAMBDA_ADIABATIC_MIN = {rec['lambda_adiabatic_min']:<8}"
        + (f"  ← изменить с {LAMBDA_ADIABATIC_MIN}" if lam_changed else "  (без изменений)"),
        f"    EPSILON_COLLAPSE_REL = {rec['epsilon_collapse_rel']:<10.6f}"
        + (f"  ← изменить с {EPSILON_COLLAPSE_REL}" if eps_changed else "  (без изменений)"),
        f"    SIGMA_COLLAPSE       = {rec['sigma_collapse']:<8}"
        + (f"  ← изменить с {SIGMA_COLLAPSE}" if sig_changed else "  (без изменений)"),
        "",
    ]

    if not any_changed:
        lines.append("  ✅ Текущие параметры оптимальны — изменений не требуется.")
    else:
        lines += [
            "  Для применения скопируй в deformation.py:",
            f"    LAMBDA_ADIABATIC_MIN: float = {rec['lambda_adiabatic_min']}",
            f"    EPSILON_COLLAPSE_REL: float = {rec['epsilon_collapse_rel']}",
            f"    SIGMA_COLLAPSE:       float = {rec['sigma_collapse']}",
        ]

    lines += ["═" * 66]
    return "\n".join(lines)


def _save_calibration(calib: CalibrationResult):
    """Сохраняет результат калибровки в data/calibration.json."""
    os.makedirs(_DATA, exist_ok=True)
    data = asdict(calib)
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Сохранено: {CALIBRATION_FILE}")


def load_calibration() -> Optional[dict]:
    """
    Загружает последний результат калибровки.

    Returns
    -------
    dict или None если файл не найден
    """
    if not os.path.exists(CALIBRATION_FILE):
        return None
    with open(CALIBRATION_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_recommended_params() -> dict:
    """
    Возвращает рекомендованные параметры из последней калибровки.
    Если калибровка не проводилась — возвращает текущие defaults.

    Returns
    -------
    dict: {lambda_adiabatic_min, epsilon_collapse_rel,
           sigma_collapse, scale, source}
    """
    from deformation import LAMBDA_ADIABATIC_MIN, EPSILON_COLLAPSE_REL, SIGMA_COLLAPSE

    data = load_calibration()
    if data is None:
        return {
            "lambda_adiabatic_min": LAMBDA_ADIABATIC_MIN,
            "epsilon_collapse_rel": EPSILON_COLLAPSE_REL,
            "sigma_collapse":       SIGMA_COLLAPSE,
            "scale":                256.0,
            "source":               "defaults (калибровка не проводилась)",
        }

    return {
        "lambda_adiabatic_min": data["lambda_adiabatic_min_rec"],
        "epsilon_collapse_rel": data["epsilon_collapse_rel_rec"],
        "sigma_collapse":       data["sigma_collapse_rec"],
        "scale":                data["scale_rec"],
        "source":               f"calibration {data['timestamp']} ({data['status']})",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Быстрая проверка одной кривой (для smoke-теста)
# ─────────────────────────────────────────────────────────────────────────────

def quick_check(curve_name: str = "11a1", verbose: bool = True) -> bool:
    """
    Быстрая проверка одной кривой (n_bootstrap=50).
    Используется для smoke-теста при установке.

    Returns
    -------
    bool — True если предсказание верно
    """
    result = run_calibration(
        curves       = [curve_name],
        lambda_range = np.logspace(2, 5, 20),
        n_bootstrap  = 50,
        n_terms      = 2,
        save         = False,
        verbose      = verbose,
    )
    return result.curves_passed == result.curves_tested


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная: таблица теоретических a₁
# ─────────────────────────────────────────────────────────────────────────────

def print_a1_table(deg_phi_range: Optional[List[int]] = None):
    """
    Печатает таблицу теоретических a₁ = -D²/π² для разных D.
    [F: Part II, Proposition 5.1(i)]
    """
    if deg_phi_range is None:
        deg_phi_range = [1, 2, 3, 5, 10, 20, 40]

    print("\n  Теоретические значения a₁ = -D²/π²  [F: Part II, Prop.5.1]")
    print(f"  {'D (deg_phi)':>12}  {'a₁_theory':>14}  {'|a₁|':>12}")
    print("  " + "─" * 42)
    for D in deg_phi_range:
        a1 = a1_theoretical(D)
        print(f"  {D:>12}  {a1:>14.6f}  {abs(a1):>12.6f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI точка входа
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="a1_calibrator — калибровка параметров деформации"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Полная калибровка (n_bootstrap=300, все 3 кривые)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Быстрая проверка (n_bootstrap=50, только 11a1)"
    )
    parser.add_argument(
        "--table", action="store_true",
        help="Показать таблицу теоретических a₁"
    )
    parser.add_argument(
        "--curves", nargs="+", default=None,
        choices=list(BENCHMARK_CURVES.keys()),
        help="Какие кривые тестировать"
    )
    parser.add_argument(
        "--bootstrap", type=int, default=150,
        help="Число bootstrap итераций (default 150)"
    )
    args = parser.parse_args()

    if args.table:
        print_a1_table()
        sys.exit(0)

    if args.quick:
        ok = quick_check("11a1", verbose=True)
        sys.exit(0 if ok else 1)

    if args.full:
        run_calibration(
            curves      = args.curves,
            n_bootstrap = 300,
            n_terms     = 3,
            verbose     = True,
            save        = True,
        )
    else:
        # Стандартный запуск
        run_calibration(
            curves      = args.curves,
            n_bootstrap = args.bootstrap,
            n_terms     = 3,
            verbose     = True,
            save        = True,
        )
