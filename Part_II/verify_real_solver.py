"""
verify_real_solver.py
Сравнение real solver vs mock solver vs контрольные числа из Part I.

Контрольные числа [F: Part I, Tab. 2-4] для 37.a1 (D=2, rank=0):
    λ=100  → ΔE = 1.99410,  ε = 2.95e-3
    λ=1000 → ΔE = 1.99955,  ε = 2.25e-4
    λ=5000 → ΔE = 1.99990,  ε = 4.24e-5
"""

import sys
import os
import numpy as np

# Настройка путей для запуска из корня репо
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорт из папки Part_II
from Part_II.mhr_real_solver import mhr_real_solver, energy_functional

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательный mock (для сравнения)
# ─────────────────────────────────────────────────────────────────────────────

def mock_solver(lmbda, params):
    deg_phi = params.get("deg_phi", 2)
    rank    = params.get("rank", 0)
    a1_true = 0.0 if rank >= 1 else float(deg_phi)
    np.random.seed(0)  # фиксированный seed для воспроизводимости
    delta_E = (deg_phi
               + a1_true / lmbda
               + 0.5 / lmbda ** 2
               + np.random.normal(0, 1e-4))
    return delta_E, True, 42, 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Контрольные числа из Part I (37.a1, D=2, rank=0)
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_37A1 = {
    100.0:  1.99410,
    1000.0: 1.99955,
    5000.0: 1.99990,
}

# ─────────────────────────────────────────────────────────────────────────────
# Тест 1: Точечное сравнение для 37.a1
# ─────────────────────────────────────────────────────────────────────────────

def test_37a1_point():
    print("=" * 80)
    print("ТЕСТ 1: Точечное сравнение 37.a1 (D=2, rank=0)")
    print("Контрольные числа из Part I [F: Tab. 2]")
    print("=" * 80)

    params = {"deg_phi": 2, "rank": 0, "label": "37a1"}

    # Заголовок таблицы
    header = f"{'λ':>8} | {'ΔE_real':>10} | {'ΔE_mock':>10} | {'ΔE_partI':>10} | {'err_real':>10} | {'err_mock':>10} | {'status':>6}"
    print(header)
    print("-" * len(header))

    all_pass = True
    for lmbda, dE_ref in sorted(CONTROL_37A1.items()):
        # Вызов реального решателя
        dE_real, ok, iters, res = mhr_real_solver(lmbda, params)
        dE_mock, _, _, _        = mock_solver(lmbda, params)

        err_real = abs(dE_real - dE_ref) / dE_ref
        err_mock = abs(dE_mock - dE_ref) / dE_ref
        ok_real  = err_real < 0.001   # порог 0.1%

        status = "PASS" if ok_real else "FAIL"
        if not ok_real:
            all_pass = False

        print(f"{lmbda:>8.0f} | {dE_real:>10.6f} | {dE_mock:>10.6f} | {dE_ref:>10.6f} | {err_real:>10.2e} | {err_mock:>10.2e} | {status:>6}")

    print()
    print(f"Минимальный критерий (<0.1% для λ∈{{100,1000,5000}}): {'✅ PASS' if all_pass else '❌ FAIL'}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Тест 2: Sweep λ от 100 до 10000 для 37.a1
# ─────────────────────────────────────────────────────────────────────────────

def test_37a1_sweep():
    print()
    print("=" * 80)
    print("ТЕСТ 2: Sweep λ ∈ [100, 10000] для 37.a1 (D=2, rank=0)")
    print("=" * 80)

    params = {"deg_phi": 2, "rank": 0, "label": "37a1"}
    lambdas = np.array([100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)

    dE_reals = []
    print(f"{'λ':>8} | {'ΔE_real':>10} | {'converged':>9} | {'iters':>5} | {'residual':>10} | {'|ΔE-2|/2':>10}")
    print("-" * 75)

    for lmbda in lambdas:
        dE, ok, iters, res = mhr_real_solver(lmbda, params)
        err = abs(dE - 2.0) / 2.0
        dE_reals.append(dE)
        print(f"{lmbda:>8.0f} | {dE:>10.6f} | {str(ok):>9} | {iters:>5} | {res:>10.2e} | {err:>10.2e}")

    # Монотонность: ΔE должен расти к 2.0
    diffs = np.diff(dE_reals)
    mono = all(d >= -1e-6 for d in diffs)  # допуск на численный шум
    print()
    print(f"Монотонность ΔE→2.0: {'✅ YES' if mono else '❌ NO'}")
    return dE_reals


# ─────────────────────────────────────────────────────────────────────────────
# Тест 3: rank=1 спектральный коллапс (37.b1)
# ─────────────────────────────────────────────────────────────────────────────

def test_37b1_collapse():
    print()
    print("=" * 80)
    print("ТЕСТ 3: Спектральный коллапс 37.b1 (D=2, rank=1)")
    print("Ожидаем: ΔE → 0 при λ→∞ (Bloch band width)")
    print("=" * 80)

    params = {"deg_phi": 2, "rank": 1, "label": "37b1"}
    lambdas = np.array([100, 500, 1000, 2000, 5000, 10000], dtype=float)

    print(f"{'λ':>8} | {'ΔE_bloch':>12} | {'collapse?':>10}")
    print("-" * 40)

    collapse_detected = True
    for lmbda in lambdas:
        dE, ok, iters, res = mhr_real_solver(lmbda, params)
        # Коллапс детектируется если ΔE значительно < D=2
        is_collapse = dE < 0.5  # достаточно малое значение
        if not is_collapse:
            collapse_detected = False
        print(f"{lmbda:>8.0f} | {dE:>12.6e} | {'YES' if is_collapse else 'NO':>10}")

    print()
    print(f"Спектральный коллапс обнаружен: {'✅ YES' if collapse_detected else '❌ NO'}")
    return collapse_detected


# ─────────────────────────────────────────────────────────────────────────────
# Тест 4: Дополнительные кривые (11.a1, 389.a1)
# ─────────────────────────────────────────────────────────────────────────────

def test_additional_curves():
    print()
    print("=" * 80)
    print("ТЕСТ 4: Дополнительные кривые")
    print("=" * 80)

    curves = [
        {"label": "11.a1",  "deg_phi": 1, "rank": 0, "expected_lim": 1.0},
        {"label": "389.a1", "deg_phi": 40, "rank": 2, "expected_lim": None},  # rank=2 → коллапс
    ]

    for c in curves:
        label = c["label"]
        print(f"\n── {label} (D={c['deg_phi']}, rank={c['rank']}) ──")
        params = {k: c[k] for k in ("deg_phi", "rank", "label")}

        for lmbda in [100.0, 1000.0, 5000.0]:
            dE, ok, iters, res = mhr_real_solver(lmbda, params)
            if c["rank"] == 0:
                D   = c["deg_phi"]
                err = abs(dE - D) / D
                print(f"  λ={lmbda:>7.0f}: ΔE={dE:.6f}  err={err:.2e}  iters={iters}")
            else:
                print(f"  λ={lmbda:>7.0f}: ΔE={dE:.4e}  (collapse proxy)  iters={iters}")


# ─────────────────────────────────────────────────────────────────────────────
# Тест 5: Smoke test (из промта)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    print()
    print("=" * 80)
    print("SMOKE TEST (из задания)")
    print("=" * 80)

    # Тест 1: 37.a1, λ=1000
    delta_E, ok, iters, res = mhr_real_solver(1000.0, {"deg_phi": 2, "rank": 0})
    assert ok, "Solver не сошёлся"
    assert abs(delta_E - 2.0) < 0.01, f"ΔE={delta_E:.5f}, ожидали ~2.0"
    print(f"[PASS] 37.a1, λ=1000: ΔE={delta_E:.6f}, iters={iters}")

    # Тест 2: rank=1 proxy
    delta_E_r1, ok2, _, _ = mhr_real_solver(1000.0, {"deg_phi": 2, "rank": 1})
    print(f"[INFO] rank=1 proxy, λ=1000: ΔE={delta_E_r1:.6e}")
    assert delta_E_r1 < 1.0, f"rank=1 должен давать ΔE << 1, получили {delta_E_r1:.4f}"
    print("[PASS] rank=1 ΔE << 1 (спектральный коллапс подтверждён)")

    print()
    print("[ALL SMOKE TESTS PASSED] — ready for deformation.py integration")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Запуск всех тестов
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║          verify_real_solver.py  |  mhr-numerics Part II                ║")
    print("║          Real MHR Solver vs Mock vs Part I контрольные числа            ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    passed_1 = test_37a1_point()
    dE_sweep = test_37a1_sweep()
    passed_3 = test_37b1_collapse()
    test_additional_curves()
    smoke_test()

    print()
    print("═" * 80)
    print("ИТОГ:")
    print(f"  Тест 1 (37.a1 точечное сравнение <0.1%): {'PASS ✅' if passed_1 else 'FAIL ❌'}")
    print(f"  Тест 3 (37.b1 спектральный коллапс):      {'PASS ✅' if passed_3 else 'FAIL ❌'}")
    print(f"  Smoke test:                                PASS ✅")
    print("═" * 80)
