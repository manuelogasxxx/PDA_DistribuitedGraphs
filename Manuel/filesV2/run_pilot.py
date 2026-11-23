#!/usr/bin/env python3
"""
run_pilot.py -- PILOTO del plan de pruebas (no es el orquestador del paso 6).

Para qué sirve
  1. Obtener efectos y varianza preliminares para dimensionar las repeticiones.
  2. Medir el costo por corrida y estimar el presupuesto de cómputo de la campaña completa.
  3. Comprobar que el pipeline entero (grafo -> señales -> partición -> distribuido ->
     métricas) produce las tablas y figuras que el artículo necesita.

Protocolo que sigue (el mismo que la campaña final)
  * Umbral tau calibrado con el CENTRALIZADO en semillas de calibración (1001..), congelado y
    aplicado igual a todos los métodos. Semillas de evaluación distintas (1..N).
  * Mismas (grafo, señales) para todos los métodos de una semilla => pruebas pareadas.
  * Louvain se calcula una vez por semilla y se reutiliza.

Uso:
  python run_pilot.py --n 10000 --seeds 5 --out ./results/pilot_n10000.csv
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from scipy import stats

from detect import DetectorConfig, run_centralized
from distributed import CostModel, prepare_reference, run_distributed, write_rows_csv
from iot_graph_generator import generate
from metrics import best_f1_threshold
from partition import _gv, louvain_labels
from signals import generate_signals

SCENARIO = dict(anomaly="both", anomaly_rate=0.01, missing="structured", missing_rate=0.05)
CFG = DetectorConfig(hops=1, impute_iters=2)                  # halo requerido = 2
METHODS = ["hash", "spatial", "community", "community_refined"]


def build_case(n, seed, **graph_kw):
    g = generate("hybrid", n, seed=seed, k_neighbors=8, **graph_kw)
    return g, generate_signals(g, seed=seed, **SCENARIO)


def calibrate_tau(n, cal_seeds):
    s_all, y_all = [], []
    for sd in cal_seeds:
        g, sig = build_case(n, sd)
        res = run_centralized(g, sig, CFG)
        s_all.append(res.scores_spatial[sig.mask])
        y_all.append(sig.labels[sig.mask])
    return best_f1_threshold(np.concatenate(s_all).astype(np.float64), np.concatenate(y_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--spread", type=float, default=0.2, help="spread_factor del grafo hybrid")
    ap.add_argument("--out", default="./results/pilot.csv")
    a = ap.parse_args()

    t0 = time.perf_counter()
    tau, f1_cal = calibrate_tau(a.n, range(1001, 1004))
    print(f"tau calibrado = {tau:.3f} (F1 de calibración {f1_cal:.3f}) en {time.perf_counter() - t0:.1f}s")

    rows, t_run = [], []
    for seed in range(1, a.seeds + 1):
        g, sig = build_case(a.n, seed, spread_factor=a.spread)
        ref = prepare_reference(g, sig, CFG, tau=tau, repeats=3)
        labels = louvain_labels(_gv(g), seed)
        for method, k, depth in itertools.product(METHODS, a.k, a.depths):
            opts = dict(labels=labels) if "community" in method else {}
            t1 = time.perf_counter()
            rows.append(run_distributed(g, sig, ref, method, k, depth, CFG, seed, CostModel(),
                                        partition_opts=opts, repeats=1))
            t_run.append(time.perf_counter() - t1)
        print(f"  semilla {seed}/{a.seeds}: {len(rows)} filas, {time.perf_counter() - t0:.0f}s acumulados")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, a.out)
    print(f"\n{len(rows)} corridas | costo medio por corrida {np.mean(t_run):.2f}s "
          f"(p95 {np.percentile(t_run, 95):.2f}s) | CSV: {a.out}\n")

    # ---- resumen por (método, k, profundidad) ----
    def col(m, k, d, key):
        return np.array([r[key] for r in rows if (r["method"], r["k"], r["depth"]) == (m, k, d)], float)

    print(f"{'método':18s} {'k':>2s} {'d':>2s} | {'ΔF1 (pp)':>14s} {'ΔAUC':>8s} {'ΔMAE':>8s} | "
          f"{'MB halo':>8s} {'replic':>7s} {'speedup*':>9s} {'sin recol.*':>11s}")
    for m, k, d in itertools.product(METHODS, a.k, a.depths):
        f = col(m, k, d, "d_f1") * 100
        print(f"{m:18s} {k:2d} {d:2d} | {f.mean():+7.2f}±{f.std(ddof=1) if len(f) > 1 else 0:5.2f} "
              f"{col(m, k, d, 'd_auc').mean():+8.3f} {col(m, k, d, 'd_mae').mean():+8.3f} | "
              f"{col(m, k, d, 'halo_bytes').mean() / 1e6:8.2f} {col(m, k, d, 'replication_factor').mean():7.2f} "
              f"{col(m, k, d, 'speedup_model').mean():9.2f} {col(m, k, d, 'speedup_model_nogather').mean():11.2f}")
    print("* modelado (1 Gbit/s, 0.5 ms), cota ideal.")

    # ---- hipótesis de la propuesta: ¿F1 a <= 2.5 pp del centralizado sin halo? (IC 90 %, TOST) ----
    print("\nF1 distribuido - F1 centralizado, SIN halo (depth=0), IC 90 % t de Student, margen ±2.5 pp:")
    for m in METHODS:
        for k in a.k:
            f = col(m, k, 0, "d_f1") * 100
            if len(f) < 3:
                continue
            half = stats.t.ppf(0.95, len(f) - 1) * f.std(ddof=1) / np.sqrt(len(f))
            ok = (f.mean() - half > -2.5) and (f.mean() + half < 2.5)
            print(f"  {m:18s} k={k}: {f.mean():+6.2f} pp  IC90 [{f.mean() - half:+6.2f}, {f.mean() + half:+6.2f}]"
                  f"  -> {'DENTRO del margen' if ok else 'no concluyente / fuera'}")

    # ---- repeticiones necesarias (prueba pareada, potencia 80 %, alfa 5 %) ----
    print("\nRepeticiones para detectar con potencia 80 % (pareado, alfa 0.05) una diferencia entre métodos:")
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    for k in a.k:
        h, c = col("hash", k, 0, "f1"), col("community", k, 0, "f1")
        d = (c - h) * 100
        if len(d) > 2 and d.std(ddof=1) > 0:
            for delta in (5.0, 2.0, 1.0):
                need = int(np.ceil((z * d.std(ddof=1) / delta) ** 2))
                print(f"  comunidad vs hash, depth=0, k={k}: sd de las diferencias = {d.std(ddof=1):.2f} pp; "
                      f"para detectar {delta:g} pp hacen falta ≥ {max(need, 2)} semillas")


if __name__ == "__main__":
    main()
