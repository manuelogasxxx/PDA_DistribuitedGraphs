#!/usr/bin/env python3
"""
run_e4b.py -- PILOTO de E4b: prueba adversa (anomalías en la frontera de la partición).

Pregunta (H8): si las anomalías caen justo donde la partición corta, ¿el halo sigue siendo
prescindible? El piloto base (run_pilot.py) lo era con anomalías uniformes.

Diseño
  * Partición de REFERENCIA (por defecto: community, k=8). Sus nodos de frontera definen dónde
    caen las anomalías del escenario "boundary" (puntuales y centros de eventos regionales).
  * TODOS los métodos se evalúan sobre esos mismos datos => comparación pareada. La partición
    de referencia sale perjudicada a propósito (es su peor caso); las demás, solo en parte.
  * Escenario "uniform" como control, con las mismas semillas (X_clean y máscara idénticas).
  * La tasa de anomalías se aplica a las lecturas ELEGIBLES (las de los nodos de frontera),
    así que el escenario "boundary" usa --rate más alto (por defecto 5 %) para tener eventos
    suficientes; el número total de positivos es menor que en "uniform".
  * Umbral tau calibrado POR ESCENARIO con el centralizado en semillas aparte (1001..1003).

Uso:
  python run_e4b.py --n 10000 --seeds 4 --k 8 --out ./results/e4b_n10000.csv
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
from partition import _gv, boundary_nodes, louvain_labels, partition_graph

CFG = DetectorConfig(hops=1, impute_iters=2)                  # halo requerido = 2
METHODS = ["hash", "spatial", "community", "community_refined"]


def case(n, seed, scenario, ref_method, k, rate, spread, mcf=0.0):
    """Grafo + señales del escenario. Devuelve (g, sig, labels_louvain)."""
    g = generate("hybrid", n, seed=seed, k_neighbors=8, spread_factor=spread)
    labels = louvain_labels(_gv(g), seed)
    kw = dict(anomaly="both", missing="structured", missing_rate=0.05)
    if scenario == "uniform":
        sig = generate_signals_(g, seed, anomaly_rate=0.01, **kw)
    else:
        ref_part, _ = partition_graph(g, ref_method, k, seed=seed,
                                      **(dict(labels=labels) if "community" in ref_method else {}))
        sig = generate_signals_(g, seed, anomaly_rate=rate, anomaly_nodes=boundary_nodes(g, ref_part, mcf),
                                anomaly_placement=f"boundary_{ref_method}_k{k}_mcf{mcf:g}", **kw)
    return g, sig, labels


def generate_signals_(g, seed, **kw):
    from signals import generate_signals
    return generate_signals(g, seed=seed, **kw)


def calibrate(n, scenario, ref_method, k, rate, spread, mcf=0.0, seeds=range(1001, 1004)):
    s_all, y_all = [], []
    for sd in seeds:
        g, sig, _ = case(n, sd, scenario, ref_method, k, rate, spread, mcf)
        res = run_centralized(g, sig, CFG)
        s_all.append(res.scores_spatial[sig.mask])
        y_all.append(sig.labels[sig.mask])
    return best_f1_threshold(np.concatenate(s_all).astype(np.float64), np.concatenate(y_all))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rate", type=float, default=0.05, help="tasa de anomalías sobre lecturas elegibles (boundary)")
    ap.add_argument("--ref-method", default="community", choices=["community", "spatial", "hash"])
    ap.add_argument("--spread", type=float, default=0.2)
    ap.add_argument("--min-cut-fraction", type=float, default=0.0,
                    help="solo nodos que pierden >= esta fracción de vecinos al cortar (0 = toda la frontera; 0.5 = adversario fuerte)")
    ap.add_argument("--out", default="./results/e4b.csv")
    a = ap.parse_args()

    t0 = time.perf_counter()
    tau = {sc: calibrate(a.n, sc, a.ref_method, a.k, a.rate, a.spread, a.min_cut_fraction) for sc in ("uniform", "boundary")}
    print(f"tau calibrado por escenario: uniform={tau['uniform']:.3f}  boundary={tau['boundary']:.3f} "
          f"({time.perf_counter() - t0:.0f}s)")

    rows = []
    for seed in range(1, a.seeds + 1):
        for sc in ("uniform", "boundary"):
            g, sig, labels = case(a.n, seed, sc, a.ref_method, a.k, a.rate, a.spread, a.min_cut_fraction)
            ref = prepare_reference(g, sig, CFG, tau=tau[sc], repeats=2)
            n_pos = int(sig.labels.sum())
            for method, depth in itertools.product(METHODS, a.depths):
                opts = dict(labels=labels) if "community" in method else {}
                row, _, plan = run_distributed(g, sig, ref, method, a.k, depth, CFG, seed, CostModel(),
                                               partition_opts=opts, return_sim=True)
                bnd = boundary_nodes(g, plan.part)
                row.update(scenario=sc, ref_method=a.ref_method, min_cut_fraction=a.min_cut_fraction, n_positives=n_pos,
                           frac_anom_on_boundary=float(sig.labels[:, bnd].sum()) / max(n_pos, 1))
                rows.append(row)
        print(f"  semilla {seed}/{a.seeds}: {len(rows)} filas, {time.perf_counter() - t0:.0f}s acumulados", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, a.out)

    def col(sc, m, d, key):
        return np.array([r[key] for r in rows if (r["scenario"], r["method"], r["depth"]) == (sc, m, d)], float)

    print(f"\n{len(rows)} corridas, CSV: {a.out}. Referencia de frontera: {a.ref_method} k={a.k}\n")
    print(f"{'escenario':9s} {'método':18s} {'d':>2s} | {'ΔF1 (pp)':>15s} {'ΔAUC':>8s} {'ΔAUC front.':>11s} | "
          f"{'MB halo':>8s} {'%anom en front.':>15s} {'positivos':>9s}")
    for sc, m, d in itertools.product(("uniform", "boundary"), METHODS, a.depths):
        f = col(sc, m, d, "d_f1") * 100
        sd = f.std(ddof=1) if len(f) > 1 else 0.0
        print(f"{sc:9s} {m:18s} {d:2d} | {f.mean():+8.2f} ±{sd:5.2f} {col(sc, m, d, 'd_auc').mean():+8.3f} "
              f"{col(sc, m, d, 'd_auc_boundary').mean():+11.3f} | {col(sc, m, d, 'halo_bytes').mean() / 1e6:8.2f} "
              f"{100 * col(sc, m, d, 'frac_anom_on_boundary').mean():14.1f}% {col(sc, m, d, 'n_positives').mean():9.0f}")

    print("\nContraste de equivalencia SIN halo (profundidad 0), IC 90 %, margen ±2.5 pp:")
    for sc, m in itertools.product(("uniform", "boundary"), METHODS):
        f = col(sc, m, 0, "d_f1") * 100
        if len(f) < 3:
            continue
        half = stats.t.ppf(0.95, len(f) - 1) * f.std(ddof=1) / np.sqrt(len(f))
        lo, hi = f.mean() - half, f.mean() + half
        verdict = "DENTRO del margen" if (lo > -2.5 and hi < 2.5) else "FUERA / no concluyente"
        print(f"  {sc:9s} {m:18s}: {f.mean():+7.2f} pp  IC90 [{lo:+7.2f}, {hi:+7.2f}]  -> {verdict}")


if __name__ == "__main__":
    main()
