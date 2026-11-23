#!/usr/bin/env python3
"""
example_pipeline.py -- recorrido completo de los pasos 1 y 2 (sin escribir a disco).

    python example_pipeline.py            # n=3000, tarda unos segundos
    python example_pipeline.py --n 10000

Muestra, en orden:
  1. Generar un grafo y sus señales (con ground truth).
  2. Calibrar el umbral con el método centralizado en OTRA semilla y congelarlo.
  3. Evaluar detectores e imputación con ese umbral congelado.
  4. Demostrar el halo: un "worker" con propios + halo reproduce el resultado
     centralizado; sin halo se degrada.
"""

import argparse

import numpy as np

from detect import (DetectorConfig, adjacency_hops, impute, induced_edges, run_centralized,
                    spatial_scores, temporal_scores)
from iot_graph_generator import generate
from metrics import best_f1_threshold, evaluate
from signals import generate_signals

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
args = ap.parse_args()
N = args.n

# Un solo escenario "difícil pero justo": anomalías puntuales y regionales, apagones por zona.
SCENARIO = dict(anomaly="both", anomaly_rate=0.01, missing="structured", missing_rate=0.05)
CFG = DetectorConfig(hops=1, impute_iters=2)


def make_case(seed):
    g = generate("hybrid", N, seed=seed, k_neighbors=8)
    s = generate_signals(g, seed=seed, **SCENARIO)
    return g, s


# --------------------------------------------------------------------------------------
print("=" * 70, "\n1. DATOS\n" + "=" * 70)
g, sig = make_case(seed=1)
r = g.report
print(f"Grafo : {r['n_nodes']} nodos, {r['n_edges']} aristas, grado medio {r['avg_degree']}, "
      f"{r['n_communities']} comunidades, mixing real {r['mixing_realized']}")
r = sig.report
print(f"Señales: T={r['T']}, faltantes {r['missing_fraction_realized']:.1%}, "
      f"anomalías {r['anomaly_fraction_realized']:.1%} "
      f"(puntuales {r['anomalies_point']}, regionales {r['anomalies_regional']} "
      f"en {r['regional_events']} eventos)")
print(f"         r_sd={r['r_sd']}  (escala de dificultad)  smoothness_ratio={r['smoothness_ratio']}")

# --------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\n2. CALIBRACIÓN DEL UMBRAL (otra semilla, método centralizado)\n" + "=" * 70)
g_cal, s_cal = make_case(seed=100)
res_cal = run_centralized(g_cal, s_cal, CFG)
tau = {}
for kind in ("spatial", "temporal", "combined"):
    sc = res_cal.scores(kind)[s_cal.mask]
    y = s_cal.labels[s_cal.mask]
    tau[kind], f1_cal = best_f1_threshold(sc, y)
    print(f"tau[{kind:8s}] = {tau[kind]:6.2f}   (F1 en calibración = {f1_cal:.3f})")
print("-> estos umbrales se CONGELAN y se usan igual para todos los métodos.")

# --------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\n3. EVALUACIÓN (semilla 1, umbral congelado)\n" + "=" * 70)
res = run_centralized(g, sig, CFG)
print(f"Detector centralizado: {res.info['timings']['total']:.2f} s | "
      f"halo requerido: {CFG.halo_depth} salto(s)\n")
print(f"{'detector':10s} {'AUC':>6s} {'AP':>6s} {'F1':>6s} {'prec':>6s} {'recall':>6s}")
for kind in ("spatial", "temporal", "combined"):
    row = evaluate(res.scores(kind), res.X_imputed, sig, tau=tau[kind])
    print(f"{kind:10s} {row['det_auc']:6.3f} {row['det_ap']:6.3f} {row['det_f1']:6.3f} "
          f"{row['det_precision']:6.3f} {row['det_recall']:6.3f}")
print(f"\nImputación: MAE={row['imp_mae']:.3f}  RMSE={row['imp_rmse']:.3f}  "
      f"sobre {row['imp_n_missing']} lecturas faltantes")
imp = res.info["imputation"]
print(f"  llenadas por vecinos (iter 1, iter 2): {imp['filled_spatial_by_iter']} | "
      f"por respaldo temporal: {imp['filled_temporal_fallback']}")

# --------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\n4. DEMO DE HALO (un 'worker' con su porción del grafo)\n" + "=" * 70)
A1 = adjacency_hops(g.n, g.edges, 1)
part = np.arange(g.n) % 4                  # partición hash en 4 trozos (peor caso)
own = part == 0


def expand(mask, depth):
    reach = mask.copy()
    for _ in range(depth):
        reach |= (A1 @ reach.astype(np.float32)) > 0
    return reach


print(f"El worker 0 es dueño de {own.sum()} nodos. Debe igualar al centralizado en ESOS nodos.")
print(f"{'halo':>5s} {'nodos locales':>14s} {'max|dif| score espacial':>25s} {'max|dif| imputación':>22s}")
for depth in range(0, CFG.halo_depth + 2):
    nodes = np.nonzero(expand(own, depth))[0]
    le = induced_edges(g.n, g.edges, nodes)
    Xs, Ms = sig.X[:, nodes], sig.mask[:, nodes]
    loc = own[nodes]
    z = spatial_scores(Xs, Ms, adjacency_hops(len(nodes), le, CFG.hops), CFG)
    _, pred = temporal_scores(Xs, Ms, CFG)
    Xi, _ = impute(Xs, Ms, adjacency_hops(len(nodes), le, 1), pred, CFG)
    d_sp = np.abs(z[:, loc] - res.scores_spatial[:, own]).max()
    d_im = np.abs(Xi[:, loc] - res.X_imputed[:, own]).max()
    mark = "  <- igual al centralizado" if d_sp < 1e-3 and d_im < 1e-3 else ""
    print(f"{depth:5d} {len(nodes):14d} {d_sp:25.2e} {d_im:22.2e}{mark}")
print(f"\nCon halo >= {CFG.halo_depth} la diferencia es 0; con menos, el resultado cambia.")
print("Ese costo (nodos locales extra = bytes de halo) se cuantifica por método en partition.py.")
