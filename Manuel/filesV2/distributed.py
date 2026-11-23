#!/usr/bin/env python3
"""
distributed.py  (v1.0.0)  --  Paso 4: ejecución distribuida SIMULADA en un solo proceso.

Qué hace
  Ejecuta k "workers" uno tras otro dentro de un proceso, pero con las mismas reglas
  que tendrá el sistema real (paso 5):
    * Cada worker solo conoce SUS lecturas (los datos "nacen" en su dueño).
    * El halo llega ÚNICAMENTE por mensajes: el dueño extrae los valores de sus propios
      datos y los envía por una red simulada que cuenta bytes y mensajes por enlace.
    * Cada worker arma su vista local (propios + halo recibido), corre detect.py sobre su
      subgrafo y devuelve solo los resultados de sus nodos propios.
    * El agregador ensambla los resultados globales.
  Así se verifica la equivalencia global con el método centralizado, y se mide qué pasa
  cuando el halo es insuficiente (profundidad < requerida).

Qué es MEDIDO y qué es MODELADO (no mezclar en el artículo)
  MEDIDO     bytes y mensajes (se cuentan sobre los arreglos realmente enviados), tiempo de
             cómputo de cada worker (cronometrado, uno a la vez), métricas de calidad.
  MODELADO   tiempo de comunicación = bytes/ancho_de_banda + mensajes*latencia (CostModel) y
             el tiempo distribuido = intercambio + max(cómputo por worker) + recolección.
             La recolección es un término CONSTANTE (~ n*T*4*3 bytes, igual para toda
             partición); por eso se reporta también speedup_model_nogather.
             Supone k núcleos libres y sin contención. Es una COTA IDEAL: no incluye
             serialización real, planificación ni sobrecarga de Ray. El speedup real se
             medirá en el paso 5 y servirá para calibrar este modelo.

Uso:
  python distributed.py --graph ./data/hybrid_n10000_seed1 \
      --signals ./data/hybrid_n10000_seed1/signals_seed1_both_structured.npz \
      --methods hash spatial community --k 4 8 --depth 0 1 2 --hops 1 --impute-iters 2

Como librería:
  ref = prepare_reference(g, sig, cfg)                       # centralizado, una vez por caso
  row = run_distributed(g, sig, ref, "community", k=8, depth=2, cfg=cfg, seed=1)
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field

import numpy as np

from detect import (DetectorConfig, adjacency_hops, impute, run_centralized, spatial_scores,
                    temporal_scores)
from iot_graph_generator import load_graph
from metrics import best_f1_threshold, detection_report, evaluate, imputation_report
from partition import (BYTES_MASK, BYTES_VALUE, PartitionPlan, _gv, build_plan, partition_graph,
                       plan_metrics, verify_plan)
from signals import load_signals

DIST_VERSION = "1.0.0"
RESULT_ARRAYS = 3            # scores espacial, temporal e imputación que cada worker devuelve


# ---------------------------------------------------------------------------
# Red simulada y modelo de costos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """Parámetros de red del MODELO de tiempo (no medidos)."""
    bandwidth_Bps: float = 125e6      # 1 Gbit/s
    latency_s: float = 5e-4           # 0.5 ms por mensaje


class SimNetwork:
    """Entrega copias de los arreglos y cuenta bytes y mensajes por enlace (src -> dst)."""

    def __init__(self, k: int):
        self.bytes = np.zeros((k, k), dtype=np.int64)
        self.msgs = np.zeros((k, k), dtype=np.int64)

    def send(self, src: int, dst: int, payload: tuple) -> tuple:
        self.bytes[src, dst] += sum(a.nbytes for a in payload)
        self.msgs[src, dst] += 1
        return tuple(a.copy() for a in payload)


# ---------------------------------------------------------------------------
# Un worker
# ---------------------------------------------------------------------------

def _worker_compute(Xl, Ml, w, cfg: DetectorConfig, owned_only: bool, repeats: int):
    """Cómputo local de un worker. Devuelve (z_sp, z_tm, X_imp) de sus nodos PROPIOS y
    tiempos por fase (mínimo de `repeats` ejecuciones, para reducir ruido)."""
    own = np.arange(w.n_owned)
    best = None
    for _ in range(max(1, repeats)):
        t = {}
        t0 = time.perf_counter()
        A_h = adjacency_hops(w.n_local, w.edges, cfg.hops)
        A_1 = A_h if cfg.hops == 1 else adjacency_hops(w.n_local, w.edges, 1)
        t["adjacency"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if owned_only:
            z_sp = spatial_scores(Xl, Ml, A_h, cfg, rows=own)
        else:
            z_sp = spatial_scores(Xl, Ml, A_h, cfg)[:, :w.n_owned]
        t["spatial"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if owned_only:     # el detector temporal no usa vecinos: halo innecesario
            z_tm, pred_own = temporal_scores(Xl[:, :w.n_owned], Ml[:, :w.n_owned], cfg)
            pred = np.zeros(Xl.shape, dtype=np.float32)
            pred[:, :w.n_owned] = pred_own
        else:
            z_tm, pred = temporal_scores(Xl, Ml, cfg)
            z_tm = z_tm[:, :w.n_owned]
        t["temporal"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_imp, _ = impute(Xl, Ml, A_1, pred, cfg)
        t["impute"] = time.perf_counter() - t0
        t["total"] = sum(t.values())
        if best is None or t["total"] < best[3]["total"]:
            best = (z_sp, z_tm, X_imp[:, :w.n_owned], t)
    return best


# ---------------------------------------------------------------------------
# Simulación completa
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    scores_spatial: np.ndarray
    scores_temporal: np.ndarray
    scores_combined: np.ndarray
    X_imputed: np.ndarray
    info: dict = field(default_factory=dict)

    def scores(self, kind: str = "spatial") -> np.ndarray:
        return {"spatial": self.scores_spatial, "temporal": self.scores_temporal,
                "combined": self.scores_combined}[kind]


def simulate(sig, plan: PartitionPlan, cfg: DetectorConfig = DetectorConfig(),
             cost: CostModel = CostModel(), owned_only: bool = True, repeats: int = 1) -> SimResult:
    """Ejecuta el plan: reparto de datos propios, intercambio de halo por mensajes,
    cómputo local y ensamblado. `sig`: SignalData o dict de load_signals."""
    X = np.asarray(sig["X"] if isinstance(sig, dict) else sig.X, dtype=np.float32)
    M = np.asarray(sig["mask"] if isinstance(sig, dict) else sig.mask, dtype=bool)
    T, n = X.shape
    k, W = plan.k, plan.workers
    net = SimNetwork(k)

    # 0) Cada dueño guarda solo sus lecturas.
    own_data = [(X[:, w.owned].copy(), M[:, w.owned].copy()) for w in W]

    # 1) Intercambio de halo: el DUEÑO extrae de sus datos y envía.
    inbox = [dict() for _ in range(k)]
    for dst, w in enumerate(W):
        for src, ids in w.recv_from.items():
            pos = np.searchsorted(W[src].owned, ids)
            Xs, Ms = own_data[src]
            inbox[dst][src] = net.send(src, dst, (Xs[:, pos], Ms[:, pos]))

    # 2) Cómputo local con lo que cada worker tiene (propios + mensajes recibidos).
    out_sp = np.zeros((T, n), dtype=np.float32)
    out_tm = np.zeros((T, n), dtype=np.float32)
    out_im = np.zeros((T, n), dtype=np.float32)
    t_workers, phase = [], []
    for q, w in enumerate(W):
        Xl = np.full((T, w.n_local), np.nan, dtype=np.float32)
        Ml = np.zeros((T, w.n_local), dtype=bool)
        Xl[:, :w.n_owned], Ml[:, :w.n_owned] = own_data[q]
        for src, (Xm, Mm) in inbox[q].items():
            idx = w.n_owned + np.searchsorted(w.halo, w.recv_from[src])
            Xl[:, idx], Ml[:, idx] = Xm, Mm
        z_sp, z_tm, x_im, t = _worker_compute(Xl, Ml, w, cfg, owned_only, repeats)
        out_sp[:, w.owned], out_tm[:, w.owned], out_im[:, w.owned] = z_sp, z_tm, x_im
        t_workers.append(t["total"])
        phase.append(t)

    # 3) Costos: bytes medidos, tiempos modelados.
    recv_bytes = net.bytes.sum(axis=0)
    recv_msgs = net.msgs.sum(axis=0)
    t_comm_w = recv_bytes / cost.bandwidth_Bps + recv_msgs * cost.latency_s
    gather_bytes = int(sum(w.n_owned for w in W) * T * 4 * RESULT_ARRAYS)
    t_gather = gather_bytes / cost.bandwidth_Bps + k * cost.latency_s
    t_exchange = float(t_comm_w.max()) if k > 1 else 0.0
    t_compute_max = float(max(t_workers))
    info = dict(
        k=k, depth=plan.depth, T=T,
        halo_bytes=int(net.bytes.sum()), halo_msgs=int(net.msgs.sum()),
        halo_bytes_max_worker=int(recv_bytes.max()), halo_msgs_max_worker=int(recv_msgs.max()),
        gather_bytes=gather_bytes,
        t_worker=np.asarray(t_workers), t_compute_max=t_compute_max,
        t_compute_sum=float(sum(t_workers)),
        t_exchange_model=t_exchange, t_gather_model=float(t_gather) if k > 1 else 0.0,
        t_dist_model=t_exchange + t_compute_max + (float(t_gather) if k > 1 else 0.0),
        t_dist_model_nogather=t_exchange + t_compute_max,   # resultados se quedan en los workers
        phase_mean={ph: float(np.mean([p[ph] for p in phase]))
                    for ph in ("adjacency", "spatial", "temporal", "impute")},
        link_bytes=net.bytes, link_msgs=net.msgs,
    )
    return SimResult(out_sp, out_tm, np.maximum(out_sp, out_tm), out_im, info)


# ---------------------------------------------------------------------------
# Referencia centralizada y filas de resultados
# ---------------------------------------------------------------------------

@dataclass
class CentralRef:
    result: object          # DetectionResult
    t_central: float
    tau: float
    tau_source: str


def prepare_reference(graph, sig, cfg: DetectorConfig = DetectorConfig(), tau=None,
                      score: str = "spatial", repeats: int = 3) -> CentralRef:
    """Corre el centralizado (una vez por caso). Tiempo = mínimo de `repeats` corridas.
    Si no das `tau`, usa el de mejor F1 sobre ESTE caso, igual para centralizado y
    distribuido (comparación justa, pero optimista: en experimentos formales pasa el tau
    calibrado en semillas aparte)."""
    best = None
    for _ in range(max(1, repeats)):
        res = run_centralized(graph, sig, cfg)
        if best is None or res.info["timings"]["total"] < best.info["timings"]["total"]:
            best = res
    labels = sig["labels"] if isinstance(sig, dict) else sig.labels
    mask = sig["mask"] if isinstance(sig, dict) else sig.mask
    if tau is None:
        tau = best_f1_threshold(best.scores(score)[mask].astype(np.float64), labels[mask])[0]
        src = "best_f1_on_case"
    else:
        src = "given"
    return CentralRef(best, best.info["timings"]["total"], float(tau), src)


def _meta(graph, sig) -> dict:
    gp = graph["manifest"]["params"] if isinstance(graph, dict) else graph.params
    sp = sig["manifest"]["params"] if isinstance(sig, dict) else sig.params
    return dict(mode=gp.get("mode"), n=gp.get("n"), graph_seed=gp.get("seed"),
                spread_factor=gp.get("spread_factor"), signals_seed=sp.get("seed"),
                anomaly=sp.get("anomaly"), missing=sp.get("missing"),
                anomaly_rate=sp.get("anomaly_rate"), missing_rate=sp.get("missing_rate"),
                anomaly_placement=sp.get("anomaly_placement", "uniform"),
                magnitude_lo=sp.get("magnitude", (None, None))[0],
                magnitude_hi=sp.get("magnitude", (None, None))[1])


def run_distributed(graph, sig, ref: CentralRef, method, k: int = 4, depth: int | None = None,
                    cfg: DetectorConfig = DetectorConfig(), seed: int = 0,
                    cost: CostModel = CostModel(), score: str = "spatial",
                    owned_only: bool = True, repeats: int = 1, partition_opts: dict | None = None,
                    verify: bool = False, return_sim: bool = False, executor=None):
    """Una corrida distribuida simulada -> una fila plana (dict) lista para analizar.
    `method` puede ser un nombre de particionador o un PartitionPlan ya construido.
    `executor` (opcional): callable(sig, plan, cfg, cost, owned_only, repeats) -> SimResult que
    sustituye a simulate(); lo usa runtime.py para ejecutar con procesos reales. Su info puede traer
    'row_extra' (columnas extra para la fila).
    depth=None usa la profundidad requerida por cfg; un valor menor mide la degradación."""
    gv = _gv(graph)
    depth = cfg.halo_depth if depth is None else depth
    t0 = time.perf_counter()
    if isinstance(method, PartitionPlan):
        plan, meta_part, name = method, {}, method.method
    else:
        part, meta_part = partition_graph(graph, method, k, seed=seed, **(partition_opts or {}))
        plan, name = build_plan(graph, part, 1 if method == "centralized" else k, depth, method), method
    t_plan = time.perf_counter() - t0
    if verify:
        verify_plan(plan, graph)

    sim = (executor or simulate)(sig, plan, cfg, cost, owned_only, repeats)
    labels = sig["labels"] if isinstance(sig, dict) else sig.labels
    mask = sig["mask"] if isinstance(sig, dict) else sig.mask
    clean = sig["X_clean"] if isinstance(sig, dict) else sig.X_clean
    cres = ref.result

    # --- calidad: global y por tipo de nodo (frontera / interior) ---
    e, part = gv["edges"], plan.part
    cross = part[e[:, 0]] != part[e[:, 1]]
    boundary = np.zeros(gv["n"], dtype=bool)
    boundary[e[cross].ravel()] = True
    interior = ~boundary

    def qual(scores, x_imp, nodes=None):
        d = detection_report(scores, labels, mask, ref.tau, nodes)
        i = imputation_report(x_imp, clean, mask, nodes)
        return d["auc"], d["ap"], d.get("f1", np.nan), i["mae"]

    s_dist, s_cen = sim.scores(score), cres.scores(score)
    auc, ap, f1, mae = qual(s_dist, sim.X_imputed)
    auc0, ap0, f10, mae0 = qual(s_cen, cres.X_imputed)
    ab, _, _, mb = qual(s_dist, sim.X_imputed, boundary)
    ab0, _, _, mb0 = qual(s_cen, cres.X_imputed, boundary)
    ai, _, _, mi = qual(s_dist, sim.X_imputed, interior) if interior.any() else (np.nan,) * 4
    ai0, _, _, mi0 = qual(s_cen, cres.X_imputed, interior) if interior.any() else (np.nan,) * 4

    diff_sp = float(np.abs(sim.scores_spatial - cres.scores_spatial).max())
    diff_im = float(np.abs(sim.X_imputed - cres.X_imputed).max())
    equivalent = bool(np.allclose(sim.scores_spatial, cres.scores_spatial, rtol=1e-3, atol=1e-3)
                      and np.allclose(sim.X_imputed, cres.X_imputed, rtol=1e-4, atol=1e-3))

    pm = plan_metrics(plan, graph, T=sim.info["T"])
    inf = sim.info
    t_dist = inf["t_dist_model"]
    row = dict(
        **_meta(graph, sig), method=name, k=plan.k, depth=plan.depth, required_depth=cfg.halo_depth,
        hops=cfg.hops, impute_iters=cfg.impute_iters, T=inf["T"], score=score, seed=seed,
        # estructura de la partición
        cut_ratio=pm["cut_ratio"], imbalance=pm["imbalance"], boundary_fraction=pm["boundary_fraction"],
        replication_factor=pm["replication_factor"], effective_imbalance=pm["effective_imbalance"],
        comm_links=pm["comm_links"],
        # comunicación (medida)
        halo_bytes=inf["halo_bytes"], halo_msgs=inf["halo_msgs"],
        halo_bytes_max_worker=inf["halo_bytes_max_worker"], halo_msgs_max_worker=inf.get("halo_msgs_max_worker", 0),
        gather_bytes=inf["gather_bytes"],
        topology_bytes=pm["topology_bytes_est"],
        # tiempos: medidos (t_central, t_compute_*) y modelados (t_*_model, speedup, eficiencia)
        t_central=ref.t_central, t_compute_max=inf["t_compute_max"], t_compute_sum=inf["t_compute_sum"],
        t_exchange_model=inf["t_exchange_model"], t_gather_model=inf["t_gather_model"],
        t_dist_model=t_dist, speedup_model=ref.t_central / t_dist if t_dist > 0 else np.nan,
        speedup_model_nogather=(ref.t_central / inf["t_dist_model_nogather"]
                                if inf["t_dist_model_nogather"] > 0 else np.nan),
        efficiency_model=(ref.t_central / t_dist) / plan.k if t_dist > 0 else np.nan,
        work_inflation=inf["t_compute_sum"] / ref.t_central, t_plan=t_plan,
        # calidad: distribuido, central y diferencia (distribuido - central)
        tau=ref.tau, auc=auc, ap=ap, f1=f1, mae=mae,
        auc_central=auc0, ap_central=ap0, f1_central=f10, mae_central=mae0,
        d_auc=auc - auc0, d_ap=ap - ap0, d_f1=f1 - f10, d_mae=mae - mae0,
        d_auc_boundary=ab - ab0, d_auc_interior=ai - ai0 if interior.any() else np.nan,
        d_mae_boundary=mb - mb0, d_mae_interior=mi - mi0 if interior.any() else np.nan,
        max_abs_diff_spatial=diff_sp, max_abs_diff_imputed=diff_im, equivalent=equivalent,
        **{f"part_{k_}": v for k_, v in meta_part.items() if np.isscalar(v)},
    )
    extra = dict(inf.get("row_extra", {}))
    if "rt_t_dist_wall" in extra:       # ejecución con procesos reales
        extra["speedup_real_wall"] = ref.t_central / extra["rt_t_dist_wall"]
        extra["speedup_cpu_est"] = ref.t_central / extra["rt_t_dist_cpu_est"]
        extra["efficiency_cpu_est"] = extra["speedup_cpu_est"] / plan.k
        extra["model_error_rel_cpu"] = (t_dist - extra["rt_t_dist_cpu_est"]) / extra["rt_t_dist_cpu_est"]
    row.update(extra)
    return (row, sim, plan) if return_sim else row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_rows_csv(rows: list, path) -> None:
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ejecución distribuida simulada (Equipo T800)")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--methods", nargs="+", default=["hash", "spatial", "community"])
    ap.add_argument("--k", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--depth", type=int, nargs="+", default=None,
                    help="Profundidades de halo a probar (por defecto: la requerida)")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--impute-iters", type=int, default=2)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bandwidth-mbps", type=float, default=1000.0)
    ap.add_argument("--latency-ms", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=3, help="Repeticiones de cronometraje (mínimo)")
    ap.add_argument("--csv", default=None, help="Guardar todas las filas en este CSV")
    a = ap.parse_args()

    g, s = load_graph(a.graph), load_signals(a.signals)
    cfg = DetectorConfig(hops=a.hops, impute_iters=a.impute_iters)
    cost = CostModel(bandwidth_Bps=a.bandwidth_mbps * 1e6 / 8, latency_s=a.latency_ms / 1e3)
    ref = prepare_reference(g, s, cfg, tau=a.tau, repeats=a.repeats)
    print(f"centralizado: {ref.t_central:.3f}s | tau={ref.tau:.2f} ({ref.tau_source}) | "
          f"halo requerido={cfg.halo_depth} | red modelada: {a.bandwidth_mbps:g} Mbit/s, "
          f"{a.latency_ms:g} ms\n")
    print(f"{'método':18s} {'k':>2s} {'halo':>4s} {'corte':>6s} {'MB halo':>8s} {'speedup*':>8s} "
          f"{'efic*':>6s} {'infl.':>6s} {'ΔAUC':>7s} {'ΔF1':>7s} {'ΔMAE':>7s} {'max|Δ|':>8s} equiv")
    rows = []
    for method in a.methods:
        for k in a.k:
            for depth in (a.depth if a.depth is not None else [cfg.halo_depth]):
                r = run_distributed(g, s, ref, method, k, depth, cfg, a.seed, cost, repeats=a.repeats)
                rows.append(r)
                print(f"{method:18s} {k:2d} {depth:4d} {r['cut_ratio']:6.3f} {r['halo_bytes'] / 1e6:8.2f} "
                      f"{r['speedup_model']:8.2f} {r['efficiency_model']:6.2f} {r['work_inflation']:6.2f} "
                      f"{r['d_auc']:7.3f} {r['d_f1']:7.3f} {r['d_mae']:7.3f} "
                      f"{r['max_abs_diff_spatial']:8.1e} {'SI' if r['equivalent'] else 'no'}")
    print("\n* modelado (cota ideal: k núcleos libres, sin sobrecarga de serialización ni de Ray).")
    if a.csv:
        write_rows_csv(rows, a.csv)
        print(f"Filas guardadas en {a.csv}")


if __name__ == "__main__":
    main()
