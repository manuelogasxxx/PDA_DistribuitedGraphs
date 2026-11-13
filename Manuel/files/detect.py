#!/usr/bin/env python3
"""
detect.py  (v1.0.0)

Detector de anomalías e imputación, DISEÑADO PARA SER DISTRIBUIBLE.

Idea de diseño: el resultado de cada nodo depende solo de su vecindad local.
Por eso un worker que tenga sus nodos propios más un halo de profundidad
suficiente obtiene EXACTAMENTE el mismo resultado que el método centralizado
en sus nodos propios (se verifica en test_detect.py). No hay estadísticos
globales (la v anterior usaba mediana/MAD globales, que exigen una reducción
entre workers y rompen la equivalencia).

Componentes
  spatial_scores   Residuo contra la media de los vecinos OBSERVADOS a h saltos,
                   normalizado por la mediana y MAD de la SERIE DEL PROPIO NODO.
                   Dependencia: vecindad de h saltos  ->  halo de profundidad h.
  temporal_scores  Ajuste armónico robusto por nodo (constante + tendencia +
                   ciclo diario, con recorte iterativo de atípicos).
                   Dependencia: solo el propio nodo   ->  CERO comunicación.
  combined         max(spatial, temporal).
  impute           Rellena huecos con la media de vecinos disponibles, en
                   `impute_iters` iteraciones síncronas (cada una avanza un
                   salto dentro de una zona apagada). Lo que queda sin llenar
                   usa la predicción temporal del propio nodo.
                   Dependencia: impute_iters saltos -> halo de esa profundidad.

Relación con el halo (lo que el proyecto mide):
  profundidad de halo requerida = max(hops, impute_iters)
  hops=1 e impute_iters=1 => halo de 1 salto; subir cualquiera cuesta bytes.

Las funciones centrales reciben (X, mask, A) de un subgrafo cualquiera, de modo
que un worker las llama con su subgrafo local (propios + halo) y se queda con
las columnas de sus nodos propios.

Uso:
  python detect.py --graph ./data/hybrid_n10000_seed1 \
      --signals ./data/hybrid_n10000_seed1/signals_seed1_both_structured.npz
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from iot_graph_generator import load_graph
from signals import _graph_view, load_signals


@dataclass(frozen=True)
class DetectorConfig:
    hops: int = 1              # profundidad espacial del detector
    impute_iters: int = 3      # iteraciones de imputación espacial
    period: int = 48           # pasos por ciclo diario (parámetro conocido del dominio)
    min_samples: int = 8       # mínimo de residuos válidos por nodo para normalizar
    temporal_iters: int = 3    # iteraciones del ajuste robusto
    temporal_k: float = 3.0    # recorte de atípicos en el ajuste (en desviaciones)
    ridge: float = 1e-3        # regularización de las ecuaciones normales

    @property
    def halo_depth(self) -> int:
        """Profundidad de halo que necesita un worker para igualar al centralizado."""
        return max(self.hops, self.impute_iters)


@dataclass
class DetectionResult:
    scores_spatial: np.ndarray     # (T, N) float32
    scores_temporal: np.ndarray
    scores_combined: np.ndarray
    X_imputed: np.ndarray
    info: dict

    def scores(self, kind: str = "spatial") -> np.ndarray:
        return {"spatial": self.scores_spatial, "temporal": self.scores_temporal,
                "combined": self.scores_combined}[kind]


# ---------------------------------------------------------------------------
# Estructuras de grafo
# ---------------------------------------------------------------------------

def _binarize_no_diag(M: csr_matrix) -> csr_matrix:
    c = M.tocoo()
    keep = c.row != c.col
    return coo_matrix((np.ones(int(keep.sum()), dtype=np.float32), (c.row[keep], c.col[keep])),
                      shape=M.shape).tocsr()


def adjacency_hops(n: int, edges: np.ndarray, hops: int = 1) -> csr_matrix:
    """Adyacencia binaria simétrica de la vecindad a `hops` saltos (sin el propio nodo)."""
    if hops < 1:
        raise ValueError("hops debe ser >= 1")
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    A1 = coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n)).tocsr()
    A1 = _binarize_no_diag(A1)
    R = A1
    for _ in range(hops - 1):
        R = _binarize_no_diag(R @ A1 + R)
    return R


def induced_edges(n: int, edges: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Aristas del subgrafo inducido por `nodes`, reetiquetadas 0..len(nodes)-1
    según la posición en `nodes`. Es lo que construye cada worker (propios + halo)."""
    local = -np.ones(n, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    keep = (local[edges[:, 0]] >= 0) & (local[edges[:, 1]] >= 0)
    return local[edges[keep]]


# ---------------------------------------------------------------------------
# Detector espacial
# ---------------------------------------------------------------------------

def _neighbor_mean(A: csr_matrix, X: np.ndarray, have: np.ndarray):
    """Media de los vecinos con dato (have=True) y conteo de vecinos con dato."""
    m = have.astype(np.float32)
    num = np.ascontiguousarray((A @ (np.nan_to_num(X) * m).T).T)
    den = np.ascontiguousarray((A @ m.T).T)
    return num, den


def spatial_scores(X: np.ndarray, mask: np.ndarray, A_h: csr_matrix,
                   cfg: DetectorConfig = DetectorConfig()) -> np.ndarray:
    """Score espacial (T, N). 0 donde no es calculable (sin dato, sin vecinos
    observados o con menos de `min_samples` residuos válidos en el nodo)."""
    num, den = _neighbor_mean(A_h, X, mask)
    valid = mask & (den > 0)
    resid = np.where(valid, np.nan_to_num(X) - num / np.maximum(den, 1e-9), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(resid, axis=0)
        mad = np.nanmedian(np.abs(resid - med[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    ok_node = valid.sum(axis=0) >= cfg.min_samples
    z = np.abs(resid - med[None, :]) / scale[None, :]
    z = np.where(valid & ok_node[None, :], z, 0.0)
    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# Detector temporal (por nodo, sin comunicación)
# ---------------------------------------------------------------------------

def _design(T: int, period: int) -> np.ndarray:
    t = np.arange(T, dtype=np.float64)
    return np.stack([np.ones(T), t / T, np.cos(2 * np.pi * t / period),
                     np.sin(2 * np.pi * t / period)], axis=1)


def temporal_fit(X: np.ndarray, mask: np.ndarray, cfg: DetectorConfig = DetectorConfig()):
    """Ajuste robusto por nodo. Devuelve (predicción (T,N), residuo (T,N), escala (N,))."""
    T, n = X.shape
    Phi = _design(T, cfg.period)
    p = Phi.shape[1]
    Phi2 = (Phi[:, :, None] * Phi[:, None, :]).reshape(T, p * p)
    Xz = np.nan_to_num(X).astype(np.float64)
    w = mask.astype(np.float64)
    eye = cfg.ridge * np.eye(p)
    for _ in range(cfg.temporal_iters):
        G = (w.T @ Phi2).reshape(n, p, p) + eye
        b = (w * Xz).T @ Phi
        coef = np.linalg.solve(G, b[..., None])[..., 0]
        pred = Phi @ coef.T
        res = Xz - pred
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = 1.4826 * np.nanmedian(np.where(mask, np.abs(res), np.nan), axis=0)
        scale = np.where(np.isfinite(scale), np.maximum(scale, 1e-6), 1.0)
        w = (mask & (np.abs(res) < cfg.temporal_k * scale[None, :])).astype(np.float64)
    return pred.astype(np.float32), res.astype(np.float32), scale


def temporal_scores(X: np.ndarray, mask: np.ndarray, cfg: DetectorConfig = DetectorConfig()):
    pred, res, scale = temporal_fit(X, mask, cfg)
    z = np.where(mask, np.abs(res) / scale[None, :], 0.0).astype(np.float32)
    return z, pred


# ---------------------------------------------------------------------------
# Imputación
# ---------------------------------------------------------------------------

def impute(X: np.ndarray, mask: np.ndarray, A1: csr_matrix, temporal_pred: np.ndarray,
           cfg: DetectorConfig = DetectorConfig()):
    """Imputación espacial iterativa (síncrona) con respaldo temporal.
    Devuelve (X_imputed, info). Nunca deja NaN."""
    Xf = np.where(mask, np.nan_to_num(X), 0.0).astype(np.float32)
    have = mask.copy()
    total_missing = int((~mask).sum())
    by_iter = []
    for _ in range(cfg.impute_iters):
        num, den = _neighbor_mean(A1, Xf, have)
        fill = (~have) & (den > 0)
        if not fill.any():
            break
        Xf[fill] = num[fill] / den[fill]
        have = have | fill
        by_iter.append(int(fill.sum()))
    leftover = ~have
    Xf[leftover] = temporal_pred[leftover]
    info = dict(missing=total_missing, filled_spatial_by_iter=by_iter,
                filled_temporal_fallback=int(leftover.sum()))
    return Xf, info


# ---------------------------------------------------------------------------
# Pipeline centralizado
# ---------------------------------------------------------------------------

def _xm(sig):
    if isinstance(sig, dict):
        return sig["X"], sig["mask"]
    return sig.X, sig.mask


def run_centralized(graph, sig, cfg: DetectorConfig = DetectorConfig()) -> DetectionResult:
    """Detector + imputación sobre el grafo completo en un solo proceso.
    `graph`: GraphData o dict de load_graph. `sig`: SignalData o dict de load_signals."""
    v = _graph_view(graph)
    X, mask = _xm(sig)
    timings = {}

    t0 = time.perf_counter()
    A_h = adjacency_hops(v["n"], v["edges"], cfg.hops)
    A_1 = A_h if cfg.hops == 1 else adjacency_hops(v["n"], v["edges"], 1)
    timings["build_adjacency"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    z_sp = spatial_scores(X, mask, A_h, cfg)
    timings["spatial_scores"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    z_tm, pred = temporal_scores(X, mask, cfg)
    timings["temporal_scores"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_imp, imp_info = impute(X, mask, A_1, pred, cfg)
    timings["impute"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    info = dict(config=asdict(cfg), halo_depth=cfg.halo_depth, timings=timings,
                imputation=imp_info)
    return DetectionResult(z_sp, z_tm, np.maximum(z_sp, z_tm), X_imp, info)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from metrics import best_f1_threshold, evaluate, _select

    ap = argparse.ArgumentParser(description="Detector centralizado + métricas (Equipo T800)")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--signals", required=True, help="Archivo .npz generado por signals.py")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--impute-iters", type=int, default=3)
    ap.add_argument("--period", type=int, default=48)
    ap.add_argument("--tau", type=float, default=None,
                    help="Umbral fijo. Sin él se reporta el umbral de mejor F1 sobre ESTOS datos "
                         "(optimista: solo exploratorio; el protocolo usa datos de calibración)")
    a = ap.parse_args()

    g, s = load_graph(a.graph), load_signals(a.signals)
    cfg = DetectorConfig(hops=a.hops, impute_iters=a.impute_iters, period=a.period)
    res = run_centralized(g, s, cfg)
    print(f"halo requerido: {cfg.halo_depth} salto(s) | tiempo total: "
          f"{res.info['timings']['total']:.2f}s")
    for kind in ("spatial", "temporal", "combined"):
        sc, y = _select(res.scores(kind), s["labels"], s["mask"])
        tau = a.tau if a.tau is not None else best_f1_threshold(sc, y)[0]
        row = evaluate(res.scores(kind), res.X_imputed, s, tau=tau)
        print(f"[{kind:8s}] AUC={row['det_auc']:.3f} AP={row['det_ap']:.3f} "
              f"F1={row['det_f1']:.3f} (tau={tau:.2f}, P={row['det_precision']:.2f}, "
              f"R={row['det_recall']:.2f})")
    print(f"[imputación] MAE={row['imp_mae']:.3f} RMSE={row['imp_rmse']:.3f} "
          f"sobre {row['imp_n_missing']} lecturas faltantes")
    print(json.dumps(res.info["imputation"], indent=2))


if __name__ == "__main__":
    main()
