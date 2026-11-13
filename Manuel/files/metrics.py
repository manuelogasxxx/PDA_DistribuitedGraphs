#!/usr/bin/env python3
"""
metrics.py  (v1.0.0)

Métricas para comparar el método centralizado contra los distribuidos.
Todos los métodos se califican con ESTAS funciones y las mismas reglas.

Reglas de evaluación (parte del protocolo experimental)
  1. Se evalúan todas las lecturas OBSERVADAS (mask=True). Una anomalía cuyo
     score no se pudo calcular (p. ej. sin vecinos observados) cuenta como
     falso negativo: score NaN -> 0. Así, un método que pierde información de
     frontera es penalizado, que es justo lo que se quiere medir.
  2. El umbral NO se ajusta por método. Protocolo recomendado:
        tau = best_f1_threshold(...) sobre datos de CALIBRACIÓN (otras
        semillas) con el método CENTRALIZADO; se congela y se aplica igual a
        centralizado, hash, comunidades y adaptativa.
     Ajustar tau por método ocultaría la degradación por partición.
     AUC y AP no dependen del umbral y son la métrica robusta.
  3. El error de imputación se mide solo sobre lecturas faltantes, contra
     X_clean (la lectura normal que el sensor habría entregado).
  4. 'nodes' (máscara booleana de nodos) permite evaluar subconjuntos, p. ej.
     solo nodos de frontera vs. interiores, para ver DÓNDE se degrada la
     calidad al particionar.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Selección de entradas a evaluar
# ---------------------------------------------------------------------------

def _select(scores, labels, mask, nodes=None):
    """Vectores 1-D (score, y) de las lecturas observadas, opcionalmente en un
    subconjunto de nodos. Score NaN -> 0."""
    sel = mask.copy()
    if nodes is not None:
        sel &= np.asarray(nodes, dtype=bool)[None, :]
    s = np.nan_to_num(scores[sel].astype(np.float64), nan=0.0, posinf=1e12, neginf=0.0)
    return s, labels[sel].astype(bool)


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------

def auc_roc(s: np.ndarray, y: np.ndarray) -> float:
    """AUC por rangos (maneja empates). NaN si no hay positivos o no hay negativos."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(s: np.ndarray, y: np.ndarray) -> float:
    """Average precision: sum_k (R_k - R_{k-1}) * P_k, evaluada en los cortes
    entre scores distintos (los empates se tratan como un solo grupo)."""
    n1 = int(y.sum())
    if n1 == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ss, ys = s[order], y[order]
    tp = np.cumsum(ys)
    last = np.append(ss[1:] != ss[:-1], True)
    precision = tp[last] / (np.nonzero(last)[0] + 1)
    d_recall = np.diff(np.concatenate([[0], tp[last]])) / n1
    return float((precision * d_recall).sum())


def prf_at(s: np.ndarray, y: np.ndarray, tau: float) -> dict:
    """Precisión, recall y F1 prediciendo anomalía si score >= tau."""
    pred = s >= tau
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return dict(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)


def best_f1_threshold(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Umbral que maximiza F1 (predice anomalía si score >= tau). Devuelve (tau, f1).
    Úsalo SOLO en datos de calibración con el método centralizado."""
    n1 = int(y.sum())
    if n1 == 0:
        return float("inf"), 0.0
    order = np.argsort(-s, kind="mergesort")
    ss, ys = s[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    f1 = 2 * tp / (tp + fp + n1)
    last = np.append(ss[1:] != ss[:-1], True)     # evaluar solo en cortes entre valores distintos
    f1 = np.where(last, f1, -1.0)
    i = int(np.argmax(f1))
    return float(ss[i]), float(f1[i])


def detection_report(scores, labels, mask, tau=None, nodes=None) -> dict:
    s, y = _select(scores, labels, mask, nodes)
    rep = dict(n_eval=int(len(y)), n_pos=int(y.sum()),
               auc=auc_roc(s, y), ap=average_precision(s, y))
    if tau is not None:
        rep.update(tau=float(tau), **prf_at(s, y, tau))
    return rep


# ---------------------------------------------------------------------------
# Imputación
# ---------------------------------------------------------------------------

def imputation_report(X_imputed, X_clean, mask, nodes=None) -> dict:
    """MAE y RMSE sobre las lecturas faltantes, contra la lectura normal real."""
    sel = ~mask
    if nodes is not None:
        sel &= np.asarray(nodes, dtype=bool)[None, :]
    n = int(sel.sum())
    if n == 0:
        return dict(n_missing=0, mae=float("nan"), rmse=float("nan"))
    err = X_imputed[sel].astype(np.float64) - X_clean[sel].astype(np.float64)
    nan_frac = float(np.isnan(err).mean())
    err = np.nan_to_num(err, nan=0.0)
    return dict(n_missing=n, mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt((err ** 2).mean())), unfilled_fraction=nan_frac)


# ---------------------------------------------------------------------------
# Evaluación completa de una corrida
# ---------------------------------------------------------------------------

def evaluate(scores, X_imputed, sig, tau=None, nodes=None) -> dict:
    """Fila plana lista para guardar: métricas de detección + imputación.
    `sig`: SignalData o dict de load_signals (necesita labels, mask, X_clean)."""
    get = (lambda k: sig[k]) if isinstance(sig, dict) else (lambda k: getattr(sig, k))
    labels, mask, X_clean = get("labels"), get("mask"), get("X_clean")
    row = {f"det_{k}": v for k, v in detection_report(scores, labels, mask, tau, nodes).items()}
    row.update({f"imp_{k}": v for k, v in imputation_report(X_imputed, X_clean, mask, nodes).items()})
    return row
