#!/usr/bin/env python3
"""
signals.py  (v1.0.0)

Genera las SEÑALES sobre un grafo producido por iot_graph_generator.py:
lecturas X[t, n], anomalías etiquetadas, datos faltantes y la verdad de
referencia para calcular F1/AUC y error de imputación.

Modelo de la señal (lectura normal de un sensor, p. ej. temperatura en °C):

    X_clean[t, n] = base + s[n] + a[n] * sin(2*pi*t/period + phi[n])
                    + trend * t/T + clima(t, n) + ruido blanco

  * s, a, phi son campos ESPACIALMENTE SUAVES (random Fourier features sobre
    las coordenadas, longitud de escala = length_scale_factor * distancia
    mediana de arista). Así, sensores vecinos miden cosas parecidas, que es
    el supuesto físico del proyecto. Se generan a partir de las coordenadas
    y NO del grafo, para que el detector basado en vecinos no tenga ventaja
    circular. Sin coordenadas (modos community/powerlaw) se usa difusión
    sobre el grafo como alternativa, y el manifest lo registra.

Ground truth (convenciones; importantes para las métricas):
  X_clean : lectura normal sin falla ni anomalía (incluye el ruido de medición).
            Es la verdad para el error de imputación.
  X       : lo que ven los métodos: X_clean + anomalías, con NaN donde falta.
  mask    : True = observado. Falta = False.
  labels  : True = lectura anómala. Solo se marcan lecturas OBSERVADAS
            (una lectura faltante no puede detectarse; evita ambigüedad).

Escenarios del protocolo experimental:
  anomaly : none | point | regional | both
              point    -> picos aislados en un sensor y un instante.
              regional -> grupo de vecinos (bola espacial) desviado a la vez
                          durante una ventana de tiempo. OJO: el desvío es
                          común a toda la región, así que un detector basado
                          en vecinos solo lo ve en el borde; es el caso donde
                          la partición y el halo más importan.
  missing : none | random | structured
              random     -> Bernoulli i.i.d.
              structured -> apagones de zona completa por una ventana de tiempo.
  magnitude: tamaño de la anomalía en unidades de r_sd, la desviación robusta
             del residuo "lectura - media de vecinos observados" de la señal
             normal. Es el factor de DIFICULTAD: 3 es difícil, 10 es fácil.

Reproducibilidad: un flujo aleatorio independiente por componente (campo,
ruido, faltantes, anomalías puntuales, anomalías regionales). Con la misma
semilla, cambiar el tipo de anomalía NO altera X_clean ni mask, lo que
permite comparaciones pareadas entre escenarios.

Uso:
  python signals.py --graph ./data/hybrid_n10000_seed42 --seed 1 \
      --anomaly both --anomaly-rate 0.01 --missing structured --missing-rate 0.05

Como librería:
  from iot_graph_generator import generate, load_graph
  from signals import generate_signals, export_signals, load_signals
  sig = generate_signals(generate("hybrid", 10000, seed=1), seed=1, anomaly="regional")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from iot_graph_generator import _versions, load_graph

SIGNALS_VERSION = "1.1.0"
ANOMALY_KINDS = ("none", "point", "regional", "both")
MISSING_KINDS = ("none", "random", "structured")

# Índices FIJOS de flujos aleatorios (no reordenar; agregar nuevos al final).
_STREAMS = {"field": 0, "noise": 1, "missing": 2, "anom_point": 3, "anom_region": 4,
            "weather": 5}

DEFAULTS = dict(
    T=96,                      # pasos de tiempo
    period=48,                 # pasos por ciclo diario
    base=20.0,
    field_std=3.0,             # desviación del campo espacial
    daily_amp=4.0,
    trend=1.0,                 # deriva total sobre T
    noise_sd=0.3,
    weather_std=1.5,           # fluctuación "clima": suave en espacio, impredecible en tiempo
    weather_k=6,               # nº de modos espaciales del clima
    weather_ar=0.95,           # autocorrelación temporal (AR(1)) de cada modo
    length_scale_factor=20.0,  # escala espacial = factor * distancia mediana de arista
    anomaly="point",
    anomaly_rate=0.01,         # fracción de lecturas observadas anómalas (objetivo)
    magnitude=(5.0, 10.0),     # en unidades de r_sd
    missing="random",
    missing_rate=0.05,         # fracción de lecturas faltantes (objetivo)
    region_radius_factor=3.0,  # radio de región = factor * distancia mediana de arista
    region_hops=2,             # radio en saltos si no hay coordenadas
    anomaly_window_frac=(1 / 24, 1 / 6),   # duración de evento anómalo, fracción de T
    missing_window_frac=(1 / 12, 1 / 4),   # duración de apagón, fracción de T
)


# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------

@dataclass
class SignalData:
    X: np.ndarray            # (T, N) float32, NaN donde falta
    X_clean: np.ndarray      # (T, N) float32
    labels: np.ndarray       # (T, N) bool
    mask: np.ndarray         # (T, N) bool, True = observado
    params: dict
    resolved: dict
    report: dict
    graph_sha256: str | None

    def signals_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.labels).tobytes())
        h.update(np.ascontiguousarray(self.mask).tobytes())
        h.update(np.round(self.X_clean, 3).astype(np.float32).tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _make_streams(seed: int) -> dict:
    children = np.random.SeedSequence(seed).spawn(len(_STREAMS))
    return {name: np.random.default_rng(children[i]) for name, i in _STREAMS.items()}


def _graph_view(g) -> dict:
    """Acepta un GraphData (generate) o el dict de load_graph."""
    if isinstance(g, dict):
        return dict(n=int(g["n"]), edges=g["edge_index"].T, coords=g.get("coords"),
                    distance=g.get("distance"), bridge=g.get("is_bridge"),
                    sha=g["manifest"].get("graph_sha256"))
    return dict(n=g.n, edges=g.edges, coords=g.coords, distance=g.distance,
                bridge=g.is_bridge, sha=g.graph_sha256())


def _adjacency(n: int, edges: np.ndarray) -> csr_matrix:
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    return coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n)).tocsr()


def _rff_field(coords: np.ndarray, ell: float, rng, n_comp: int = 64) -> np.ndarray:
    """Campo gaussiano suave (random Fourier features), estandarizado."""
    omega = rng.normal(0.0, 1.0 / ell, size=(n_comp, 2))
    b = rng.uniform(0.0, 2 * np.pi, size=n_comp)
    z = np.sqrt(2.0 / n_comp) * np.cos(coords @ omega.T + b).sum(axis=1)
    return (z - z.mean()) / (z.std() + 1e-12)


def _graph_field(A: csr_matrix, rng, iters: int = 30) -> np.ndarray:
    """Alternativa sin coordenadas: difusión de ruido sobre el grafo."""
    deg = np.asarray(A.sum(axis=1)).ravel().clip(min=1.0)
    z = rng.standard_normal(A.shape[0])
    for _ in range(iters):
        z = 0.5 * z + 0.5 * (A @ z) / deg
    return (z - z.mean()) / (z.std() + 1e-12)


def _neighbor_residual(A: csr_matrix, X: np.ndarray, mask: np.ndarray):
    """residuo = x - media de vecinos OBSERVADOS. Devuelve (resid, valid), (T, N)."""
    m = mask.astype(np.float32)
    num = (A @ (np.nan_to_num(X) * m).T).T
    den = (A @ m.T).T
    valid = mask & (den > 0)
    resid = np.where(valid, X - num / np.maximum(den, 1e-9), np.nan)
    return resid, valid


def _robust_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


class _Regions:
    """Regiones de vecinos: bola espacial (con coordenadas) o BFS en saltos."""

    def __init__(self, view: dict, A: csr_matrix, radius_factor: float, hops: int):
        self.n, self.A, self.hops = view["n"], A, hops
        self.coords = view["coords"]
        if self.coords is not None:
            d = view["distance"]
            core = d[~view["bridge"]] if view["bridge"] is not None and (~view["bridge"]).any() else d
            self.med_edge = float(np.median(core))
            self.radius = radius_factor * self.med_edge
            self.tree = cKDTree(self.coords)

    def ball(self, center: int) -> np.ndarray:
        if self.coords is not None:
            return np.asarray(self.tree.query_ball_point(self.coords[center], self.radius),
                              dtype=np.int64)
        seen = {int(center)}
        frontier = [int(center)]
        for _ in range(self.hops):
            nxt = []
            for u in frontier:
                for v in self.A.indices[self.A.indptr[u]:self.A.indptr[u + 1]]:
                    if int(v) not in seen:
                        seen.add(int(v)); nxt.append(int(v))
            frontier = nxt
        return np.fromiter(seen, dtype=np.int64)


def _window(frac, T: int) -> tuple[int, int]:
    lo = max(2, int(round(frac[0] * T)))
    hi = max(lo, min(T, int(round(frac[1] * T))))
    return lo, hi


# ---------------------------------------------------------------------------
# Datos faltantes
# ---------------------------------------------------------------------------

def _apply_missing(kind, rate, mask, rng, regions, n, T, win, resolved):
    if kind == "none" or rate <= 0:
        return
    if kind == "random":
        mask &= ~(rng.random(mask.shape, dtype=np.float32) < rate)
        resolved["missing_events"] = None
        return
    target = rate * mask.size
    missing, events, attempts = 0, 0, 0
    lo, hi = win
    while missing < target and attempts < 200000:
        attempts += 1
        nodes = regions.ball(int(rng.integers(n)))
        length = int(rng.integers(lo, hi + 1))
        t0 = int(rng.integers(0, T - length + 1))
        block = mask[t0:t0 + length, nodes]
        new = int(block.sum())
        if new:
            mask[t0:t0 + length, nodes] = False
            missing += new
            events += 1
    resolved["missing_events"] = events


# ---------------------------------------------------------------------------
# Anomalías
# ---------------------------------------------------------------------------

def _inject_point(rate, mask, labels, delta, r_sd, mag, rng):
    sel = (rng.random(mask.shape, dtype=np.float32) < rate) & mask & ~labels
    k = int(sel.sum())
    amp = rng.choice([-1.0, 1.0], size=k) * rng.uniform(mag[0], mag[1], size=k) * r_sd
    delta[sel] = amp
    labels |= sel
    return k


def _inject_regional(rate, mask, labels, delta, r_sd, mag, rng, regions, n, T, win):
    target = rate * mask.sum()
    added, events, attempts = 0, 0, 0
    lo, hi = win
    while added < target and attempts < 200000:
        attempts += 1
        nodes = regions.ball(int(rng.integers(n)))
        length = int(rng.integers(lo, hi + 1))
        t0 = int(rng.integers(0, T - length + 1))
        sub = mask[t0:t0 + length, nodes] & ~labels[t0:t0 + length, nodes]
        cnt = int(sub.sum())
        if not cnt:
            continue
        amp = float(rng.choice([-1.0, 1.0]) * rng.uniform(mag[0], mag[1]) * r_sd)
        delta[t0:t0 + length, nodes] = np.where(sub, amp, delta[t0:t0 + length, nodes])
        labels[t0:t0 + length, nodes] |= sub
        added += cnt
        events += 1
    return added, events


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def generate_signals(graph, seed: int, **overrides) -> SignalData:
    """Genera señales sobre un grafo (GraphData o dict de load_graph)."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"Parámetros desconocidos: {sorted(unknown)}")
    p = {**DEFAULTS, **overrides}
    if p["anomaly"] not in ANOMALY_KINDS or p["missing"] not in MISSING_KINDS:
        raise ValueError(f"anomaly ∈ {ANOMALY_KINDS}, missing ∈ {MISSING_KINDS}")
    mag = tuple(p["magnitude"])
    if not 0 < mag[0] <= mag[1]:
        raise ValueError("magnitude debe cumplir 0 < min <= max")
    T, period = int(p["T"]), int(p["period"])
    if T < 8:
        raise ValueError("T debe ser >= 8")

    v = _graph_view(graph)
    n = v["n"]
    A = _adjacency(n, v["edges"])
    streams = _make_streams(seed)
    resolved: dict = {}

    # --- 1. Señal normal --------------------------------------------------
    rf = streams["field"]
    regions = _Regions(v, A, p["region_radius_factor"], p["region_hops"])
    if v["coords"] is not None:
        ell = p["length_scale_factor"] * regions.med_edge
        s, a, ph = (_rff_field(v["coords"], ell, rf) for _ in range(3))
        resolved.update(field_source="coordinates_rff", length_scale=round(ell, 3))
    else:
        s, a, ph = (_graph_field(A, rf) for _ in range(3))
        resolved.update(field_source="graph_diffusion_fallback")
    t = np.arange(T, dtype=np.float32)[:, None]
    signal = (p["base"] + p["field_std"] * s[None, :]
              + (p["daily_amp"] * (1.0 + 0.2 * a))[None, :]
              * np.sin(2 * np.pi * t / period + (0.5 * ph)[None, :])
              + p["trend"] * t / T).astype(np.float32)
    X_clean = signal + (p["noise_sd"] * streams["noise"].standard_normal((T, n),
                                                                          dtype=np.float32))
    del signal

    # Clima: sum_k c_k(t) * f_k(n). f_k suaves en el espacio (mismo supuesto que el
    # campo base) y c_k(t) un AR(1). Los vecinos lo comparten; el historial propio
    # de un sensor NO lo predice. Sin esto, un detector temporal sería un oráculo.
    if p["weather_std"] > 0:
        rw, K, rho = streams["weather"], int(p["weather_k"]), float(p["weather_ar"])
        if v["coords"] is not None:
            F = np.stack([_rff_field(v["coords"], ell, rw) for _ in range(K)])
        else:
            F = np.stack([_graph_field(A, rw) for _ in range(K)])
        C = np.empty((T, K))
        C[0] = rw.standard_normal(K)
        for i in range(1, T):
            C[i] = rho * C[i - 1] + np.sqrt(1 - rho ** 2) * rw.standard_normal(K)
        X_clean += ((p["weather_std"] / np.sqrt(K)) * (C @ F)).astype(np.float32)

    # --- 2. Datos faltantes (antes de las anomalías) ----------------------
    mask = np.ones((T, n), dtype=bool)
    _apply_missing(p["missing"], p["missing_rate"], mask, streams["missing"], regions, n, T,
                   _window(p["missing_window_frac"], T), resolved)

    # --- 3. Escala de dificultad: r_sd -------------------------------------
    resid, _ = _neighbor_residual(A, X_clean, mask)
    r_sd = _robust_sd(resid)
    del resid
    resolved["r_sd"] = round(r_sd, 5)

    # --- 4. Anomalías ------------------------------------------------------
    labels = np.zeros((T, n), dtype=bool)
    delta = np.zeros((T, n), dtype=np.float32)
    kind, rate = p["anomaly"], p["anomaly_rate"]
    n_point = n_region = region_events = 0
    if kind in ("point", "both") and rate > 0:
        n_point = _inject_point(rate / (2 if kind == "both" else 1), mask, labels, delta,
                                r_sd, mag, streams["anom_point"])
    if kind in ("regional", "both") and rate > 0:
        n_region, region_events = _inject_regional(
            rate / (2 if kind == "both" else 1), mask, labels, delta, r_sd, mag,
            streams["anom_region"], regions, n, T, _window(p["anomaly_window_frac"], T))

    X = X_clean + delta
    X[~mask] = np.nan

    # --- 5. Reporte --------------------------------------------------------
    obs = int(mask.sum())
    sig_std = float(X_clean[mask].std())
    report = dict(
        T=T, n_nodes=n, entries=int(mask.size),
        missing_fraction_realized=round(1 - obs / mask.size, 4),
        anomaly_fraction_realized=round(float(labels.sum()) / max(obs, 1), 4),
        anomalies_point=n_point, anomalies_regional=n_region,
        regional_events=region_events,
        signal_std=round(sig_std, 4), r_sd=round(r_sd, 4),
        smoothness_ratio=round(r_sd / sig_std, 4),      # << 1 => vecinos informativos
        anomaly_abs_magnitude=[round(mag[0] * r_sd, 4), round(mag[1] * r_sd, 4)],
    )
    return SignalData(X, X_clean, labels, mask, {"seed": seed, **p}, resolved, report, v["sha"])


# ---------------------------------------------------------------------------
# Exportación y carga
# ---------------------------------------------------------------------------

def export_signals(sig: SignalData, run_dir, tag: str = "") -> Path:
    run_dir = Path(run_dir)
    pr = sig.params
    name = f"signals_seed{pr['seed']}_{pr['anomaly']}_{pr['missing']}" + (f"_{tag}" if tag else "")
    np.savez_compressed(run_dir / f"{name}.npz", X=sig.X, X_clean=sig.X_clean,
                        labels=sig.labels, mask=sig.mask)
    manifest = dict(params=pr, resolved=sig.resolved, report=sig.report,
                    signals_sha256=sig.signals_sha256(), graph_sha256=sig.graph_sha256,
                    signals_version=SIGNALS_VERSION, versions=_versions(),
                    created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    command=" ".join(sys.argv))
    with open(run_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return run_dir / f"{name}.npz"


def load_signals(npz_path) -> dict:
    npz_path = Path(npz_path)
    with np.load(npz_path) as z:
        data = {k: z[k] for k in z.files}
    with open(npz_path.with_suffix(".json"), encoding="utf-8") as f:
        data["manifest"] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    d = DEFAULTS
    ap = argparse.ArgumentParser(description="Señales sintéticas sobre un grafo IoT (Equipo T800)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--graph", required=True, help="Carpeta de corrida del generador de grafos")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--T", type=int, default=d["T"])
    ap.add_argument("--period", type=int, default=d["period"])
    ap.add_argument("--noise-sd", type=float, default=d["noise_sd"])
    ap.add_argument("--anomaly", choices=ANOMALY_KINDS, default=d["anomaly"])
    ap.add_argument("--anomaly-rate", type=float, default=d["anomaly_rate"])
    ap.add_argument("--magnitude-min", type=float, default=d["magnitude"][0])
    ap.add_argument("--magnitude-max", type=float, default=d["magnitude"][1])
    ap.add_argument("--missing", choices=MISSING_KINDS, default=d["missing"])
    ap.add_argument("--missing-rate", type=float, default=d["missing_rate"])
    return ap


def main() -> None:
    a = build_parser().parse_args()
    g = load_graph(a.graph)
    sig = generate_signals(g, a.seed, T=a.T, period=a.period, noise_sd=a.noise_sd,
                           anomaly=a.anomaly, anomaly_rate=a.anomaly_rate,
                           magnitude=(a.magnitude_min, a.magnitude_max),
                           missing=a.missing, missing_rate=a.missing_rate)
    out = export_signals(sig, a.graph, a.tag)
    print(f"Señales guardadas en: {out}")
    print(json.dumps(sig.report, indent=2, ensure_ascii=False))
    print("signals_sha256:", sig.signals_sha256())


if __name__ == "__main__":
    main()
