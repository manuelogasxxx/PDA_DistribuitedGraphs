#!/usr/bin/env python3
"""
iot_graph_generator.py  (v2.0.0)

Generador de grafos sintéticos reproducibles para el proyecto:
"Analítica distribuida basada en grafos para detección de anomalías y calidad
de datos en redes IoT" (Equipo T800).

Este módulo genera SOLO la topología (nodos, aristas, coordenadas, comunidad
de referencia). Las señales (X[t, n], anomalías, datos faltantes) van en un
módulo aparte (signals.py), que consume la salida de este.

Cambios respecto a v1 (cada uno corrige un problema concreto)
-------------------------------------------------------------
1. 'hybrid' ahora es espacialmente coherente: las coordenadas se generan
   PRIMERO (mezcla de gaussianas, una por comunidad) y las aristas salen de
   k-NN / radio sobre esas coordenadas. Antes las aristas inter-comunidad
   unían puntos arbitrarios del plano, rompiendo el supuesto "vecino cercano
   mide algo parecido".
2. Peso = SIMILITUD (exp(-d^2 / 2*sigma^2)), no distancia. Louvain/METIS
   interpretan el peso como fuerza de conexión. La distancia física se guarda
   aparte en 'distance'.
3. Los IDs de nodo se PERMUTAN al final. Sin esto, un particionador por rangos
   de ID recuperaba las comunidades sin mirar el grafo (fuga del ground truth).
4. La fuerza de las comunidades es un FACTOR EXPERIMENTAL explícito:
   --mixing (modo community: fracción esperada de aristas inter-comunidad) y
   --spread-factor (modo hybrid: solapamiento espacial entre clústeres).
   Se reporta siempre el 'mixing_realized'. (v1 usaba 'community_ratio', que
   al escalar n deja de significar lo mismo: con K ~ sqrt(n) las comunidades
   se vuelven indetectables a 100k nodos aunque el ratio sea el mismo.)
5. Número de comunidades que escala con n (--communities auto = sqrt(n)/2,
   mínimo 8) y densidad espacial constante (el área crece con n), de modo que
   los escenarios 1k/10k/100k difieren en ESCALA y no en estructura local.
6. Política explícita para nodos aislados / componentes: en modos espaciales
   se garantiza conectividad añadiendo el árbol de expansión mínimo euclídeo
   (aristas marcadas 'is_bridge'); en 'community' se conserva la componente
   gigante (con guarda: aborta si habría que descartar > 5 % de los nodos).
7. Jerarquía de semillas: un flujo aleatorio INDEPENDIENTE por componente
   (tamaños, centroides, coordenadas, aristas, permutación) derivado de
   np.random.SeedSequence(seed).spawn(). Añadir un flujo nuevo en el futuro
   no altera los existentes.
8. El manifest registra versiones (python, numpy, scipy, networkx), hash de
   git, parámetros resueltos y un SHA-256 del grafo para verificar
   reproducibilidad entre máquinas (Fase 4: congelamiento de versiones).

Rendimiento: todo está vectorizado y es O(n log n) o O(m); no se enumeran
pares de nodos. Pensado para el escenario de 100k nodos.

Modos
-----
  community -> Stochastic Block Model (sparse, vectorizado). Sin coordenadas.
               Topología pura: baseline de partición por comunidad.
  spatial   -> Coordenadas uniformes + k-NN (o radio). Sin comunidades.
               Baseline de partición espacial / geohash.
  powerlaw  -> Barabási-Albert (hubs reales). Sin coordenadas.
               Para estrategias conscientes de grado (DBH/HDRF).
  hybrid    -> Clústeres gaussianos + k-NN espacial. RECOMENDADO: comunidad y
               vecindad física coinciden, como en una red IoT real.

Salida (una carpeta por corrida: <mode>_n<N>_seed<S>[_<tag>]/)
  graph.npz      edge_index (2,m), weight, distance, is_bridge, coords, community
  manifest.json  parámetros, versiones, git, hash, reporte de validación
  nodes.csv / edges.csv   (solo con --csv)

Uso desde la línea de comandos
  python iot_graph_generator.py --mode hybrid --n 1000   --seed 42 --out ./data
  python iot_graph_generator.py --mode hybrid --n 10000  --seed 42 --out ./data
  python iot_graph_generator.py --mode hybrid --n 100000 --seed 42 --out ./data

  # 30 repeticiones (cada semilla = un grafo distinto):
  python iot_graph_generator.py --mode hybrid --n 10000 --seeds 1-30 --out ./data

  # Barrido de fuerza de comunidad (modo community):
  for mu in 0.05 0.1 0.2 0.4; do
    python iot_graph_generator.py --mode community --n 10000 --seed 1 \
        --mixing $mu --tag mu$mu --out ./data
  done

Uso como librería
  from iot_graph_generator import generate, export_graph, load_graph, to_csr
  g = generate("hybrid", n=10000, seed=7)
  run_dir = export_graph(g, "./data")
  data = load_graph(run_dir); A = to_csr(data, weighted=True)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import Delaunay, cKDTree

GENERATOR_VERSION = "2.0.0"
MODES = ("community", "spatial", "powerlaw", "hybrid")
COMPONENT_POLICIES = ("giant", "keep", "strict")

# Índices FIJOS de los flujos aleatorios. No reordenar: cambiaría los grafos
# generados con una semilla dada. Para añadir un flujo, agregarlo al final.
_STREAMS = {"sizes": 0, "centroids": 1, "coords": 2, "edges": 3,
            "permutation": 4, "external": 5}

# Máxima fracción de nodos que la política 'giant' puede descartar.
_MAX_DROP_FRACTION = 0.05

DEFAULTS = dict(
    communities="auto",     # int o "auto"
    avg_degree=10.0,        # community / powerlaw
    mixing=0.1,             # community: fracción esperada de aristas inter-comunidad
    size_cv=0.3,            # variación del tamaño de comunidad (0 = balanceado)
    k_neighbors=6,          # spatial / hybrid (grado resultante ~ 1.5-1.8 * k)
    radius=None,            # spatial / hybrid: radio fijo en vez de k-NN
    density=1e-3,           # spatial / hybrid: nodos por unidad de área (constante en n)
    cluster_spread=None,    # hybrid: sigma absoluto (anula spread_factor)
    spread_factor=0.2,      # hybrid: sigma = spread_factor * lado de celda
    connect=True,           # spatial / hybrid: forzar conectividad (EMST)
    component_policy="giant",
)


# ---------------------------------------------------------------------------
# Estructuras internas
# ---------------------------------------------------------------------------

@dataclass
class _Raw:
    """Grafo en construcción. Las aristas son (src < dst), sin duplicados."""
    n: int
    edges: np.ndarray               # (m, 2) int64
    bridge: np.ndarray              # (m,) bool
    coords: np.ndarray | None       # (n, 2) float64
    community: np.ndarray | None    # (n,) int64

    def keep_nodes(self, mask: np.ndarray) -> None:
        """Conserva solo los nodos con mask=True y reetiqueta 0..n'-1."""
        new_id = -np.ones(self.n, dtype=np.int64)
        new_id[mask] = np.arange(int(mask.sum()))
        keep = mask[self.edges[:, 0]] & mask[self.edges[:, 1]]
        self.edges = new_id[self.edges[keep]]
        self.bridge = self.bridge[keep]
        if self.coords is not None:
            self.coords = self.coords[mask]
        if self.community is not None:
            self.community = self.community[mask]
        self.n = int(mask.sum())

    def permute(self, perm: np.ndarray) -> None:
        """El nodo i pasa a llamarse perm[i]; deja las aristas en orden canónico."""
        inv = np.empty_like(perm)
        inv[perm] = np.arange(self.n)
        if self.coords is not None:
            self.coords = self.coords[inv]
        if self.community is not None:
            self.community = self.community[inv]
        e = perm[self.edges]
        lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
        order = np.lexsort((hi, lo))
        self.edges = np.stack([lo[order], hi[order]], axis=1)
        self.bridge = self.bridge[order]


@dataclass
class GraphData:
    n: int
    edges: np.ndarray               # (m, 2), src < dst, aristas no dirigidas
    weight: np.ndarray              # (m,) similitud (1.0 en modos topológicos)
    distance: np.ndarray | None     # (m,) distancia euclídea, si hay coordenadas
    is_bridge: np.ndarray           # (m,) bool, arista añadida para conectividad
    coords: np.ndarray | None       # (n, 2)
    community: np.ndarray | None    # (n,) ground truth de comunidad
    params: dict
    resolved: dict
    report: dict

    def edge_index(self) -> np.ndarray:
        """Formato PyTorch Geometric (2, m), una sola dirección."""
        return self.edges.T.copy()

    def graph_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.edges, dtype=np.int64).tobytes())
        if self.community is not None:
            h.update(np.ascontiguousarray(self.community, dtype=np.int64).tobytes())
        if self.coords is not None:
            h.update(np.round(self.coords, 6).tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _make_streams(seed: int):
    """Flujos aleatorios independientes (uno por componente del generador)."""
    children = np.random.SeedSequence(seed).spawn(len(_STREAMS))
    streams = {name: np.random.default_rng(children[i]) for name, i in _STREAMS.items()}
    external_seed = int(children[_STREAMS["external"]].generate_state(1)[0])
    return streams, external_seed


def _canon_edges(edges: np.ndarray, n: int) -> np.ndarray:
    """Aristas (src < dst), sin auto-lazos ni duplicados, orden lexicográfico."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    ok = lo != hi
    key = np.unique(lo[ok] * n + hi[ok])
    return np.stack([key // n, key % n], axis=1)


def _resolve_communities(value, n: int) -> int:
    if value in (None, "auto"):
        k = int(max(8, round(np.sqrt(n) / 2)))
        return max(2, min(k, n // 5))      # ajuste para n pequeños
    k = int(value)
    if k < 2 or k * 5 > n:
        raise ValueError(f"communities={k} inválido para n={n} (mín. 2, y 5 nodos por comunidad)")
    return k


def _community_sizes(n: int, k: int, size_cv: float, rng, min_size: int = 5) -> np.ndarray:
    """Tamaños de comunidad con variación lognormal (coef. de variación ~ size_cv)."""
    if size_cv > 0:
        sigma = np.sqrt(np.log(1.0 + size_cv ** 2))
        w = rng.lognormal(-sigma ** 2 / 2, sigma, size=k)
    else:
        w = np.ones(k)
    rest = n - k * min_size
    raw = w / w.sum() * rest
    extra = np.floor(raw).astype(np.int64)
    sizes = min_size + extra
    remainder = n - int(sizes.sum())
    if remainder > 0:
        sizes[np.argsort(-(raw - extra))[:remainder]] += 1
    assert sizes.sum() == n
    return sizes


# ---------------------------------------------------------------------------
# Modo community: Stochastic Block Model (sparse, vectorizado)
# ---------------------------------------------------------------------------

def _sample_sbm_edges(sizes: np.ndarray, avg_degree: float, mixing: float, rng) -> np.ndarray:
    """SBM exacto por pares de bloques, sin enumerar pares de nodos.

    Se calibra para que el grado promedio sea `avg_degree` y la fracción
    esperada de aristas inter-comunidad sea `mixing`, INDEPENDIENTEMENTE de n
    y de K:
        E_intra = (1-mixing) * E,  E_inter = mixing * E,  E = avg_degree*n/2.
    """
    n = int(sizes.sum())
    k = len(sizes)
    s = sizes.astype(np.float64)
    pairs_in = float((s * (s - 1) / 2).sum())
    pairs_out = float((s.sum() ** 2 - (s ** 2).sum()) / 2)
    total_edges = avg_degree * n / 2.0
    p_in = (1.0 - mixing) * total_edges / pairs_in
    p_out = mixing * total_edges / pairs_out
    if p_in > 1.0:
        raise ValueError(
            f"Parámetros inconsistentes: probabilidad intra-comunidad = {p_in:.2f} > 1. "
            f"Reduce --avg-degree o --communities (comunidades demasiado pequeñas)."
        )
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    chunks = []

    # Aristas dentro de cada comunidad: pares i<j codificados como índice triangular.
    for a in range(k):
        sa = int(sizes[a])
        t = sa * (sa - 1) // 2
        m = int(rng.binomial(t, p_in))
        if m == 0:
            continue
        idx = rng.choice(t, size=m, replace=False).astype(np.int64)
        j = ((1 + np.sqrt(1 + 8 * idx.astype(np.float64))) // 2).astype(np.int64)
        j = np.where(j * (j - 1) // 2 > idx, j - 1, j)          # corrección por redondeo
        j = np.where((j + 1) * j // 2 <= idx, j + 1, j)
        i = idx - j * (j - 1) // 2
        chunks.append(np.stack([offsets[a] + i, offsets[a] + j], axis=1))

    # Aristas entre comunidades: conteos vectorizados, bucle solo en pares no vacíos.
    ia, ib = np.triu_indices(k, 1)
    t_out = sizes[ia].astype(np.int64) * sizes[ib].astype(np.int64)
    counts = rng.binomial(t_out, p_out)
    for q in np.nonzero(counts)[0]:
        a, b = int(ia[q]), int(ib[q])
        sb = int(sizes[b])
        idx = rng.choice(int(t_out[q]), size=int(counts[q]), replace=False).astype(np.int64)
        chunks.append(np.stack([offsets[a] + idx // sb, offsets[b] + idx % sb], axis=1))

    edges = np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.int64)
    return _canon_edges(edges, n)


def _build_community(n, p, streams, resolved) -> _Raw:
    k = _resolve_communities(p["communities"], n)
    sizes = _community_sizes(n, k, p["size_cv"], streams["sizes"])
    edges = _sample_sbm_edges(sizes, p["avg_degree"], p["mixing"], streams["edges"])
    resolved.update(communities=k, community_sizes_min=int(sizes.min()),
                    community_sizes_max=int(sizes.max()))
    return _Raw(n, edges, np.zeros(len(edges), bool), None, np.repeat(np.arange(k), sizes))


# ---------------------------------------------------------------------------
# Modos espaciales: k-NN / radio + conectividad por EMST
# ---------------------------------------------------------------------------

def _emst_edges(coords: np.ndarray) -> np.ndarray:
    """Árbol de expansión mínimo euclídeo vía Delaunay (exacto, O(n log n))."""
    n = len(coords)
    try:
        s = Delaunay(coords).simplices
    except Exception as exc:  # QhullError y similares
        raise RuntimeError(f"Delaunay falló (¿puntos degenerados?): {exc}") from exc
    e = _canon_edges(np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]]), n)
    d = np.linalg.norm(coords[e[:, 0]] - coords[e[:, 1]], axis=1) + 1e-12
    g = coo_matrix((d, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(g).tocoo()
    return np.stack([mst.row, mst.col], axis=1).astype(np.int64)


def _spatial_edges(coords: np.ndarray, k: int, radius, connect: bool):
    n = len(coords)
    tree = cKDTree(coords)
    if radius is not None:
        e = tree.query_pairs(r=float(radius), output_type="ndarray")
    else:
        kk = min(k + 1, n)
        _, idx = tree.query(coords, k=kk, workers=-1)
        e = np.stack([np.repeat(np.arange(n), kk - 1), idx[:, 1:].ravel()], axis=1)
    e = _canon_edges(e, n)
    bridge = np.zeros(len(e), bool)

    if connect and n >= 4:
        mst = _emst_edges(coords)
        allx = _canon_edges(np.concatenate([e, mst]), n)
        key_all = allx[:, 0] * n + allx[:, 1]
        key_knn = e[:, 0] * n + e[:, 1]
        key_mst = mst[:, 0] * n + mst[:, 1]
        bridge = np.isin(key_all, key_mst) & ~np.isin(key_all, key_knn)
        e = allx
    return e, bridge


def _grid_centroids(k: int, cell: float, rng, jitter: float = 0.6) -> np.ndarray:
    """Centroides en rejilla con jitter: separación mínima garantizada, a
    diferencia de uniforme puro (que puede fusionar dos comunidades)."""
    g = int(np.ceil(np.sqrt(k)))
    cells = rng.choice(g * g, size=k, replace=False)
    base = np.stack([(cells % g + 0.5) * cell, (cells // g + 0.5) * cell], axis=1)
    return base + rng.uniform(-0.5 * jitter, 0.5 * jitter, size=(k, 2)) * cell


def _build_spatial(n, p, streams, resolved) -> _Raw:
    area = float(np.sqrt(n / p["density"]))
    coords = streams["coords"].uniform(0, area, size=(n, 2))
    e, bridge = _spatial_edges(coords, p["k_neighbors"], p["radius"], p["connect"])
    resolved.update(area_side=round(area, 3))
    return _Raw(n, e, bridge, coords, None)


def _build_hybrid(n, p, streams, resolved) -> _Raw:
    k = _resolve_communities(p["communities"], n)
    sizes = _community_sizes(n, k, p["size_cv"], streams["sizes"])
    area = float(np.sqrt(n / p["density"]))
    g = int(np.ceil(np.sqrt(k)))
    cell = area / g
    centroids = _grid_centroids(k, cell, streams["centroids"])
    spread = float(p["cluster_spread"]) if p["cluster_spread"] is not None \
        else float(p["spread_factor"]) * cell
    community = np.repeat(np.arange(k), sizes)
    coords = centroids[community] + streams["coords"].normal(0.0, spread, size=(n, 2))
    e, bridge = _spatial_edges(coords, p["k_neighbors"], p["radius"], p["connect"])
    resolved.update(communities=k, area_side=round(area, 3), cell_side=round(cell, 3),
                    cluster_sigma=round(spread, 3), community_sizes_min=int(sizes.min()),
                    community_sizes_max=int(sizes.max()))
    return _Raw(n, e, bridge, coords, community)


def _build_powerlaw(n, p, external_seed, resolved) -> _Raw:
    import networkx as nx  # solo este modo lo necesita
    m = max(1, int(round(p["avg_degree"] / 2)))
    G = nx.barabasi_albert_graph(n, m, seed=external_seed)
    e = _canon_edges(np.array(G.edges(), dtype=np.int64), n)
    resolved.update(ba_m=m)
    return _Raw(n, e, np.zeros(len(e), bool), None, None)


# ---------------------------------------------------------------------------
# Política de componentes
# ---------------------------------------------------------------------------

def _csr(n: int, edges: np.ndarray) -> csr_matrix:
    return coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()


def _apply_component_policy(raw: _Raw, policy: str, resolved: dict) -> None:
    ncomp, labels = connected_components(_csr(raw.n, raw.edges), directed=False)
    resolved["components_before_policy"] = int(ncomp)
    if ncomp == 1:
        return
    sizes = np.bincount(labels)
    giant = int(np.argmax(sizes))
    dropped = raw.n - int(sizes[giant])
    if policy == "strict":
        raise RuntimeError(f"El grafo tiene {ncomp} componentes (política 'strict').")
    if policy == "keep":
        return
    if dropped / raw.n > _MAX_DROP_FRACTION:
        raise RuntimeError(
            f"Política 'giant' descartaría {dropped} nodos ({dropped / raw.n:.1%}) > "
            f"{_MAX_DROP_FRACTION:.0%}. Sube --avg-degree o usa --component-policy keep."
        )
    raw.keep_nodes(labels == giant)
    resolved["nodes_dropped_by_policy"] = dropped


# ---------------------------------------------------------------------------
# Validación (chequeo de sanidad antes de particionar)
# ---------------------------------------------------------------------------

def _validate(raw: _Raw, weight: np.ndarray, distance: np.ndarray | None) -> dict:
    n, e = raw.n, raw.edges
    m = len(e)
    deg = np.bincount(e.ravel(), minlength=n)
    ncomp, labels = connected_components(_csr(n, e), directed=False)
    giant = int(np.bincount(labels).max())

    rep = {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": round(float(deg.mean()), 3),
        "degree_std": round(float(deg.std()), 3),
        "degree_p50_p95_max": [int(np.percentile(deg, 50)), int(np.percentile(deg, 95)),
                               int(deg.max())],
        "isolated_nodes": int((deg == 0).sum()),
        "n_connected_components": int(ncomp),
        "giant_component_fraction": round(giant / n, 4),
        "bridge_edges": int(raw.bridge.sum()),
    }

    if raw.community is not None:
        c = raw.community
        k = int(c.max()) + 1
        cu, cv = c[e[:, 0]], c[e[:, 1]]
        intra = cu == cv
        # Modularidad TOPOLÓGICA (sin pesos) de la comunidad plantada.
        l_c = np.bincount(cu[intra], minlength=k)
        d_c = np.bincount(c, weights=deg, minlength=k)
        q = float((l_c / m - (d_c / (2.0 * m)) ** 2).sum())
        csz = np.bincount(c, minlength=k)
        rep.update(
            n_communities=k,
            community_size_min_median_max=[int(csz.min()), int(np.median(csz)), int(csz.max())],
            modularity_ground_truth=round(q, 4),
            mixing_realized=round(float(1.0 - intra.mean()), 4),
        )

    if distance is not None and m:
        core = distance[~raw.bridge] if (~raw.bridge).any() else distance
        med = float(np.median(core))
        rep.update(
            edge_distance_median=round(med, 3),
            edge_distance_p95=round(float(np.percentile(core, 95)), 3),
            edge_distance_max_over_median=round(float(distance.max()) / med, 2) if med else None,
        )
    return rep


# ---------------------------------------------------------------------------
# Generación (API principal)
# ---------------------------------------------------------------------------

def generate(mode: str, n: int, seed: int, **overrides) -> GraphData:
    """Genera un grafo sintético. Mismos (mode, n, seed, parámetros) => mismo grafo."""
    if mode not in MODES:
        raise ValueError(f"mode debe ser uno de {MODES}")
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"Parámetros desconocidos: {sorted(unknown)}")
    if n < 20:
        raise ValueError("n debe ser >= 20")
    p = {**DEFAULTS, **overrides}
    if p["component_policy"] not in COMPONENT_POLICIES:
        raise ValueError(f"component_policy debe ser uno de {COMPONENT_POLICIES}")
    if not 0.0 <= p["mixing"] <= 1.0:
        raise ValueError("mixing debe estar en [0, 1]")

    t0 = time.perf_counter()
    streams, external_seed = _make_streams(seed)
    resolved: dict = {"n_requested": n}

    if mode == "community":
        raw = _build_community(n, p, streams, resolved)
    elif mode == "spatial":
        raw = _build_spatial(n, p, streams, resolved)
    elif mode == "powerlaw":
        raw = _build_powerlaw(n, p, external_seed, resolved)
    else:
        raw = _build_hybrid(n, p, streams, resolved)

    _apply_component_policy(raw, p["component_policy"], resolved)

    # Permutar IDs: ningún particionador por rango de ID puede "ver" la comunidad.
    raw.permute(streams["permutation"].permutation(raw.n))

    # Pesos: similitud gaussiana (modos con coordenadas) o 1.0 (topológicos).
    if raw.coords is not None:
        distance = np.linalg.norm(raw.coords[raw.edges[:, 0]] - raw.coords[raw.edges[:, 1]], axis=1)
        core = distance[~raw.bridge] if (~raw.bridge).any() else distance
        sigma = max(float(np.median(core)), 1e-12)
        weight = np.exp(-(distance ** 2) / (2.0 * sigma ** 2))
        resolved["weight_sigma"] = round(sigma, 4)
    else:
        distance = None
        weight = np.ones(len(raw.edges))

    report = _validate(raw, weight, distance)
    report["generation_time_seconds"] = round(time.perf_counter() - t0, 3)

    params = {"mode": mode, "n": n, "seed": seed, **p}
    return GraphData(raw.n, raw.edges, weight, distance, raw.bridge, raw.coords,
                     raw.community, params, resolved, report)


# ---------------------------------------------------------------------------
# Exportación y carga
# ---------------------------------------------------------------------------

def _git_hash():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parent, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict:
    v = {"generator": GENERATOR_VERSION, "python": platform.python_version(),
         "numpy": np.__version__, "platform": platform.platform()}
    import scipy
    v["scipy"] = scipy.__version__
    try:
        import networkx
        v["networkx"] = networkx.__version__
    except ImportError:
        v["networkx"] = None
    return v


def export_graph(g: GraphData, out_dir, tag: str = "", write_csv: bool = False) -> Path:
    name = f"{g.params['mode']}_n{g.params['n']}_seed{g.params['seed']}" + (f"_{tag}" if tag else "")
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    arrays = {"edge_index": g.edge_index().astype(np.int32),
              "weight": g.weight, "is_bridge": g.is_bridge.astype(np.uint8)}
    if g.distance is not None:
        arrays["distance"] = g.distance
    if g.coords is not None:
        arrays["coords"] = g.coords
    if g.community is not None:
        arrays["community"] = g.community.astype(np.int32)
    np.savez_compressed(run_dir / "graph.npz", **arrays)

    if write_csv:
        cols, fmt, head = [np.arange(g.n)], ["%d"], ["id"]
        if g.coords is not None:
            cols += [g.coords[:, 0], g.coords[:, 1]]; fmt += ["%.4f", "%.4f"]; head += ["x", "y"]
        if g.community is not None:
            cols.append(g.community); fmt.append("%d"); head.append("community")
        np.savetxt(run_dir / "nodes.csv", np.column_stack(cols), delimiter=",",
                   header=",".join(head), comments="", fmt=fmt)
        cols, fmt, head = [g.edges[:, 0], g.edges[:, 1], g.weight], ["%d", "%d", "%.6g"], \
            ["source", "target", "weight"]
        if g.distance is not None:
            cols.append(g.distance); fmt.append("%.6g"); head.append("distance")
        cols.append(g.is_bridge.astype(int)); fmt.append("%d"); head.append("is_bridge")
        np.savetxt(run_dir / "edges.csv", np.column_stack(cols), delimiter=",",
                   header=",".join(head), comments="", fmt=fmt)

    manifest = {
        "params": g.params,
        "resolved": g.resolved,
        "validation": g.report,
        "graph_sha256": g.graph_sha256(),
        "git_commit": _git_hash(),
        "versions": _versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return run_dir


def load_graph(run_dir) -> dict:
    """Carga graph.npz + manifest.json. Devuelve dict con arrays y 'manifest'."""
    run_dir = Path(run_dir)
    with np.load(run_dir / "graph.npz") as z:
        data = {k: z[k] for k in z.files}
    data["edge_index"] = data["edge_index"].astype(np.int64)
    data["is_bridge"] = data["is_bridge"].astype(bool)
    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        data["manifest"] = json.load(f)
    data["n"] = int(data["manifest"]["validation"]["n_nodes"])
    return data


def to_csr(data: dict, weighted: bool = True, symmetric: bool = True) -> csr_matrix:
    """Matriz de adyacencia dispersa (n x n), simétrica por defecto."""
    n = data["n"]
    src, dst = data["edge_index"]
    w = data["weight"] if weighted else np.ones(len(src))
    if symmetric:
        src, dst, w = np.concatenate([src, dst]), np.concatenate([dst, src]), np.concatenate([w, w])
    return coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generador de grafos sintéticos IoT reproducibles (Equipo T800)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--n", type=int, required=True,
                    help="Número de nodos (protocolo: 1000, 10000, 100000)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int, help="Semilla única (obligatoria para reproducibilidad)")
    g.add_argument("--seeds", type=str, help='Varias semillas, p. ej. "1-30" o "1,5,9-12"')
    ap.add_argument("--out", type=str, default="./data")
    ap.add_argument("--tag", type=str, default="",
                    help="Sufijo del nombre de carpeta (útil en barridos de parámetros)")
    ap.add_argument("--csv", action="store_true", help="Escribir también nodes.csv y edges.csv")

    d = DEFAULTS
    ap.add_argument("--communities", type=str, default=d["communities"],
                    help='Número de comunidades (community/hybrid) o "auto" (~sqrt(n)/2, mín. 8)')
    ap.add_argument("--avg-degree", type=float, default=d["avg_degree"],
                    help="Grado promedio objetivo (community/powerlaw). IoT realista: 5-20")
    ap.add_argument("--mixing", type=float, default=d["mixing"],
                    help="[community] fracción esperada de aristas inter-comunidad (0.05 clara .. 0.5 débil)")
    ap.add_argument("--size-cv", type=float, default=d["size_cv"],
                    help="Variación del tamaño de comunidad (0 = todas iguales)")
    ap.add_argument("--k-neighbors", type=int, default=d["k_neighbors"],
                    help="[spatial/hybrid] k del k-NN (grado resultante ~ 1.5-1.8 k)")
    ap.add_argument("--radius", type=float, default=d["radius"],
                    help="[spatial/hybrid] radio fijo en vez de k-NN")
    ap.add_argument("--density", type=float, default=d["density"],
                    help="[spatial/hybrid] nodos por unidad de área (el área crece con n)")
    ap.add_argument("--spread-factor", type=float, default=d["spread_factor"],
                    help="[hybrid] sigma del clúster = factor * lado de celda (0.1 clara .. 0.5 solapada)")
    ap.add_argument("--cluster-spread", type=float, default=d["cluster_spread"],
                    help="[hybrid] sigma absoluto (anula --spread-factor)")
    ap.add_argument("--no-connect", dest="connect", action="store_false",
                    help="[spatial/hybrid] NO forzar conectividad con el EMST")
    ap.add_argument("--component-policy", choices=COMPONENT_POLICIES, default=d["component_policy"],
                    help="giant: conserva la componente gigante | keep: no toca | strict: error si no es conexo")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    seeds = [args.seed] if args.seed is not None else _parse_seeds(args.seeds)
    overrides = dict(
        communities=args.communities, avg_degree=args.avg_degree, mixing=args.mixing,
        size_cv=args.size_cv, k_neighbors=args.k_neighbors, radius=args.radius,
        density=args.density, spread_factor=args.spread_factor,
        cluster_spread=args.cluster_spread, connect=args.connect,
        component_policy=args.component_policy)

    for seed in seeds:
        g = generate(args.mode, args.n, seed, **overrides)
        run_dir = export_graph(g, args.out, tag=args.tag, write_csv=args.csv)
        print(f"[seed={seed}] {run_dir}")
        if len(seeds) == 1:
            print(json.dumps(g.report, indent=2, ensure_ascii=False))
            print("graph_sha256:", g.graph_sha256())
        else:
            r = g.report
            print(f"    nodos={r['n_nodes']} aristas={r['n_edges']} grado={r['avg_degree']} "
                  f"componentes={r['n_connected_components']} "
                  f"mixing={r.get('mixing_realized', '-')} Q={r.get('modularity_ground_truth', '-')} "
                  f"t={r['generation_time_seconds']}s")


if __name__ == "__main__":
    main()
