#!/usr/bin/env python3
"""
partition.py  (v1.0.0)  --  Paso 3: particionadores, halos y plan de comunicación.

Qué produce
  1. Una partición  part[n]  (id de worker de cada nodo) con cualquiera de:
       centralized        todo en un worker (k = 1), línea base.
       hash               hash estable del id (baseline "ciego" a la estructura).
       spatial            bisección recursiva por coordenadas (balance exacto).
       community          Louvain + empaquetado en k trozos balanceados.
       community_refined  community + refinamiento de frontera (tipo FM).
       metis              opcional, solo si pymetis está instalado.
  2. Un PLAN por worker (propios, halo de profundidad d, subgrafo local con ids
     locales, y de quién recibe cada nodo de halo). Es exactamente lo que el
     worker del paso 5 necesita, y es compatible con detect.py: un worker que
     corra el detector sobre `nodes` (propios primero, luego halo) con
     d >= DetectorConfig.halo_depth reproduce el resultado centralizado.
  3. Métricas de calidad de la partición y ESTIMACIÓN de comunicación:
     corte de aristas, balance, nodos frontera, factor de replicación,
     carga efectiva por worker (propios + halo), enlaces entre workers y bytes.

Fuga de información: ningún particionador lee `community` (ground truth del
generador). Solo se usa para reportar NMI, y un test lo verifica.

Modelo de bytes (estimación, los reales se medirán en el paso 5)
  Intercambio único de datos crudos: cada nodo de halo cuesta T * (4 + 1) bytes
  (T lecturas float32 + T bits de máscara contados como 1 byte). Como detect.py
  recalcula localmente sobre el halo, NO hace falta intercambiar scores ni
  imputaciones intermedias: un halo de profundidad d cambia comunicación por
  cómputo redundante. La topología se cuenta aparte (costo único de arranque).

Uso:
  python partition.py --graph ./data/hybrid_n10000_seed1 \
      --methods hash spatial community community_refined --k 2 4 8 \
      --depth 2 --seed 1 --T 96 --verify --out ./partitions

Como librería:
  from iot_graph_generator import generate
  from partition import make_plan
  g = generate("hybrid", 10000, seed=1, k_neighbors=8)
  plan, metrics = make_plan(g, "community", k=4, depth=2, seed=1)
  w = plan.workers[0]      # w.owned, w.halo, w.nodes, w.edges, w.recv_from
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from detect import adjacency_hops, induced_edges
from iot_graph_generator import load_graph

PARTITION_VERSION = "1.0.0"
METHODS = ("centralized", "hash", "spatial", "community", "community_refined")
BYTES_VALUE, BYTES_MASK, BYTES_ID = 4, 1, 4


# ---------------------------------------------------------------------------
# Vista del grafo y utilidades
# ---------------------------------------------------------------------------

def _gv(g) -> dict:
    """Acepta GraphData (generate) o el dict de load_graph."""
    if isinstance(g, dict):
        return dict(n=int(g["n"]), edges=g["edge_index"].T.astype(np.int64), weight=g["weight"],
                    coords=g.get("coords"), community=g.get("community"))
    return dict(n=g.n, edges=g.edges, weight=g.weight, coords=g.coords, community=g.community)


def _csr(n: int, edges: np.ndarray, weight=None) -> csr_matrix:
    w = np.ones(len(edges)) if weight is None else np.asarray(weight, dtype=np.float64)
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    return coo_matrix((np.concatenate([w, w]), (src, dst)), shape=(n, n)).tocsr()


def _modularity(n: int, edges: np.ndarray, labels: np.ndarray) -> float:
    m = len(edges)
    if m == 0:
        return 0.0
    k = int(labels.max()) + 1
    lu, lv = labels[edges[:, 0]], labels[edges[:, 1]]
    intra = lu == lv
    l_c = np.bincount(lu[intra], minlength=k)
    d_c = np.bincount(labels, weights=np.bincount(edges.ravel(), minlength=n), minlength=k)
    return float((l_c / m - (d_c / (2.0 * m)) ** 2).sum())


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    n = len(a)
    ct = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(ct, (a, b), 1)
    p = ct / n
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum()
    ha, hb = -(pa * np.log(pa)).sum(), -(pb * np.log(pb)).sum()
    return 1.0 if ha + hb == 0 else float(2 * mi / (ha + hb))


# ---------------------------------------------------------------------------
# Particionadores:  fn(gv, k, seed, **opts) -> (part int64[n], meta dict)
# ---------------------------------------------------------------------------

def partition_centralized(gv, k=1, seed=0, **_):
    return np.zeros(gv["n"], dtype=np.int64), {}


def _mix64(x: np.ndarray) -> np.ndarray:
    """Finalizador splitmix64: hash entero vectorizado, idéntico en toda plataforma."""
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def partition_hash(gv, k, seed=0, **_):
    """hash(id, sal=seed) mod k. Ignora la estructura: corte esperado (k-1)/k."""
    with np.errstate(over="ignore"):
        ids = np.arange(gv["n"], dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
    return (_mix64(ids) % np.uint64(k)).astype(np.int64), {}


def partition_spatial(gv, k, seed=0, **_):
    """Bisección recursiva por coordenadas: en cada paso corta por la mediana
    (ponderada por k1/k) del eje más largo. Balance exacto (±1 nodo)."""
    coords = gv["coords"]
    if coords is None:
        raise ValueError("'spatial' requiere coordenadas (modos hybrid/spatial del generador)")
    part = np.empty(gv["n"], dtype=np.int64)

    def rec(idx, kk, base):
        if kk == 1:
            part[idx] = base
            return
        k1 = kk // 2
        size1 = int(round(len(idx) * k1 / kk))
        axis = int(np.argmax(np.ptp(coords[idx], axis=0)))
        order = np.argpartition(coords[idx, axis], size1)
        rec(idx[order[:size1]], k1, base)
        rec(idx[order[size1:]], kk - k1, base + k1)

    rec(np.arange(gv["n"]), k, 0)
    return part, {}


def louvain_labels(gv, seed=0, resolution=1.0, weighted=False) -> np.ndarray:
    """Etiqueta de comunidad por nodo (NO usa el ground truth del generador).
    weighted=False: topología pura. True: usa la similitud 'weight' como afinidad."""
    import networkx as nx
    n, e = gv["n"], gv["edges"]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if weighted:
        G.add_weighted_edges_from(zip(e[:, 0].tolist(), e[:, 1].tolist(), gv["weight"].tolist()))
    else:
        G.add_edges_from(zip(e[:, 0].tolist(), e[:, 1].tolist()))
    comms = nx.community.louvain_communities(G, weight="weight" if weighted else None,
                                             resolution=resolution, seed=seed)
    labels = np.empty(n, dtype=np.int64)
    for i, c in enumerate(comms):
        labels[list(c)] = i
    return labels


def _bfs_within(A: csr_matrix, nodes: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Orden BFS de una comunidad desde su nodo de mayor grado. Un prefijo de este
    orden es compacto, así que si hay que partir la comunidad el corte es corto."""
    sub = A[nodes][:, nodes]
    visited = np.zeros(len(nodes), dtype=bool)
    chunks = []
    for s in np.argsort(-deg[nodes], kind="stable"):
        if visited[s]:
            continue
        reach = breadth_first_order(sub, s, directed=False, return_predecessors=False)
        reach = reach[~visited[reach]]
        visited[reach] = True
        chunks.append(reach)
    return nodes[np.concatenate(chunks)]


def _community_sequence(A: csr_matrix, labels: np.ndarray, deg: np.ndarray) -> list:
    """Comunidades ordenadas para que las adyacentes queden contiguas (recorrido
    voraz sobre el grafo cociente) y cada una internamente en orden BFS."""
    n, c = len(labels), int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=c)
    P = coo_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, c)).tocsr()
    Q = (P.T @ A @ P).tocsr()
    members = [[] for _ in range(c)]
    for v in np.argsort(labels, kind="stable"):
        members[labels[v]].append(v)
    remaining = np.ones(c, dtype=bool)
    cur = int(np.argmax(sizes))
    order = []
    while True:
        order.append(cur)
        remaining[cur] = False
        if not remaining.any():
            break
        row = Q.getrow(cur).toarray().ravel()
        cur = int(np.argmax(np.where(remaining, row * (n + 1) + sizes, -1)))
    return [_bfs_within(A, np.asarray(members[i], dtype=np.int64), deg) for i in order]


def _snap_cuts(sizes: np.ndarray, n: int, k: int, tol: float) -> np.ndarray:
    """Posiciones de corte en la secuencia de nodos. Cada corte se ajusta al límite de
    comunidad más cercano al punto ideal si queda dentro de la tolerancia; si no hay
    ninguno, se corta exacto (partiendo una comunidad). El objetivo se recalcula con lo
    que queda (n_restante / k_restante), así el error no se acumula."""
    ends = np.cumsum(sizes)
    cuts, prev = [], 0
    for j in range(1, k):
        left = k - j + 1
        avg = (n - prev) / left
        target = prev + avg
        lo = max(prev + int(math.floor(avg * (1 - tol))), prev + 1)
        hi = min(prev + int(math.ceil(avg * (1 + tol))), n - (k - j))
        i = int(np.searchsorted(ends, target))
        cand = [int(ends[t]) for t in (i - 1, i) if 0 <= t < len(ends) and lo <= ends[t] <= hi]
        c = min(cand, key=lambda b: abs(b - target)) if cand else int(round(min(max(target, lo), hi)))
        cuts.append(c)
        prev = c
    return np.asarray(cuts, dtype=np.int64)


def partition_community(gv, k, seed=0, tol=0.05, resolution=1.0, weighted=False, labels=None, **_):
    """Louvain + empaquetado balanceado. Pasa `labels` para reutilizar un Louvain ya calculado
    (su costo domina; conviene calcularlo una vez por grafo y semilla)."""
    t0 = time.perf_counter()
    if labels is None:
        labels = louvain_labels(gv, seed, resolution, weighted)
    t_louvain = time.perf_counter() - t0
    n, e = gv["n"], gv["edges"]
    A = _csr(n, e)
    deg = np.bincount(e.ravel(), minlength=n)
    seq = _community_sequence(A, labels, deg)
    sizes = np.array([len(s) for s in seq])
    order = np.concatenate(seq)
    cuts = _snap_cuts(sizes, n, k, tol)
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    part = np.searchsorted(cuts, pos, side="right").astype(np.int64)
    meta = dict(louvain_n_communities=int(labels.max()) + 1,
                louvain_modularity=round(_modularity(n, e, labels), 4),
                louvain_time_s=round(t_louvain, 3),
                cuts_on_community_boundary=int(np.isin(cuts, np.cumsum(sizes)).sum()),
                n_cuts=int(len(cuts)))
    if gv.get("community") is not None:           # SOLO evaluación, nunca para particionar
        meta["louvain_nmi_vs_ground_truth"] = round(_nmi(gv["community"], labels), 4)
    return part, meta


def refine_boundary(gv, part, k, tol=0.05, max_passes=10, seed=0):
    """Refinamiento de frontera tipo Fiduccia-Mattheyses: solo nodos de borde, cada uno
    pasa a la partición donde tiene más vecinos si la ganancia es > 0 y el balance se
    mantiene dentro de la tolerancia. Cada movimiento baja el corte al menos en 1, así que
    termina siempre y NUNCA aumenta el corte."""
    part = part.copy()
    n, e = gv["n"], gv["edges"]
    A = _csr(n, e)
    ip, ix = A.indptr, A.indices
    avg = n / k
    hi, lo = math.ceil(avg * (1 + tol)), math.floor(avg * (1 - tol))
    sizes = np.bincount(part, minlength=k)
    rng = np.random.default_rng(seed)
    cut0 = int((part[e[:, 0]] != part[e[:, 1]]).sum())
    moves = []
    for _ in range(max_passes):
        cross = part[e[:, 0]] != part[e[:, 1]]
        cand = np.unique(np.concatenate([e[cross, 0], e[cross, 1]]))
        rng.shuffle(cand)
        moved = 0
        for v in cand:
            p = part[v]
            counts = np.bincount(part[ix[ip[v]:ip[v + 1]]], minlength=k)
            internal = counts[p]
            counts[p] = -1
            q = int(np.argmax(counts))
            if counts[q] - internal > 0 and sizes[q] < hi and sizes[p] > lo:
                part[v] = q
                sizes[p] -= 1
                sizes[q] += 1
                moved += 1
        moves.append(moved)
        if moved == 0:
            break
    return part, dict(refine_passes=len(moves), refine_moves_total=int(sum(moves)),
                      edge_cut_before_refine=cut0)


def partition_community_refined(gv, k, seed=0, tol=0.05, max_passes=10, **kw):
    part, meta = partition_community(gv, k, seed=seed, tol=tol, **kw)
    part, rmeta = refine_boundary(gv, part, k, tol=tol, max_passes=max_passes, seed=seed)
    return part, {**meta, **rmeta}


def partition_metis(gv, k, seed=0, **_):
    """Opcional: requiere `pip install pymetis`. NO verificado en el entorno de desarrollo
    (sin red para instalarlo); revisa el resultado con verify_plan y las métricas."""
    try:
        import pymetis
    except ImportError as exc:
        raise ImportError("metis requiere pymetis (pip install pymetis)") from exc
    A = _csr(gv["n"], gv["edges"])
    adj = [A.indices[A.indptr[i]:A.indptr[i + 1]].tolist() for i in range(gv["n"])]
    try:
        _, parts = pymetis.part_graph(k, adjacency=adj, options=pymetis.Options(seed=seed))
    except TypeError:
        _, parts = pymetis.part_graph(k, adjacency=adj)
    return np.asarray(parts, dtype=np.int64), {}


PARTITIONERS = {"centralized": partition_centralized, "hash": partition_hash,
                "spatial": partition_spatial, "community": partition_community,
                "community_refined": partition_community_refined, "metis": partition_metis}


def partition_graph(graph, method: str, k: int, seed: int = 0, **opts):
    """Devuelve (part, meta). Para 'centralized' fuerza k = 1."""
    if method not in PARTITIONERS:
        raise ValueError(f"método desconocido: {method}. Opciones: {sorted(PARTITIONERS)}")
    gv = _gv(graph)
    if method == "centralized":
        k = 1
    if not 1 <= k <= gv["n"]:
        raise ValueError(f"k={k} inválido para n={gv['n']}")
    part, meta = PARTITIONERS[method](gv, k, seed=seed, **opts)
    assert part.shape == (gv["n"],) and part.min() >= 0 and part.max() < k
    return part, meta


# ---------------------------------------------------------------------------
# Plan de workers (propios + halo + plan de comunicación)
# ---------------------------------------------------------------------------

@dataclass
class WorkerPlan:
    rank: int
    owned: np.ndarray              # ids globales de sus nodos propios
    halo: np.ndarray               # ids globales de nodos fantasma (profundidad <= depth)
    nodes: np.ndarray              # concat(owned, halo): el orden define los ids locales
    edges: np.ndarray              # (m_local, 2) aristas inducidas, ids LOCALES
    recv_from: dict = field(default_factory=dict)   # rank origen -> ids globales a recibir

    @property
    def n_owned(self) -> int:
        return len(self.owned)

    @property
    def n_local(self) -> int:
        return len(self.nodes)


@dataclass
class PartitionPlan:
    method: str
    k: int
    depth: int
    n: int
    part: np.ndarray
    workers: list

    def send_to(self, src: int, dst: int) -> np.ndarray:
        """Ids globales que `src` debe enviar a `dst` (los recibe dst desde src)."""
        return self.workers[dst].recv_from.get(src, np.empty(0, dtype=np.int64))


def build_plan(graph, part: np.ndarray, k: int, depth: int, method: str = "") -> PartitionPlan:
    """Halo EXACTO de profundidad `depth`: nodos no propios a <= depth saltos de algún
    nodo propio. Ni más (sería comunicación de sobra) ni menos (rompe la equivalencia)."""
    gv = _gv(graph)
    n, e = gv["n"], gv["edges"]
    Ad = adjacency_hops(n, e, depth) if depth >= 1 else None
    workers = []
    for q in range(k):
        own_mask = part == q
        owned = np.nonzero(own_mask)[0]
        if Ad is None or len(owned) == 0:
            halo = np.empty(0, dtype=np.int64)
        else:
            reach = np.asarray(Ad[owned].sum(axis=0)).ravel() > 0
            halo = np.nonzero(reach & ~own_mask)[0]
        nodes = np.concatenate([owned, halo])
        recv = {}
        if len(halo):
            owners = part[halo]
            for p in np.unique(owners):
                recv[int(p)] = halo[owners == p]
        workers.append(WorkerPlan(q, owned, halo, nodes, induced_edges(n, e, nodes), recv))
    return PartitionPlan(method, k, depth, n, part, workers)


def verify_plan(plan: PartitionPlan, graph) -> None:
    """Invariantes del plan. Lanza AssertionError con el motivo si alguno falla."""
    gv = _gv(graph)
    n, e = gv["n"], gv["edges"]
    allown = np.sort(np.concatenate([w.owned for w in plan.workers]))
    assert np.array_equal(allown, np.arange(n)), "cada nodo debe tener exactamente un dueño"
    Ad = adjacency_hops(n, e, plan.depth) if plan.depth >= 1 else None
    for w in plan.workers:
        assert len(np.intersect1d(w.owned, w.halo)) == 0, f"worker {w.rank}: halo ∩ propios ≠ ∅"
        assert len(np.unique(w.nodes)) == len(w.nodes), f"worker {w.rank}: nodos locales repetidos"
        local = np.zeros(n, dtype=bool)
        local[w.nodes] = True
        need = np.zeros(n, dtype=bool)
        need[w.owned] = True
        if Ad is not None and len(w.owned):
            need |= np.asarray(Ad[w.owned].sum(axis=0)).ravel() > 0
        assert not (need & ~local).any(), f"worker {w.rank}: halo INCOMPLETO"
        assert not (local & ~need).any(), f"worker {w.rank}: halo con nodos innecesarios"
        got = sum(len(v) for v in w.recv_from.values())
        assert got == len(w.halo), f"worker {w.rank}: recv_from no suma el halo"
        for src, ids in w.recv_from.items():
            assert (plan.part[ids] == src).all(), f"worker {w.rank}: dueño erróneo para recv {src}"
        expected = int((local[e[:, 0]] & local[e[:, 1]]).sum())
        assert len(w.edges) == expected, f"worker {w.rank}: subgrafo local incorrecto"
        assert w.edges.size == 0 or (w.edges.min() >= 0 and w.edges.max() < w.n_local)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def boundary_nodes(graph, part: np.ndarray, min_cut_fraction: float = 0.0) -> np.ndarray:
    """Ids de los nodos de frontera: los que tienen al menos un vecino en otra partición.
    min_cut_fraction > 0 conserva solo los nodos cuya fracción de vecinos en OTRA partición es
    >= ese valor (p. ej. 0.5 = pierden la mitad o más de sus vecinos si no hay halo): el
    adversario más fuerte para la prueba E4b.
    Uso: generate_signals(..., anomaly_nodes=boundary_nodes(g, part, 0.5))."""
    gv = _gv(graph)
    e, n = gv["edges"], gv["n"]
    cross = part[e[:, 0]] != part[e[:, 1]]
    cut_deg = np.bincount(e[cross].ravel(), minlength=n)
    deg = np.bincount(e.ravel(), minlength=n)
    frac = cut_deg / np.maximum(deg, 1)
    return np.nonzero((cut_deg > 0) & (frac >= min_cut_fraction))[0]


def plan_metrics(plan: PartitionPlan, graph, T: int = 96) -> dict:
    gv = _gv(graph)
    n, e, w = gv["n"], gv["edges"], gv["weight"]
    k, part = plan.k, plan.part
    m = len(e)
    sizes = np.array([x.n_owned for x in plan.workers])
    cross = part[e[:, 0]] != part[e[:, 1]]
    boundary = np.zeros(n, dtype=bool)
    boundary[e[cross].ravel()] = True
    halo = np.array([len(x.halo) for x in plan.workers])
    local = np.array([x.n_local for x in plan.workers])
    vol = np.zeros((k, k), dtype=np.int64)               # vol[src, dst] en nodos
    for x in plan.workers:
        for src, ids in x.recv_from.items():
            vol[src, x.rank] = len(ids)
    per_node = T * (BYTES_VALUE + BYTES_MASK)
    recv_nodes = vol.sum(axis=0)
    peers = (vol > 0).sum(axis=0)
    return dict(
        method=plan.method, k=k, depth=plan.depth, n_nodes=n, n_edges=m, T=T,
        sizes=sizes.tolist(),
        imbalance=round(float(sizes.max() / (n / k)), 4),
        size_cv=round(float(sizes.std() / (n / k)), 4),
        edge_cut=int(cross.sum()),
        cut_ratio=round(float(cross.mean()), 4) if m else 0.0,
        weighted_cut_ratio=round(float(w[cross].sum() / w.sum()), 4) if m else 0.0,
        boundary_fraction=round(float(boundary.mean()), 4),
        halo_sizes=halo.tolist(), halo_total=int(halo.sum()),
        replication_factor=round(float(local.sum() / n), 4),
        effective_imbalance=round(float(local.max() / (n / k)), 4),   # carga = propios + halo
        peers_avg=round(float(peers.mean()), 3), peers_max=int(peers.max()),
        comm_links=int((vol > 0).sum()),
        halo_bytes_est=int(halo.sum() * per_node),
        halo_bytes_max_worker_est=int(recv_nodes.max() * per_node),
        topology_bytes_est=int(sum(len(x.edges) for x in plan.workers) * 2 * BYTES_ID),
        partition_modularity=round(_modularity(n, e, part), 4),
    )


def make_plan(graph, method: str, k: int, depth: int, seed: int = 0, T: int = 96, **opts):
    """Partición + plan + métricas en un paso. Devuelve (plan, metrics)."""
    t0 = time.perf_counter()
    part, meta = partition_graph(graph, method, k, seed=seed, **opts)
    t_part = time.perf_counter() - t0
    kk = 1 if method == "centralized" else k
    t0 = time.perf_counter()
    plan = build_plan(graph, part, kk, depth, method)
    t_plan = time.perf_counter() - t0
    met = plan_metrics(plan, graph, T)
    met.update(meta)
    met.update(seed=seed, partition_time_s=round(t_part, 4), plan_time_s=round(t_plan, 4))
    return plan, met


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def save_partition_csv(part: np.ndarray, path) -> None:
    """CSV id,partition (interoperable con otras herramientas del equipo)."""
    np.savetxt(path, np.column_stack([np.arange(len(part)), part]), delimiter=",",
               header="id,partition", comments="", fmt="%d")


def main() -> None:
    ap = argparse.ArgumentParser(description="Particionadores, halos y plan de comunicación (T800)")
    ap.add_argument("--graph", required=True, help="Carpeta de corrida del generador")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=sorted(PARTITIONERS))
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--depth", type=int, default=1, help="Profundidad del halo (saltos)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--T", type=int, default=96, help="Pasos de tiempo para estimar bytes")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--weighted", action="store_true", help="Louvain usando la similitud como peso")
    ap.add_argument("--verify", action="store_true", help="Comprobar invariantes de cada plan")
    ap.add_argument("--csv", action="store_true", help="Guardar también id,partition en CSV")
    ap.add_argument("--out", default="./partitions")
    a = ap.parse_args()

    g = load_graph(a.graph)
    gv = _gv(g)
    out = Path(a.out) / Path(a.graph).resolve().name
    out.mkdir(parents=True, exist_ok=True)

    labels = None
    if any(m in ("community", "community_refined") for m in a.methods):
        t0 = time.perf_counter()
        labels = louvain_labels(gv, a.seed, weighted=a.weighted)
        print(f"Louvain: {labels.max() + 1} comunidades en {time.perf_counter() - t0:.1f}s "
              f"(se reutiliza para todos los k)\n")

    rows = []
    print(f"{'método':18s} {'k':>3s} {'corte':>6s} {'desbal':>7s} {'frontera':>8s} "
          f"{'replic':>7s} {'desbal_ef':>9s} {'MB halo':>8s} {'t_part':>7s}")
    for method in a.methods:
        for k in ([1] if method == "centralized" else a.k):
            opts = dict(tol=a.tol, labels=labels) if "community" in method else {}
            plan, m = make_plan(g, method, k, a.depth, a.seed, a.T, **opts)
            if a.verify:
                verify_plan(plan, g)
            np.save(out / f"part_{method}_k{k}_seed{a.seed}.npy", plan.part)
            if a.csv:
                save_partition_csv(plan.part, out / f"part_{method}_k{k}_seed{a.seed}.csv")
            rows.append(m)
            print(f"{method:18s} {k:3d} {m['cut_ratio']:6.3f} {m['imbalance']:7.3f} "
                  f"{m['boundary_fraction']:8.3f} {m['replication_factor']:7.2f} "
                  f"{m['effective_imbalance']:9.2f} {m['halo_bytes_est'] / 1e6:8.1f} "
                  f"{m['partition_time_s']:7.2f}")
    with open(out / f"metrics_seed{a.seed}_depth{a.depth}.json", "w", encoding="utf-8") as f:
        json.dump(dict(version=PARTITION_VERSION, rows=rows), f, indent=2, default=str)
    print(f"\nResultados en: {out}" + ("  (planes verificados)" if a.verify else ""))


if __name__ == "__main__":
    main()
