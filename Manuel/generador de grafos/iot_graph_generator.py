"""
iot_graph_generator.py

Generador de grafos sintéticos reproducibles para el proyecto:
"Analítica distribuida basada en grafos para detección de anomalías
y calidad de datos en redes IoT" (Equipo T800).

Produce grafos con:
  - Estructura de comunidades REAL (ground truth), vía Stochastic Block Model,
    para poder validar si un particionador (Louvain/METIS) realmente la
    recupera antes de medir F1 de anomalías río abajo.
  - Coordenadas espaciales coherentes con la comunidad (modo 'hybrid'),
    para habilitar el baseline de partición espacial/geohash.
  - Distribución de grado opcionalmente heterogénea (ley de potencia),
    para dar hubs reales a estrategias conscientes de grado (DBH/HDRF).
  - manifest.json con TODOS los parámetros + semilla usados, requisito de
    la Fase 4 del protocolo experimental ("congelamiento de versiones...
    semillas aleatorias").
  - Reporte de validación: conectividad, grado, y modularidad de la
    comunidad plantada (chequeo de sanidad antes de particionar).

Todas las generaciones evitan enumerar pares de nodos O(n^2): el SBM usa
nx.stochastic_block_model(sparse=True), y el modo espacial usa un cKDTree
(O(n log n)). Esto es necesario para el escenario de 100k nodos del
protocolo experimental -- enumerar todos los pares a esa escala es inviable.

Modos disponibles:
    community  -> Stochastic Block Model puro (baseline de partición por comunidad)
    spatial    -> coordenadas + k-NN o radio fijo (baseline de partición espacial/geohash)
    powerlaw   -> Barabási-Albert (hubs reales, para DBH/HDRF)
    hybrid     -> SBM + coordenadas coherentes con la comunidad (RECOMENDADO
                  para este proyecto: así describe la hipótesis "comunidad/vecindad")

Ejemplos (las tres escalas que pide el protocolo experimental):
    python iot_graph_generator.py --mode hybrid --n 1000    --seed 42 --out ./data
    python iot_graph_generator.py --mode hybrid --n 10000   --seed 42 --out ./data
    python iot_graph_generator.py --mode hybrid --n 100000  --seed 42 --out ./data

    # Para comparar estrategias de partición manteniendo todo lo demás fijo,
    # repite solo cambiando la semilla (Fase 4: 20-30 repeticiones):
    python iot_graph_generator.py --mode hybrid --n 10000 --seed 1 --out ./data
    python iot_graph_generator.py --mode hybrid --n 10000 --seed 2 --out ./data
    ...
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Conversión grado-objetivo -> probabilidades del SBM
# ---------------------------------------------------------------------------

def _degree_to_sbm_probs(n, k, avg_degree, community_ratio):
    """Convierte un grado promedio OBJETIVO (realista para una red IoT: 5-20)
    en las probabilidades p_intra / p_inter del SBM.

    Sin esto, exponer p_intra/p_inter directamente produce densidades muy
    distintas al cambiar n o el número de comunidades, lo cual rompe la
    comparabilidad entre los escenarios de 1k/10k/100k nodos que pide el
    protocolo experimental.

    community_ratio: cuántas veces más probable es una arista dentro de la
    misma comunidad que entre comunidades distintas (valores típicos: 15-50;
    más alto = comunidades más "fáciles" de detectar por el particionador).
    """
    denom = n * (community_ratio / k + 1)
    p_inter = avg_degree / denom
    p_intra = community_ratio * p_inter
    return min(p_intra, 1.0), min(p_inter, 1.0)


def _balanced_sizes(n, n_communities, seed):
    """Tamaños de comunidad balanceados con una pequeña variación aleatoria
    (más realista que comunidades idénticas en tamaño)."""
    rng = np.random.default_rng(seed)
    base = n // n_communities
    sizes = [base] * n_communities
    remainder = n - base * n_communities
    if remainder:
        for i in rng.choice(n_communities, size=remainder, replace=False):
            sizes[i] += 1
    return sizes


# ---------------------------------------------------------------------------
# Modos de generación
# ---------------------------------------------------------------------------

def generate_community_graph(n, n_communities, avg_degree, community_ratio, seed):
    """Modo 'community': Stochastic Block Model puro.
    Ground truth de comunidad queda en el atributo de nodo 'community'.
    """
    sizes = _balanced_sizes(n, n_communities, seed)
    p_intra, p_inter = _degree_to_sbm_probs(n, n_communities, avg_degree, community_ratio)
    probs = [[p_inter] * n_communities for _ in range(n_communities)]
    for i in range(n_communities):
        probs[i][i] = p_intra

    G = nx.stochastic_block_model(sizes, probs, seed=seed, directed=False, sparse=True)
    for node in G.nodes():
        G.nodes[node]["community"] = G.nodes[node].pop("block")
    nx.set_edge_attributes(G, 1.0, "weight")
    return G


def generate_spatial_graph(n, seed, area=1000.0, k_neighbors=6, radius=None):
    """Modo 'spatial': coordenadas uniformes en [0, area]^2, aristas por
    k-NN (o por radio fijo si se especifica 'radius'). Usa cKDTree
    (O(n log n)), nunca compara todos los pares.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, area, size=(n, 2))
    tree = cKDTree(coords)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, (x, y) in enumerate(coords):
        G.nodes[i]["x"] = float(x)
        G.nodes[i]["y"] = float(y)

    if radius is not None:
        pairs = tree.query_pairs(r=radius)
    else:
        k = min(k_neighbors + 1, n)
        _, idx = tree.query(coords, k=k)
        pairs = set()
        for i, neighbors in enumerate(np.atleast_2d(idx)):
            for j in neighbors[1:]:
                pairs.add((min(i, int(j)), max(i, int(j))))

    for i, j in pairs:
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        G.add_edge(i, j, weight=dist)

    return G


def generate_powerlaw_graph(n, seed, avg_degree=10):
    """Modo 'powerlaw': Barabási-Albert. Produce hubs reales (grado alto en
    pocos nodos) para que estrategias conscientes de grado (DBH/HDRF) tengan
    algo que optimizar frente al hash simple.
    """
    m = max(1, round(avg_degree / 2))
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    nx.set_edge_attributes(G, 1.0, "weight")
    return G


def generate_hybrid_graph(n, n_communities, avg_degree, community_ratio, seed,
                           area=1000.0, cluster_spread=40.0):
    """Modo 'hybrid' (RECOMENDADO para este proyecto): comunidades vía SBM +
    coordenadas espacialmente coherentes con cada comunidad (cada comunidad
    ocupa una 'zona' del plano). Es lo más fiel a una red IoT real -- sensores
    de una misma comunidad/vecindad suelen estar físicamente cerca -- y es
    el escenario que la hipótesis del proyecto necesita para comparar
    partición por comunidad vs. espacial vs. hash sobre el MISMO grafo base.
    """
    rng = np.random.default_rng(seed)
    G = generate_community_graph(n, n_communities, avg_degree, community_ratio, seed)

    centroids = rng.uniform(0, area, size=(n_communities, 2))
    for node in G.nodes():
        c = G.nodes[node]["community"]
        cx, cy = centroids[c]
        x = float(np.clip(rng.normal(cx, cluster_spread), 0, area))
        y = float(np.clip(rng.normal(cy, cluster_spread), 0, area))
        G.nodes[node]["x"] = x
        G.nodes[node]["y"] = y

    for u, v in G.edges():
        p1 = np.array([G.nodes[u]["x"], G.nodes[u]["y"]])
        p2 = np.array([G.nodes[v]["x"], G.nodes[v]["y"]])
        G.edges[u, v]["weight"] = float(np.linalg.norm(p1 - p2))

    return G


# ---------------------------------------------------------------------------
# Validación (chequeo de sanidad antes de particionar)
# ---------------------------------------------------------------------------

def validate_graph(G, has_communities=False):
    n = G.number_of_nodes()
    degrees = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))
    giant = max(components, key=len) if components else set()

    report = {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "avg_degree": round(sum(degrees) / n, 3) if n else 0,
        "max_degree": max(degrees) if degrees else 0,
        "n_connected_components": len(components),
        "giant_component_fraction": round(len(giant) / n, 4) if n else 0,
        "isolated_nodes": sum(1 for d in degrees if d == 0),
    }

    if has_communities:
        comms = {}
        for node, data in G.nodes(data=True):
            comms.setdefault(data["community"], set()).add(node)
        partition = list(comms.values())
        report["n_communities"] = len(partition)
        # weight=None a propósito: queremos la modularidad TOPOLÓGICA de la
        # comunidad plantada, independiente de qué signifique 'weight' en
        # este modo (uniforme en 'community', distancia espacial en 'hybrid').
        # Si se usa el peso por defecto, 'hybrid' da una modularidad distinta
        # a 'community' aunque la topología/comunidades sean idénticas.
        report["modularity_ground_truth"] = round(
            nx.algorithms.community.quality.modularity(G, partition, weight=None), 4
        )

    return report


# ---------------------------------------------------------------------------
# Exportación: nodes.csv + edges.csv + manifest.json
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    if not rows:
        Path(path).touch()
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def export_graph(G, out_dir, run_name, params, report):
    """Formato liviano (CSV), más rápido y más chico que GraphML a 100k
    nodos, y fácil de cargar tanto en NetworkX (nx.from_pandas_edgelist)
    como en PyTorch Geometric (edge_index desde edges[['source','target']]).
    """
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    node_rows = [{"id": node, **data} for node, data in G.nodes(data=True)]
    _write_csv(run_dir / "nodes.csv", node_rows)

    edge_rows = [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
    _write_csv(run_dir / "edges.csv", edge_rows)

    manifest = {"params": params, "validation": report}
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generador de grafos sintéticos IoT reproducibles (Equipo T800)"
    )
    parser.add_argument("--mode", choices=["community", "spatial", "powerlaw", "hybrid"],
                         required=True)
    parser.add_argument("--n", type=int, required=True,
                         help="Número de nodos (escenarios del protocolo: 1000, 10000, 100000)")
    parser.add_argument("--seed", type=int, required=True,
                         help="Semilla, OBLIGATORIA para reproducibilidad (Fase 4)")
    parser.add_argument("--out", type=str, default="./data", help="Directorio de salida")

    parser.add_argument("--communities", type=int, default=10,
                         help="Número de comunidades (modos community/hybrid)")
    parser.add_argument("--avg-degree", type=float, default=10.0,
                         help="Grado promedio objetivo (realista para IoT: 5-20)")
    parser.add_argument("--community-ratio", type=float, default=25.0,
                         help="Cuántas veces más probable una arista intra- vs. inter-comunidad")

    parser.add_argument("--area", type=float, default=1000.0, help="Lado del plano espacial")
    parser.add_argument("--k-neighbors", type=int, default=6, help="k-NN (modo spatial)")
    parser.add_argument("--radius", type=float, default=None,
                         help="Radio fijo en vez de k-NN (modo spatial)")
    parser.add_argument("--cluster-spread", type=float, default=40.0,
                         help="Dispersión espacial dentro de cada comunidad (modo hybrid)")

    return parser


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)

    has_communities = args.mode in ("community", "hybrid")
    t0 = time.perf_counter()

    if args.mode == "community":
        G = generate_community_graph(args.n, args.communities, args.avg_degree,
                                      args.community_ratio, args.seed)
    elif args.mode == "spatial":
        G = generate_spatial_graph(args.n, args.seed, args.area, args.k_neighbors, args.radius)
    elif args.mode == "powerlaw":
        G = generate_powerlaw_graph(args.n, args.seed, args.avg_degree)
    else:  # hybrid
        G = generate_hybrid_graph(args.n, args.communities, args.avg_degree,
                                   args.community_ratio, args.seed, args.area, args.cluster_spread)

    elapsed = time.perf_counter() - t0

    report = validate_graph(G, has_communities=has_communities)
    report["generation_time_seconds"] = round(elapsed, 3)

    run_name = f"{args.mode}_n{args.n}_seed{args.seed}"
    out_dir = export_graph(G, args.out, run_name, vars(args), report)

    print(f"Grafo generado en: {out_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()