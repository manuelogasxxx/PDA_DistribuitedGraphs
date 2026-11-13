"""Tests del generador. Ejecutar:  pytest -q test_iot_graph_generator.py"""

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from iot_graph_generator import export_graph, generate, load_graph, to_csr

N = 3000


def _is_canonical(g):
    e = g.edges
    assert (e[:, 0] < e[:, 1]).all(), "aristas deben ser src < dst (sin auto-lazos)"
    keys = e[:, 0] * g.n + e[:, 1]
    assert len(np.unique(keys)) == len(keys), "aristas duplicadas"
    assert e.min() >= 0 and e.max() < g.n


@pytest.mark.parametrize("mode", ["community", "spatial", "powerlaw", "hybrid"])
def test_determinism_and_structure(mode):
    a = generate(mode, N, seed=5)
    b = generate(mode, N, seed=5)
    c = generate(mode, N, seed=6)
    assert a.graph_sha256() == b.graph_sha256()
    assert a.graph_sha256() != c.graph_sha256()
    _is_canonical(a)
    assert a.report["n_connected_components"] == 1
    assert a.report["isolated_nodes"] == 0


@pytest.mark.parametrize("mode", ["community", "hybrid"])
def test_no_id_leakage(mode):
    """Con IDs permutados, la comunidad no debe ser predecible por rango de ID."""
    g = generate(mode, N, seed=1)
    k = g.report["n_communities"]
    by_range = (np.arange(g.n) * k // g.n)  # particionador "por rangos de ID"
    # Pureza de la partición por rangos respecto al ground truth
    purity = sum(np.bincount(g.community[by_range == r], minlength=k).max()
                 for r in range(k)) / g.n
    assert purity < 0.25, f"los IDs filtran la comunidad (pureza={purity:.2f})"


def test_sbm_degree_and_mixing_calibration():
    for mu in (0.05, 0.2, 0.4):
        g = generate("community", 10000, seed=3, mixing=mu, avg_degree=10.0)
        assert abs(g.report["avg_degree"] - 10.0) < 0.5
        assert abs(g.report["mixing_realized"] - mu) < 0.03


def test_mixing_is_monotone_in_spread_factor():
    mix = [generate("hybrid", 5000, seed=2, spread_factor=s).report["mixing_realized"]
           for s in (0.1, 0.25, 0.5)]
    assert mix[0] < mix[1] < mix[2]


def test_hybrid_edges_are_spatially_local():
    """Ninguna arista no-puente debe ser 'cross-plano' (v1 sí lo permitía)."""
    g = generate("hybrid", N, seed=4)
    d = g.distance[~g.is_bridge]
    assert d.max() < 15 * np.median(d)


def test_weight_is_similarity_not_distance():
    g = generate("hybrid", N, seed=4)
    order = np.argsort(g.distance)
    w_sorted = g.weight[order]
    assert (np.diff(w_sorted) <= 1e-12).all(), "el peso debe decrecer al aumentar la distancia"
    assert 0.0 < g.weight.min() and g.weight.max() <= 1.0


def test_connect_flag_and_bridges():
    g = generate("hybrid", N, seed=9, spread_factor=0.08, connect=False,
                 component_policy="keep")
    h = generate("hybrid", N, seed=9, spread_factor=0.08, connect=True)
    assert g.report["n_connected_components"] > 1
    assert h.report["n_connected_components"] == 1
    assert h.report["bridge_edges"] >= 1


def test_independent_streams_graph_vs_permutation():
    """Cambiar un parámetro de aristas no debe alterar tamaños de comunidad."""
    # component_policy="keep": 'giant' podría descartar nodos distintos en cada grafo
    a = generate("community", N, seed=11, mixing=0.1, component_policy="keep")
    b = generate("community", N, seed=11, mixing=0.3, component_policy="keep")
    assert (np.bincount(a.community) == np.bincount(b.community)).all()


def test_roundtrip_export_load(tmp_path):
    g = generate("hybrid", N, seed=12)
    run_dir = export_graph(g, tmp_path, write_csv=True)
    d = load_graph(run_dir)
    assert d["n"] == g.n
    assert np.array_equal(d["edge_index"], g.edge_index())
    A = to_csr(d, weighted=True)
    assert (A != A.T).nnz == 0
    assert connected_components(A, directed=False)[0] == 1
    assert d["manifest"]["graph_sha256"] == g.graph_sha256()
    assert (run_dir / "nodes.csv").exists() and (run_dir / "edges.csv").exists()


def test_invalid_params():
    with pytest.raises(ValueError):
        generate("community", 200, seed=1, communities=100)
    with pytest.raises(ValueError):
        generate("community", 1000, seed=1, avg_degree=500.0, communities=40)
    with pytest.raises(TypeError):
        generate("hybrid", 1000, seed=1, parametro_inexistente=3)
