"""Tests de partition.py. Ejecutar:  pytest -q test_partition.py"""

import dataclasses

import numpy as np
import pytest

from detect import (DetectorConfig, adjacency_hops, impute, run_centralized, spatial_scores,
                    temporal_scores)
from iot_graph_generator import export_graph, generate, load_graph
from partition import (PartitionPlan, _gv, _snap_cuts, boundary_nodes, build_plan, make_plan, partition_graph,
                       partition_hash, plan_metrics, refine_boundary, verify_plan)
from signals import generate_signals

N = 2000
METHODS = ["hash", "spatial", "community", "community_refined"]


@pytest.fixture(scope="module")
def graph():
    return generate("hybrid", N, seed=3, k_neighbors=8)


@pytest.fixture(scope="module")
def sig(graph):
    return generate_signals(graph, seed=3, anomaly="both", missing="structured")


# ----------------------------- particiones válidas -----------------------------

@pytest.mark.parametrize("method", METHODS)
def test_valid_and_deterministic(graph, method):
    for k in (2, 4, 8):
        a, _ = partition_graph(graph, method, k, seed=5)
        b, _ = partition_graph(graph, method, k, seed=5)
        assert a.shape == (N,) and a.min() == 0 and a.max() == k - 1
        assert len(np.unique(a)) == k, "ninguna partición debe quedar vacía"
        assert np.array_equal(a, b)


def test_hash_regression_values():
    """Fija el hash: debe ser idéntico entre máquinas y versiones de numpy."""
    class G:
        n, edges, weight, coords, community = 12, np.zeros((0, 2), int), np.zeros(0), None, None
    assert partition_hash(_gv(G), 4, seed=0)[0].tolist() == [0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 1, 1]
    assert partition_hash(_gv(G), 4, seed=7)[0].tolist() == [1, 1, 2, 0, 2, 1, 0, 2, 3, 0, 1, 0]


def test_balance(graph):
    for k in (2, 4, 8):
        n_k = N / k
        sp = np.bincount(partition_graph(graph, "spatial", k)[0], minlength=k)
        assert sp.max() - sp.min() <= 1, "spatial debe ser exacta (±1 nodo)"
        for m in ("community", "community_refined"):
            sizes = np.bincount(partition_graph(graph, m, k, seed=1, tol=0.05)[0], minlength=k)
            assert sizes.max() <= np.ceil(n_k * 1.05) + 1, (m, k, sizes)
            assert sizes.min() >= np.floor(n_k * 0.95) - 1, (m, k, sizes)
        hs = np.bincount(partition_graph(graph, "hash", k, seed=1)[0], minlength=k)
        assert hs.max() / n_k < 1.15


def test_hash_cut_matches_theory(graph):
    for k in (2, 4, 8):
        _, m = make_plan(graph, "hash", k, 1, seed=1)
        assert abs(m["cut_ratio"] - (k - 1) / k) < 0.03


def test_structure_aware_beats_hash(graph):
    for k in (2, 4, 8):
        h = make_plan(graph, "hash", k, 1, seed=1)[1]["cut_ratio"]
        for method in ("spatial", "community", "community_refined"):
            c = make_plan(graph, method, k, 1, seed=1)[1]["cut_ratio"]
            assert c < 0.1 * h, (method, k, c, h)


def test_refinement_never_worsens_cut(graph):
    for k in (4, 8):
        base, _ = partition_graph(graph, "community", k, seed=1)
        ref, meta = refine_boundary(_gv(graph), base, k, tol=0.05, seed=1)
        e = graph.edges
        cut = lambda p: int((p[e[:, 0]] != p[e[:, 1]]).sum())
        assert cut(ref) <= cut(base)
        assert meta["edge_cut_before_refine"] == cut(base)
        sizes = np.bincount(ref, minlength=k)
        assert sizes.max() <= np.ceil(N / k * 1.05) + 1


def test_no_ground_truth_leakage(graph):
    """Quitar el atributo 'community' no puede cambiar la partición."""
    blind = dataclasses.replace(graph, community=None)
    for method in ("community", "community_refined"):
        a, meta_a = partition_graph(graph, method, 4, seed=2)
        b, meta_b = partition_graph(blind, method, 4, seed=2)
        assert np.array_equal(a, b)
        assert "louvain_nmi_vs_ground_truth" in meta_a
        assert "louvain_nmi_vs_ground_truth" not in meta_b


def test_louvain_recovers_planted_structure(graph):
    _, meta = partition_graph(graph, "community", 4, seed=1)
    assert meta["louvain_nmi_vs_ground_truth"] > 0.8


# ----------------------------- plan y halos -----------------------------

def test_centralized_is_trivial(graph):
    plan, m = make_plan(graph, "centralized", 8, 2)
    assert plan.k == 1 and len(plan.workers[0].halo) == 0
    assert m["edge_cut"] == 0 and m["halo_bytes_est"] == 0 and m["comm_links"] == 0


def test_halo_grows_with_depth(graph):
    part, _ = partition_graph(graph, "community", 4, seed=1)
    tot = [plan_metrics(build_plan(graph, part, 4, d), graph)["halo_total"] for d in (0, 1, 2, 3)]
    assert tot[0] == 0 and tot[0] < tot[1] < tot[2] < tot[3]


@pytest.mark.parametrize("method", METHODS)
def test_verify_plan_passes(graph, method):
    for depth in (0, 1, 2):
        plan, _ = make_plan(graph, method, 4, depth, seed=1)
        verify_plan(plan, graph)


def test_verify_plan_detects_corruption(graph):
    plan, _ = make_plan(graph, "community", 4, 2, seed=1)
    w = plan.workers[0]
    # (a) halo incompleto: se pierde un nodo de halo
    broken = dataclasses.replace(w, halo=w.halo[1:], nodes=np.concatenate([w.owned, w.halo[1:]]))
    bad = PartitionPlan(plan.method, plan.k, plan.depth, plan.n, plan.part,
                        [broken] + plan.workers[1:])
    with pytest.raises(AssertionError):
        verify_plan(bad, graph)
    # (b) dueño duplicado
    dup = dataclasses.replace(plan.workers[1], owned=np.concatenate([plan.workers[1].owned, w.owned[:1]]))
    bad2 = PartitionPlan(plan.method, plan.k, plan.depth, plan.n, plan.part,
                         [plan.workers[0], dup] + plan.workers[2:])
    with pytest.raises(AssertionError):
        verify_plan(bad2, graph)


def test_send_recv_accounting(graph):
    plan, m = make_plan(graph, "hash", 4, 2, seed=1)
    total = 0
    for src in range(4):
        for dst in range(4):
            ids = plan.send_to(src, dst)
            total += len(ids)
            if src == dst:
                assert len(ids) == 0
            elif len(ids):
                assert (plan.part[ids] == src).all(), "solo se envía lo que se posee"
    assert total == m["halo_total"]


def test_bytes_model_scales_with_T(graph):
    part, _ = partition_graph(graph, "community", 4, seed=1)
    plan = build_plan(graph, part, 4, 1)
    m96, m48 = plan_metrics(plan, graph, T=96), plan_metrics(plan, graph, T=48)
    assert m96["halo_bytes_est"] == m96["halo_total"] * 96 * 5
    assert m96["halo_bytes_est"] == 2 * m48["halo_bytes_est"]
    assert m96["halo_bytes_max_worker_est"] <= m96["halo_bytes_est"]


def test_snap_cuts_unit():
    # comunidades iguales: cortes exactamente en sus fronteras
    assert _snap_cuts(np.array([250, 250, 250, 250]), 1000, 4, 0.05).tolist() == [250, 500, 750]
    # una comunidad gigante: la frontera (600) queda fuera de ±5 % => corte exacto (parte la comunidad)
    assert _snap_cuts(np.array([600, 400]), 1000, 2, 0.05).tolist() == [500]
    # frontera dentro de tolerancia: se ajusta a ella
    assert _snap_cuts(np.array([510, 490]), 1000, 2, 0.05).tolist() == [510]


# ----------------------------- integración con el detector (lo más importante) -----------------------------

@pytest.mark.parametrize("method", METHODS)
def test_plan_halo_gives_exact_centralized_result(graph, sig, method):
    """Cada worker, con SU plan (propios + halo) y el mismo algoritmo, reproduce el
    resultado centralizado en sus nodos propios."""
    cfg = DetectorConfig(hops=1, impute_iters=2)
    full = run_centralized(graph, sig, cfg)
    plan, _ = make_plan(graph, method, 4, cfg.halo_depth, seed=1)
    for w in plan.workers:
        Xs, Ms = sig.X[:, w.nodes], sig.mask[:, w.nodes]
        own = slice(0, w.n_owned)
        z = spatial_scores(Xs, Ms, adjacency_hops(w.n_local, w.edges, cfg.hops), cfg)[:, own]
        _, pred = temporal_scores(Xs, Ms, cfg)
        xi, _ = impute(Xs, Ms, adjacency_hops(w.n_local, w.edges, 1), pred, cfg)
        # float32 y otro orden de suma en el subgrafo local => ruido ~1e-5 relativo; una
        # degradación real es >= 0.05, tres órdenes de magnitud mayor.
        assert np.allclose(z, full.scores_spatial[:, w.owned], rtol=1e-3, atol=1e-3)
        assert np.allclose(xi[:, own], full.X_imputed[:, w.owned], rtol=1e-4, atol=1e-3)


def test_one_hop_short_of_requirement_breaks_equivalence(graph, sig):
    cfg = DetectorConfig(hops=1, impute_iters=2)
    full = run_centralized(graph, sig, cfg)
    plan, _ = make_plan(graph, "hash", 4, cfg.halo_depth - 1, seed=1)
    w = plan.workers[0]
    Xs, Ms = sig.X[:, w.nodes], sig.mask[:, w.nodes]
    _, pred = temporal_scores(Xs, Ms, cfg)
    xi, _ = impute(Xs, Ms, adjacency_hops(w.n_local, w.edges, 1), pred, cfg)
    assert np.abs(xi[:, :w.n_owned] - full.X_imputed[:, w.owned]).max() > 0.05


# ----------------------------- entradas -----------------------------

def test_graph_dict_path_roundtrip(tmp_path, graph):
    run_dir = export_graph(graph, tmp_path)
    d = load_graph(run_dir)
    a, _ = partition_graph(graph, "community", 4, seed=1)
    b, _ = partition_graph(d, "community", 4, seed=1)
    assert np.array_equal(a, b)
    plan, _ = make_plan(d, "spatial", 4, 1, seed=1)
    verify_plan(plan, d)


def test_invalid_inputs(graph):
    with pytest.raises(ValueError):
        partition_graph(graph, "inexistente", 4)
    with pytest.raises(ValueError):
        partition_graph(graph, "hash", 0)
    with pytest.raises(ValueError):
        partition_graph(graph, "hash", N + 1)
    no_coords = generate("community", N, seed=1)
    with pytest.raises(ValueError):
        partition_graph(no_coords, "spatial", 4)
    # los métodos topológicos sí funcionan sin coordenadas
    plan, _ = make_plan(no_coords, "community", 4, 1, seed=1)
    verify_plan(plan, no_coords)


def test_metis_optional(graph):
    pytest.importorskip("pymetis")
    plan, m = make_plan(graph, "metis", 4, 1, seed=1)
    verify_plan(plan, graph)
    assert m["cut_ratio"] < 0.1


def test_boundary_nodes(graph):
    part, _ = partition_graph(graph, "community", 4, seed=1)
    b = boundary_nodes(graph, part)
    _, m = make_plan(graph, "community", 4, 1, seed=1)
    assert len(b) == round(m["boundary_fraction"] * N)
    e = graph.edges
    is_b = np.zeros(N, dtype=bool)
    is_b[b] = True
    cross = part[e[:, 0]] != part[e[:, 1]]
    assert is_b[e[cross]].all(), "todo extremo de arista cortada es de frontera"
    # un nodo interior no tiene ningún vecino en otra partición
    inner = np.nonzero(~is_b)[0]
    assert not np.isin(inner, e[cross].ravel()).any()
    assert len(boundary_nodes(graph, np.zeros(N, dtype=np.int64))) == 0


def test_boundary_nodes_min_cut_fraction(graph):
    part, _ = partition_graph(graph, "hash", 4, seed=1)
    allb = boundary_nodes(graph, part)
    half = boundary_nodes(graph, part, 0.5)
    strict = boundary_nodes(graph, part, 0.9)
    assert set(strict) <= set(half) <= set(allb) and len(strict) < len(half) < len(allb)
    e = graph.edges
    cross = part[e[:, 0]] != part[e[:, 1]]
    cut_deg = np.bincount(e[cross].ravel(), minlength=N)
    deg = np.bincount(e.ravel(), minlength=N)
    assert (cut_deg[half] / deg[half] >= 0.5).all()
