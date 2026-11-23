"""Tests de distributed.py. Ejecutar:  pytest -q test_distributed.py"""

import dataclasses

import numpy as np
import pytest

from detect import DetectorConfig
from distributed import CostModel, SimNetwork, prepare_reference, run_distributed, simulate
from iot_graph_generator import generate
from partition import PartitionPlan, build_plan, make_plan, partition_graph
from signals import generate_signals

N = 2000
CFG = DetectorConfig(hops=1, impute_iters=2)          # halo requerido = 2
METHODS = ["hash", "spatial", "community", "community_refined"]


@pytest.fixture(scope="module")
def graph():
    return generate("hybrid", N, seed=3, k_neighbors=8)


@pytest.fixture(scope="module")
def sig(graph):
    return generate_signals(graph, seed=3, anomaly="both", missing="structured")


@pytest.fixture(scope="module")
def ref(graph, sig):
    return prepare_reference(graph, sig, CFG, tau=4.0, repeats=1)


# ----------------------------- equivalencia global -----------------------------

@pytest.mark.parametrize("method", METHODS)
def test_global_equivalence_with_required_depth(graph, sig, ref, method):
    plan, _ = make_plan(graph, method, 4, CFG.halo_depth, seed=1)
    sim = simulate(sig, plan, CFG)
    c = ref.result
    assert np.allclose(sim.scores_spatial, c.scores_spatial, rtol=1e-3, atol=1e-3)
    assert np.allclose(sim.scores_temporal, c.scores_temporal, rtol=1e-3, atol=1e-3)
    assert np.allclose(sim.scores_combined, c.scores_combined, rtol=1e-3, atol=1e-3)
    assert np.allclose(sim.X_imputed, c.X_imputed, rtol=1e-4, atol=1e-3)


def test_owned_only_is_a_pure_optimization(graph, sig):
    plan, _ = make_plan(graph, "community", 4, CFG.halo_depth, seed=1)
    a = simulate(sig, plan, CFG, owned_only=True)
    b = simulate(sig, plan, CFG, owned_only=False)
    assert np.allclose(a.scores_spatial, b.scores_spatial, rtol=1e-4, atol=1e-4)
    assert np.allclose(a.scores_temporal, b.scores_temporal, rtol=1e-4, atol=1e-4)
    assert np.allclose(a.X_imputed, b.X_imputed, rtol=1e-4, atol=1e-3)


def test_centralized_method_through_distributed_path(graph, sig, ref):
    plan, _ = make_plan(graph, "centralized", 1, CFG.halo_depth)
    sim = simulate(sig, plan, CFG)
    assert sim.info["halo_bytes"] == 0 and sim.info["t_exchange_model"] == 0.0
    assert np.allclose(sim.scores_spatial, ref.result.scores_spatial, rtol=1e-4, atol=1e-4)


# ----------------------------- halo insuficiente -----------------------------

def test_insufficient_depth_degrades_monotonically(graph, sig, ref):
    d = []
    for depth in (0, 1, 2):
        plan, _ = make_plan(graph, "hash", 4, depth, seed=1)
        sim = simulate(sig, plan, CFG)
        d.append(float(np.abs(sim.scores_spatial - ref.result.scores_spatial).max()))
    assert d[0] > 1.0 and d[0] > d[1] or d[1] < 1e-2
    assert d[2] < 1e-2, "con la profundidad requerida debe coincidir"


def test_hash_without_halo_loses_quality(graph, sig, ref):
    row = run_distributed(graph, sig, ref, "hash", k=8, depth=0, cfg=CFG, seed=1)
    assert row["d_auc"] < -0.01 and row["d_f1"] < -0.02 and not row["equivalent"]
    assert row["halo_bytes"] == 0


def test_structured_partition_without_halo_is_nearly_lossless(graph, sig, ref):
    """Interior exacto; solo se degradan los nodos de frontera, que son pocos."""
    plan, _ = make_plan(graph, "community", 4, 0, seed=1)
    sim = simulate(sig, plan, CFG)
    gap = np.abs(sim.scores_spatial - ref.result.scores_spatial)
    e = graph.edges
    cross = plan.part[e[:, 0]] != plan.part[e[:, 1]]
    boundary = np.zeros(N, dtype=bool)
    boundary[e[cross].ravel()] = True
    assert gap[:, ~boundary].max() < 1e-3, "los nodos interiores no deben cambiar"
    assert gap[:, boundary].max() > 0.1, "los nodos de frontera sí cambian sin halo"


# ----------------------------- red y bytes -----------------------------

def test_network_accounting_matches_plan(graph, sig):
    plan, m = make_plan(graph, "hash", 4, 2, seed=1)
    sim = simulate(sig, plan, CFG)
    T = sig.X.shape[0]
    for src in range(4):
        for dst in range(4):
            expect = len(plan.send_to(src, dst)) * T * (4 + 1)
            assert sim.info["link_bytes"][src, dst] == expect
            assert sim.info["link_msgs"][src, dst] == (1 if expect else 0)
    assert sim.info["halo_bytes"] == m["halo_bytes_est"], "bytes medidos = modelo del paso 3"


def test_sim_network_delivers_copies():
    net = SimNetwork(2)
    a = np.arange(4, dtype=np.float32)
    (b,) = net.send(0, 1, (a,))
    b[0] = 99
    assert a[0] == 0 and net.bytes[0, 1] == 16 and net.msgs[0, 1] == 1


def test_workers_depend_on_messages(graph, sig, ref):
    """Si se suprimen mensajes de halo, el resultado cambia: el halo solo llega por la red."""
    plan, _ = make_plan(graph, "hash", 4, CFG.halo_depth, seed=1)
    w = plan.workers[0]
    src = next(iter(w.recv_from))
    cut = dataclasses.replace(w, recv_from={s: (np.empty(0, dtype=np.int64) if s == src else v)
                                            for s, v in w.recv_from.items()})
    broken = PartitionPlan(plan.method, plan.k, plan.depth, plan.n, plan.part,
                           [cut] + plan.workers[1:])
    sim = simulate(sig, broken, CFG)
    assert np.abs(sim.scores_spatial - ref.result.scores_spatial).max() > 0.1


def test_halo_arrays_are_sorted_invariant(graph):
    plan, _ = make_plan(graph, "community", 4, 2, seed=1)
    for w in plan.workers:
        assert np.all(np.diff(w.halo) > 0) and np.all(np.diff(w.owned) > 0)


# ----------------------------- modelo de costos -----------------------------

def test_cost_model_identities(graph, sig):
    plan, _ = make_plan(graph, "community", 4, 2, seed=1)
    sim = simulate(sig, plan, CFG)
    i = sim.info
    assert i["t_dist_model"] == pytest.approx(i["t_exchange_model"] + i["t_compute_max"] + i["t_gather_model"])
    assert i["t_compute_max"] <= i["t_compute_sum"] + 1e-12
    for cost in (CostModel(bandwidth_Bps=1e15, latency_s=0.0), CostModel(bandwidth_Bps=1e6, latency_s=0.01)):
        j = simulate(sig, plan, CFG, cost=cost).info
        expected = (j["link_bytes"].sum(axis=0) / cost.bandwidth_Bps
                    + j["link_msgs"].sum(axis=0) * cost.latency_s).max()
        assert j["t_exchange_model"] == pytest.approx(expected)
        assert j["t_dist_model_nogather"] == pytest.approx(j["t_exchange_model"] + j["t_compute_max"])
        assert j["halo_bytes"] == i["halo_bytes"], "los bytes no dependen del modelo de red"
    fast = simulate(sig, plan, CFG, cost=CostModel(bandwidth_Bps=1e15, latency_s=0.0)).info
    slow = simulate(sig, plan, CFG, cost=CostModel(bandwidth_Bps=1e6, latency_s=0.01)).info
    assert slow["t_exchange_model"] > 1000 * fast["t_exchange_model"]


def test_no_halo_means_no_exchange_time(graph, sig):
    plan, _ = make_plan(graph, "community", 4, 0, seed=1)
    i = simulate(sig, plan, CFG).info
    assert i["halo_bytes"] == 0 and i["t_exchange_model"] == 0.0


# ----------------------------- filas de resultados -----------------------------

def test_row_schema_and_consistency(graph, sig, ref):
    row = run_distributed(graph, sig, ref, "community", k=4, depth=None, cfg=CFG, seed=1, verify=True)
    required = ["method", "k", "depth", "required_depth", "n", "graph_seed", "anomaly", "missing",
                "cut_ratio", "replication_factor", "halo_bytes", "halo_bytes_max_worker", "t_central",
                "t_compute_max", "t_dist_model", "speedup_model", "efficiency_model", "work_inflation",
                "auc", "f1", "mae", "d_auc", "d_f1", "d_mae", "d_auc_boundary", "d_auc_interior",
                "max_abs_diff_spatial", "equivalent"]
    for key in required:
        assert key in row, key
    assert row["depth"] == CFG.halo_depth and row["equivalent"]
    assert abs(row["d_auc"]) < 1e-3 and abs(row["d_f1"]) < 0.01
    assert row["efficiency_model"] == pytest.approx(row["speedup_model"] / 4)
    assert row["tau"] == 4.0


def test_reference_tau_policy(graph, sig):
    r_given = prepare_reference(graph, sig, CFG, tau=3.3, repeats=1)
    r_auto = prepare_reference(graph, sig, CFG, tau=None, repeats=1)
    assert r_given.tau == 3.3 and r_given.tau_source == "given"
    assert r_auto.tau_source == "best_f1_on_case" and r_auto.tau > 0


def test_run_with_prebuilt_plan(graph, sig, ref):
    plan, _ = make_plan(graph, "spatial", 4, CFG.halo_depth)
    row = run_distributed(graph, sig, ref, plan, cfg=CFG)
    assert row["method"] == "spatial" and row["k"] == 4 and row["equivalent"]
