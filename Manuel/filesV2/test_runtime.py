"""Tests de runtime.py (procesos reales). Ejecutar:  pytest -q test_runtime.py"""

import dataclasses

import numpy as np
import pytest

from detect import DetectorConfig
from distributed import CostModel, prepare_reference, run_distributed, simulate
from iot_graph_generator import generate
from partition import PartitionPlan, make_plan
from runtime import WorkerPool
from signals import generate_signals

N = 2000
CFG = DetectorConfig(hops=1, impute_iters=2)          # halo requerido = 2


@pytest.fixture(scope="module")
def graph():
    return generate("hybrid", N, seed=3, k_neighbors=8)


@pytest.fixture(scope="module")
def sig(graph):
    return generate_signals(graph, seed=3, anomaly="both", missing="structured")


@pytest.fixture(scope="module")
def ref(graph, sig):
    return prepare_reference(graph, sig, CFG, tau=4.0, repeats=1)


def _run(pool, sig, plan, **kw):
    return pool.run(sig.X, sig.mask, plan, CFG, **kw)


# ----------------------------- corrección con procesos reales -----------------------------

@pytest.mark.parametrize("method", ["hash", "spatial", "community"])
def test_real_workers_equal_centralized(graph, sig, ref, method):
    plan, _ = make_plan(graph, method, 4, CFG.halo_depth, seed=1)
    with WorkerPool(4) as pool:
        out = _run(pool, sig, plan)
    c = ref.result
    assert np.allclose(out.scores_spatial, c.scores_spatial, rtol=1e-3, atol=1e-3)
    assert np.allclose(out.scores_temporal, c.scores_temporal, rtol=1e-3, atol=1e-3)
    assert np.allclose(out.X_imputed, c.X_imputed, rtol=1e-4, atol=1e-3)


def test_real_equals_simulated_even_without_enough_halo(graph, sig):
    """Con halo insuficiente el resultado no coincide con el centralizado, pero debe coincidir con
    la simulación: el runtime ejecuta exactamente el mismo cómputo."""
    plan, _ = make_plan(graph, "hash", 4, 1, seed=1)
    with WorkerPool(4) as pool:
        real = _run(pool, sig, plan)
    sim = simulate(sig, plan, CFG)
    assert np.allclose(real.scores_spatial, sim.scores_spatial, rtol=1e-5, atol=1e-5)
    assert np.allclose(real.X_imputed, sim.X_imputed, rtol=1e-5, atol=1e-4)


def test_pool_reuse_with_different_plans(graph, sig, ref):
    """Corridas consecutivas con k distintos en el mismo pool: sin mensajes viejos mezclados."""
    with WorkerPool(4) as pool:
        for method, k in [("community", 2), ("hash", 4), ("spatial", 3), ("community", 4)]:
            plan, _ = make_plan(graph, method, k, CFG.halo_depth, seed=1)
            out = _run(pool, sig, plan)
            assert np.allclose(out.scores_spatial, ref.result.scores_spatial, rtol=1e-3, atol=1e-3), (method, k)


# ----------------------------- bytes y estadísticas -----------------------------

def test_real_bytes_match_estimate_and_simulation(graph, sig):
    plan, m = make_plan(graph, "hash", 4, 2, seed=1)
    with WorkerPool(4) as pool:
        real = _run(pool, sig, plan)
    sim = simulate(sig, plan, CFG)
    est = m["halo_bytes_est"]
    assert est <= real.info["halo_bytes"] <= est * 1.05, "el encabezado de pickle debe ser < 5 %"
    assert real.info["halo_msgs"] == sim.info["halo_msgs"]
    assert np.array_equal(real.info["link_msgs"], sim.info["link_msgs"])
    assert real.info["gather_bytes"] >= 3 * sig.X.size * 4 * 0.99


def test_no_halo_means_no_messages(graph, sig):
    plan, _ = make_plan(graph, "community", 4, 0, seed=1)
    with WorkerPool(4) as pool:
        out = _run(pool, sig, plan)
    assert out.info["halo_bytes"] == 0 and out.info["halo_msgs"] == 0


def test_cpu_accounting_excludes_timing_repeats(graph, sig):
    """Repetir el cómputo para cronometrar no debe inflar el CPU del worker."""
    plan, _ = make_plan(graph, "community", 2, 2, seed=1)
    with WorkerPool(2) as pool:
        r1 = _run(pool, sig, plan, repeats=1).info["row_extra"]["rt_cpu_max_worker"]
        r3 = _run(pool, sig, plan, repeats=3).info["row_extra"]["rt_cpu_max_worker"]
    assert r3 < 2.0 * r1, (r1, r3)


# ----------------------------- integración con run_distributed -----------------------------

def test_run_distributed_with_real_executor(graph, sig, ref):
    with WorkerPool(4) as pool:
        row = run_distributed(graph, sig, ref, "community", 4, None, CFG, seed=1, repeats=1,
                              executor=pool.executor)
    for key in ("rt_t_dist_wall", "rt_t_dist_cpu_est", "rt_cpu_count", "rt_oversubscribed", "rt_halo_bytes_est",
                "speedup_real_wall", "speedup_cpu_est", "efficiency_cpu_est", "model_error_rel_cpu",
                "halo_bytes", "equivalent", "auc", "d_f1"):
        assert key in row, key
    assert row["equivalent"] and abs(row["d_f1"]) < 0.01
    assert row["rt_oversubscribed"] == (4 > row["rt_cpu_count"])
    assert row["speedup_cpu_est"] == pytest.approx(row["t_central"] / row["rt_t_dist_cpu_est"])
    assert row["efficiency_cpu_est"] == pytest.approx(row["speedup_cpu_est"] / 4)


def test_default_run_distributed_is_unchanged(graph, sig, ref):
    row = run_distributed(graph, sig, ref, "community", 4, None, CFG, seed=1)
    assert "rt_t_dist_wall" not in row and "speedup_model" in row


# ----------------------------- fallos y límites -----------------------------

def test_k_larger_than_pool_is_rejected(graph, sig):
    plan, _ = make_plan(graph, "hash", 4, 1, seed=1)
    with WorkerPool(2) as pool:
        with pytest.raises(ValueError):
            _run(pool, sig, plan)


def test_worker_error_is_propagated_and_breaks_pool(graph, sig):
    plan, _ = make_plan(graph, "community", 2, 2, seed=1)
    bad_worker = dataclasses.replace(plan.workers[0], edges=np.array([[0, 10 ** 6]], dtype=np.int64))
    bad = PartitionPlan(plan.method, plan.k, plan.depth, plan.n, plan.part, [bad_worker, plan.workers[1]])
    pool = WorkerPool(2)
    try:
        with pytest.raises(RuntimeError):
            _run(pool, sig, bad, timeout=20)
        with pytest.raises(RuntimeError):                 # el pool queda inutilizable, no colgado
            _run(pool, sig, plan, timeout=20)
    finally:
        pool.close(force=True)
    assert not any(p.is_alive() for p in pool.procs)


def test_timeout_breaks_pool_cleanly(graph, sig):
    plan, _ = make_plan(graph, "community", 2, 2, seed=1)
    pool = WorkerPool(2)
    try:
        with pytest.raises(RuntimeError):
            _run(pool, sig, plan, timeout=1e-4)
    finally:
        pool.close(force=True)
    assert not any(p.is_alive() for p in pool.procs)


def test_clean_shutdown(graph, sig):
    plan, _ = make_plan(graph, "spatial", 3, 1, seed=1)
    with WorkerPool(3) as pool:
        _run(pool, sig, plan)
        procs = pool.procs
        assert all(p.is_alive() for p in procs)
    assert not any(p.is_alive() for p in procs)


# ----------------------------- transporte -----------------------------

def test_calibrate_ipc_is_sane():
    with WorkerPool(2) as pool:
        cost = pool.calibrate_ipc(sizes=(1_000, 100_000, 4_000_000), reps=(20, 10, 3), trials=3)
        pts = dict(pool.ipc_points)
    assert 1e7 < cost.bandwidth_Bps < 1e11
    assert 1e-7 < cost.latency_s < 1e-2
    assert pts[4_000_000] > pts[1_000], "un mensaje grande tarda más que uno pequeño"
    with pytest.raises(ValueError):
        with WorkerPool(1) as p1:
            p1.calibrate_ipc()
