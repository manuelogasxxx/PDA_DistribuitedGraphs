"""Tests de metrics.py y detect.py. Ejecutar:  pytest -q test_detect.py"""

import numpy as np
import pytest

from detect import (DetectorConfig, adjacency_hops, impute, induced_edges, run_centralized,
                    spatial_scores, temporal_scores)
from iot_graph_generator import generate
from metrics import (auc_roc, average_precision, best_f1_threshold, detection_report, evaluate,
                     imputation_report, prf_at)
from signals import generate_signals

N = 2000


@pytest.fixture(scope="module")
def graph():
    return generate("hybrid", N, seed=2, k_neighbors=8)


@pytest.fixture(scope="module")
def sig(graph):
    return generate_signals(graph, seed=2, anomaly="both", missing="structured")


# ----------------------------- métricas (casos calculados a mano) -----------------------------

S = np.array([.9, .8, .7, .6, .5, .4])
Y = np.array([1, 1, 0, 1, 0, 0], bool)


def test_auc_ap_known_values():
    assert auc_roc(S, Y) == pytest.approx(8 / 9)
    assert average_precision(S, Y) == pytest.approx((1 + 1 + 3 / 4) / 3)


def test_auc_ties_and_degenerate():
    assert auc_roc(np.ones(4), np.array([1, 0, 1, 0], bool)) == pytest.approx(0.5)
    assert np.isnan(auc_roc(S, np.zeros(6, bool)))
    assert np.isnan(average_precision(S, np.zeros(6, bool)))


def test_threshold_and_prf():
    tau, f1 = best_f1_threshold(S, Y)
    assert tau == pytest.approx(0.6) and f1 == pytest.approx(6 / 7)
    r = prf_at(S, Y, 0.6)
    assert (r["tp"], r["fp"], r["fn"]) == (3, 1, 0)
    assert r["f1"] == pytest.approx(6 / 7)


def test_detection_report_nan_scores_count_as_misses():
    scores = np.array([[np.nan, 5.0], [0.1, 0.2]])
    labels = np.array([[True, False], [False, False]])
    mask = np.ones((2, 2), bool)
    rep = detection_report(scores, labels, mask, tau=1.0)
    assert rep["fn"] == 1 and rep["tp"] == 0, "score NaN debe ser falso negativo"


def test_detection_ignores_missing_entries():
    scores = np.array([[9.0, 0.0]])
    labels = np.array([[True, False]])
    mask = np.array([[False, True]])
    assert detection_report(scores, labels, mask)["n_eval"] == 1


def test_imputation_report():
    clean = np.array([[1.0, 2.0, 3.0]])
    imputed = np.array([[1.0, 2.5, 2.0]])
    mask = np.array([[True, False, False]])
    r = imputation_report(imputed, clean, mask)
    assert r["n_missing"] == 2 and r["mae"] == pytest.approx(0.75)
    assert r["rmse"] == pytest.approx(np.sqrt((0.25 + 1.0) / 2))


# ----------------------------- detector -----------------------------

def test_shapes_and_no_nan(graph, sig):
    r = run_centralized(graph, sig)
    for arr in (r.scores_spatial, r.scores_temporal, r.scores_combined, r.X_imputed):
        assert arr.shape == sig.X.shape and not np.isnan(arr).any()
    assert (r.scores_spatial[~sig.mask] == 0).all(), "sin dato no hay score"
    assert np.array_equal(r.X_imputed[sig.mask], sig.X[sig.mask]), "no alterar lo observado"


def test_spatial_beats_temporal_on_point_anomalies(graph):
    s = generate_signals(graph, seed=3, anomaly="point", missing="none")
    r = run_centralized(graph, s)
    ap_sp = detection_report(r.scores_spatial, s.labels, s.mask)["ap"]
    ap_tm = detection_report(r.scores_temporal, s.labels, s.mask)["ap"]
    assert ap_sp > 0.8 and ap_sp > ap_tm + 0.2, (ap_sp, ap_tm)


def test_spatial_imputation_beats_temporal_only(graph, sig):
    r = run_centralized(graph, sig)
    _, pred = temporal_scores(sig.X, sig.mask)
    X_t = np.where(sig.mask, np.nan_to_num(sig.X), pred)
    mae_sp = imputation_report(r.X_imputed, sig.X_clean, sig.mask)["mae"]
    mae_tm = imputation_report(X_t, sig.X_clean, sig.mask)["mae"]
    assert mae_sp < 0.7 * mae_tm


def test_deeper_halo_improves_imputation(graph, sig):
    mae = [imputation_report(run_centralized(graph, sig, DetectorConfig(impute_iters=k)).X_imputed,
                             sig.X_clean, sig.mask)["mae"] for k in (1, 2, 4)]
    assert mae[0] > mae[1] > mae[2]


def test_adjacency_hops(graph):
    A1, A2 = adjacency_hops(graph.n, graph.edges, 1), adjacency_hops(graph.n, graph.edges, 2)
    assert A1.diagonal().sum() == 0 and A2.diagonal().sum() == 0
    assert (A1 != A1.T).nnz == 0 and (A2 != A2.T).nnz == 0
    assert A2.nnz > A1.nnz
    assert (A1.multiply(A2) != A1).nnz == 0, "vecinos a 1 salto deben estar dentro de los de 2"


# ----------------------------- equivalencia distribuida (el test más importante) -----------------------------

def _expand(A1, own_mask, depth):
    reach = own_mask.copy()
    for _ in range(depth):
        reach |= (A1 @ reach.astype(np.float32)) > 0
    return reach


def _local_run(graph, sig, own_mask, depth, cfg):
    """Lo que haría un worker: propios + halo de `depth` saltos, mismo algoritmo."""
    A1 = adjacency_hops(graph.n, graph.edges, 1)
    nodes = np.nonzero(_expand(A1, own_mask, depth))[0]
    le = induced_edges(graph.n, graph.edges, nodes)
    Xs, Ms = sig.X[:, nodes], sig.mask[:, nodes]
    own_loc = own_mask[nodes]
    z_sp = spatial_scores(Xs, Ms, adjacency_hops(len(nodes), le, cfg.hops), cfg)
    z_tm, pred = temporal_scores(Xs, Ms, cfg)
    X_imp, _ = impute(Xs, Ms, adjacency_hops(len(nodes), le, 1), pred, cfg)
    return z_sp[:, own_loc], z_tm[:, own_loc], X_imp[:, own_loc]


@pytest.mark.parametrize("hops,iters", [(1, 1), (2, 2), (1, 3)])
def test_distributed_equals_centralized_with_enough_halo(graph, sig, hops, iters):
    cfg = DetectorConfig(hops=hops, impute_iters=iters)
    full = run_centralized(graph, sig, cfg)
    own = (np.arange(graph.n) % 4) == 0          # hash: peor caso de frontera
    z_sp, z_tm, X_imp = _local_run(graph, sig, own, cfg.halo_depth, cfg)
    assert np.allclose(z_sp, full.scores_spatial[:, own], rtol=1e-3, atol=1e-3)
    assert np.allclose(z_tm, full.scores_temporal[:, own], rtol=1e-3, atol=1e-3)
    assert np.allclose(X_imp, full.X_imputed[:, own], rtol=1e-4, atol=1e-3)


def test_no_halo_degrades(graph, sig):
    cfg = DetectorConfig(hops=1, impute_iters=1)
    full = run_centralized(graph, sig, cfg)
    own = (np.arange(graph.n) % 4) == 0
    z_sp, z_tm, X_imp = _local_run(graph, sig, own, 0, cfg)
    assert np.abs(z_sp - full.scores_spatial[:, own]).max() > 1.0
    assert np.abs(X_imp - full.X_imputed[:, own]).max() > 0.5
    # el detector temporal no usa vecinos: no necesita halo
    assert np.allclose(z_tm, full.scores_temporal[:, own], rtol=1e-3, atol=1e-3)


def test_insufficient_halo_is_detected(graph, sig):
    """hops=2 con halo de 1 salto NO es suficiente."""
    cfg = DetectorConfig(hops=2, impute_iters=2)
    full = run_centralized(graph, sig, cfg)
    own = (np.arange(graph.n) % 4) == 0
    z_sp, _, _ = _local_run(graph, sig, own, 1, cfg)
    assert np.abs(z_sp - full.scores_spatial[:, own]).max() > 0.1


def test_evaluate_row(graph, sig):
    r = run_centralized(graph, sig)
    row = evaluate(r.scores_spatial, r.X_imputed, sig, tau=3.0)
    for k in ("det_auc", "det_ap", "det_f1", "det_precision", "det_recall", "imp_mae", "imp_rmse"):
        assert k in row and np.isfinite(row[k])


def test_spatial_rows_matches_full(graph, sig):
    """Calcular solo un subconjunto de nodos da lo mismo que calcular todos y recortar."""
    cfg = DetectorConfig(hops=2)
    A = adjacency_hops(graph.n, graph.edges, 2)
    full = spatial_scores(sig.X, sig.mask, A, cfg)
    rows = np.sort(np.random.default_rng(0).choice(graph.n, size=300, replace=False))
    part = spatial_scores(sig.X, sig.mask, A, cfg, rows=rows)
    assert part.shape == (sig.X.shape[0], 300)
    assert np.allclose(part, full[:, rows], rtol=1e-4, atol=1e-4)
