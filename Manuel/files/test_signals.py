"""Tests de signals.py. Ejecutar:  pytest -q test_signals.py"""

import numpy as np
import pytest

from iot_graph_generator import export_graph, generate, load_graph
from signals import export_signals, generate_signals, load_signals

N = 2000


@pytest.fixture(scope="module")
def hybrid():
    return generate("hybrid", N, seed=3, k_neighbors=8)


def test_shapes_and_conventions(hybrid):
    s = generate_signals(hybrid, seed=1, anomaly="both", missing="structured", T=64)
    assert s.X.shape == s.X_clean.shape == s.labels.shape == s.mask.shape == (64, N)
    assert np.array_equal(np.isnan(s.X), ~s.mask), "NaN debe coincidir exactamente con ~mask"
    assert not np.isnan(s.X_clean).any()
    assert not (s.labels & ~s.mask).any(), "solo se etiquetan lecturas observadas"


def test_determinism(hybrid):
    a = generate_signals(hybrid, seed=5, anomaly="both", missing="structured")
    b = generate_signals(hybrid, seed=5, anomaly="both", missing="structured")
    c = generate_signals(hybrid, seed=6, anomaly="both", missing="structured")
    assert a.signals_sha256() == b.signals_sha256()
    assert a.signals_sha256() != c.signals_sha256()


def test_paired_scenarios_share_clean_signal_and_mask(hybrid):
    """Misma semilla, distinto tipo de anomalía => misma señal limpia y misma máscara."""
    a = generate_signals(hybrid, seed=2, anomaly="point", missing="random")
    b = generate_signals(hybrid, seed=2, anomaly="regional", missing="random")
    assert np.array_equal(a.X_clean, b.X_clean)
    assert np.array_equal(a.mask, b.mask)
    assert not np.array_equal(a.labels, b.labels)


def test_rates_close_to_targets(hybrid):
    s = generate_signals(hybrid, seed=4, anomaly="point", anomaly_rate=0.02,
                         missing="random", missing_rate=0.1)
    assert abs(s.report["missing_fraction_realized"] - 0.10) < 0.01
    assert abs(s.report["anomaly_fraction_realized"] - 0.02) < 0.004
    r = generate_signals(hybrid, seed=4, anomaly="regional", anomaly_rate=0.02,
                         missing="structured", missing_rate=0.1)
    assert 0.10 <= r.report["missing_fraction_realized"] < 0.16
    assert 0.02 <= r.report["anomaly_fraction_realized"] < 0.04


def test_none_scenarios(hybrid):
    s = generate_signals(hybrid, seed=1, anomaly="none", missing="none")
    assert s.mask.all() and not s.labels.any()
    assert np.array_equal(s.X, s.X_clean)


def test_neighbors_are_informative(hybrid):
    """El residuo contra vecinos debe ser mucho menor que la variación de la señal."""
    s = generate_signals(hybrid, seed=1, anomaly="none", missing="none")
    assert s.report["smoothness_ratio"] < 0.2


def test_point_anomaly_magnitude(hybrid):
    s = generate_signals(hybrid, seed=1, anomaly="point", missing="none", magnitude=(5, 10))
    dev = np.abs(s.X - s.X_clean)[s.labels]
    r_sd = s.report["r_sd"]
    assert dev.min() >= 5 * r_sd * 0.999 and dev.max() <= 10 * r_sd * 1.001
    assert np.allclose((s.X - s.X_clean)[~s.labels], 0.0)


def test_regional_events_are_spatially_local(hybrid):
    """Con una tasa mínima cae un solo evento: su extensión no puede superar el
    diámetro de la bola (2 * region_radius_factor * distancia mediana de arista)."""
    s = generate_signals(hybrid, seed=1, anomaly="regional", missing="none",
                         anomaly_rate=0.0005, T=96)
    assert s.report["regional_events"] == 1
    med = np.median(hybrid.distance[~hybrid.is_bridge])
    diameter = 2 * 3.0 * med
    idx = np.nonzero(s.labels.any(axis=0))[0]
    assert len(idx) > 1
    assert np.ptp(hybrid.coords[idx], axis=0).max() <= diameter * 1.001
    # y es un bloque temporal contiguo
    t_idx = np.nonzero(s.labels.any(axis=1))[0]
    assert np.all(np.diff(t_idx) == 1)


def test_fallback_without_coordinates():
    g = generate("community", N, seed=3)
    s = generate_signals(g, seed=1, anomaly="regional", missing="structured")
    assert s.resolved["field_source"] == "graph_diffusion_fallback"
    assert s.report["smoothness_ratio"] < 0.3
    assert s.report["regional_events"] >= 1


def test_roundtrip_export_load(tmp_path, hybrid):
    run_dir = export_graph(hybrid, tmp_path)
    s = generate_signals(load_graph(run_dir), seed=7, anomaly="both", missing="random")
    npz = export_signals(s, run_dir, tag="t")
    d = load_signals(npz)
    assert np.array_equal(d["labels"], s.labels) and np.array_equal(d["mask"], s.mask)
    assert np.allclose(d["X_clean"], s.X_clean)
    assert d["manifest"]["graph_sha256"] == hybrid.graph_sha256()
    assert d["manifest"]["signals_sha256"] == s.signals_sha256()


def test_invalid_params(hybrid):
    with pytest.raises(ValueError):
        generate_signals(hybrid, seed=1, anomaly="xxx")
    with pytest.raises(ValueError):
        generate_signals(hybrid, seed=1, magnitude=(10, 5))
    with pytest.raises(TypeError):
        generate_signals(hybrid, seed=1, inexistente=1)
