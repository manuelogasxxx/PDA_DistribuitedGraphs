"""Tests de analysis.py. Ejecutar:  pytest -q test_analysis.py"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis as an
import orchestrator as orc


# ----------------------------- estadística (casos calculados a mano) -----------------------------

def test_bootstrap_ci():
    x = np.random.default_rng(0).normal(10, 1, 40)
    lo, hi = an.bootstrap_ci(x, "mean", 2000, seed=1)
    assert lo < x.mean() < hi and hi - lo < 1.0
    assert an.bootstrap_ci(x, "mean", 2000, seed=1) == (lo, hi), "reproducible con la misma semilla"
    lo2, hi2 = an.bootstrap_ci(np.tile(x, 4), "mean", 2000, seed=1)
    assert (hi2 - lo2) < (hi - lo), "más datos => IC más estrecho"
    assert an.bootstrap_ci([5.0]) == (5.0, 5.0)
    assert np.isnan(an.bootstrap_ci([np.nan, np.nan])[0])
    med = an.bootstrap_ci(x, "median", 1000)
    assert med[0] < np.median(x) < med[1]


def test_holm_known_example():
    adj = an.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    p = np.array([0.001, 0.2, 0.5])
    a = an.holm(p)
    assert (a >= p).all() and (a <= 1).all()
    nan = an.holm([0.01, np.nan, 0.04])
    assert np.isnan(nan[1]) and np.allclose(nan[[0, 2]], [0.02, 0.04])
    assert np.isnan(an.holm([np.nan])[0])


def test_cliffs_delta():
    assert an.cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    assert an.cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert an.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    # P(x>y) - P(x<y) con x=[1,3], y=[2]: (1 - 1)/2 = 0
    assert an.cliffs_delta([1, 3], [2]) == 0.0
    assert np.isnan(an.cliffs_delta([], [1]))


def test_hodges_lehmann():
    assert an.hodges_lehmann([1, 2, 3]) == 2.0
    assert an.hodges_lehmann([1, 1, 1, 100]) == 1.0, "robusto a un valor extremo (la media sería 25.75)"
    assert np.isnan(an.hodges_lehmann([]))


def test_paired_wilcoxon():
    a = np.arange(15, dtype=float)
    assert an.paired_wilcoxon(a, a) == 1.0
    assert an.paired_wilcoxon(a + 1.0 + 0.01 * np.arange(15), a) < 0.001
    assert np.isnan(an.paired_wilcoxon([1.0], [2.0]))
    assert 0.0 <= an.paired_wilcoxon([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]) <= 1.0


def test_tost():
    rng = np.random.default_rng(0)
    eq = an.tost_ci(rng.normal(0.0, 0.5, 30), 2.5)
    assert eq["equivalent"] and eq["p_tost"] < 0.05 and -2.5 < eq["lo"] < eq["hi"] < 2.5
    ne = an.tost_ci(rng.normal(3.0, 0.5, 30), 2.5)
    assert not ne["equivalent"] and ne["p_tost"] > 0.05
    # IC 90 % exacto para datos conocidos: media 1, sd 1, n=4 -> t(0.95, 3)=2.3534
    d = np.array([0.0, 1.0, 1.0, 2.0])
    r = an.tost_ci(d, 5.0)
    half = 2.353363 * d.std(ddof=1) / 2.0
    assert r["mean"] == 1.0 and r["lo"] == pytest.approx(1.0 - half, abs=1e-4) and r["hi"] == pytest.approx(1.0 + half, abs=1e-4)
    assert an.tost_ci([0.0, 0.0, 0.0], 0.5)["equivalent"], "varianza cero y media dentro del margen"
    assert not an.tost_ci([1.0, 1.0, 1.0], 0.5)["equivalent"]
    assert not an.tost_ci([0.1], 2.5)["equivalent"], "con una sola semilla no se puede concluir"


def test_loglog_fit_recovers_exponent():
    rng = np.random.default_rng(1)
    x = np.geomspace(0.001, 0.9, 60)
    y = 3.0 * x ** 1.5 * np.exp(rng.normal(0, 0.05, 60))
    f = an.loglog_fit(x, y, n_boot=500)
    assert f["b"] == pytest.approx(1.5, abs=0.05) and f["a"] == pytest.approx(3.0, rel=0.15) and f["r2"] > 0.99
    assert f["b_lo"] < 1.5 < f["b_hi"]
    assert np.isnan(an.loglog_fit([0.1, 0.1, 0.1], [1, 2, 3])["b"]), "x sin variación"
    assert np.isnan(an.loglog_fit([0.1, 0.2], [1, 2])["b"]), "menos de 3 puntos"


def test_pareto_front():
    m = an.pareto_front([0, 1, 2, 1], [5, 3, 1, 4])
    assert m.tolist() == [True, True, True, False]
    assert an.pareto_front([1, 1], [2, 2]).tolist() == [True, True], "empates exactos no se dominan"


# ----------------------------- tablas sobre datos sintéticos -----------------------------

def synth(n_seeds=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_seeds + 1):
        for method, shift, mb in (("A", 0.0, 1.0), ("B", -8.0, 30.0)):
            for depth in (0, 2):
                loss = (shift + rng.normal(0, 0.3)) if depth == 0 else 0.0
                rows.append(dict(
                    experiment="E1_x", n=1000, spread_factor=0.2, anomaly="both", missing="structured",
                    magnitude_lo=5, magnitude_hi=10, anomaly_rate=0.01, anomaly_placement="uniform", hops=1,
                    impute_iters=2, method=method, k=4, depth=depth, required_depth=2, graph_seed=s,
                    d_f1=loss / 100, d_auc=loss / 100, d_mae=-loss / 50, halo_bytes=mb * 1e6 if depth else 0.0,
                    replication_factor=1.0 + depth * mb / 20, cut_ratio=0.01 if method == "A" else 0.75,
                    speedup_model=3.0, speedup_model_nogather=6.0, efficiency_model=0.75, work_inflation=1.0,
                    equivalent=(depth == 2), max_abs_diff_spatial=0.0 if depth == 2 else 5.0,
                    t_central=1.0, t_compute_max=0.25, halo_bytes_max_worker=mb * 1e5 * depth,
                    halo_msgs_max_worker=3 * depth, gather_bytes=4e6, d_auc_boundary=loss / 100,
                    d_auc_interior=0.0))
    return pd.DataFrame(rows)


def prepared(df):
    df = df.copy()
    df["d_f1_pp"], df["halo_mb"] = df["d_f1"] * 100, df["halo_bytes"] / 1e6
    return df


def test_equivalence_table_verdicts():
    eq = an.equivalence_table(prepared(synth()), depth=0).set_index("method")
    assert bool(eq.loc["A", "equivalent_25pp"]) and bool(eq.loc["A", "equivalent_05pp"])
    assert not bool(eq.loc["B", "equivalent_25pp"]) and eq.loc["B", "mean_pp"] < -7
    assert eq.loc["A", "n"] == 12


def test_compare_methods_pairing_and_holm():
    df = prepared(synth())
    c = an.compare_methods(df)
    row = c[(c.metric == "d_f1_pp") & (c.depth == 0)].iloc[0]
    assert (row.method_a, row.method_b) == ("A", "B") and row.n_pairs == 12
    assert row.mean_diff == pytest.approx(8.0, abs=0.5) and row.hodges_lehmann == pytest.approx(8.0, abs=0.5)
    assert row.p_wilcoxon < 0.01 and row.cliffs_delta == 1.0
    assert (c["p_holm"].dropna() >= c.loc[c["p_holm"].notna(), "p_wilcoxon"]).all()
    same = c[(c.metric == "d_f1_pp") & (c.depth == 2)].iloc[0]
    assert same.p_wilcoxon == 1.0, "dos métodos idénticos"


def test_summary_columns_and_ci():
    s = an.summarize(prepared(synth()), n_boot=300)
    r = s[(s.method == "B") & (s.depth == 0)].iloc[0]
    assert r.n_seeds == 12 and r["d_f1_pp_ci_lo"] < r["d_f1_pp_mean"] < r["d_f1_pp_ci_hi"]
    assert r["halo_mb_mean"] == 0.0


def test_network_sensitivity_monotone_in_bandwidth():
    df = prepared(synth())
    ns = an.network_sensitivity(df)
    g = ns[(ns.method == "B") & (ns.depth == 2) & (ns.latency_ms == 0.5)].groupby("bandwidth_mbps")["speedup"].mean()
    assert g.is_monotonic_increasing and g.iloc[0] < g.iloc[-1]
    nog = an.network_sensitivity(df, gather=False)
    assert (nog["speedup"] >= ns["speedup"] - 1e-12).all(), "sin recolección nunca es más lento"
    assert an.network_sensitivity(df.drop(columns=["gather_bytes"])).empty


def test_load_results_derives_columns(tmp_path):
    df = synth(3).drop(columns=["experiment"])
    df["equivalent"] = df["equivalent"].map({True: "True", False: "False"})
    p = tmp_path / "r.csv"
    df.to_csv(p, index=False)
    out = an.load_results(p)
    assert (out["experiment"] == "pilot").all() and "d_f1_pp" in out and "halo_mb" in out
    assert out["equivalent"].dtype == bool


# ----------------------------- integración: orquestador -> análisis -----------------------------

@pytest.fixture(scope="module")
def campaign_results():
    cfg = {
        "campaign": "it", "seeds": {"eval": {"range": [1, 3]}, "cal": {"range": [1001, 1001]}},
        "repeats": 1, "order_seed": 3,
        "defaults": {"graph": {"mode": "hybrid", "n": 600, "k_neighbors": 8, "spread_factor": 0.2},
                     "scenario": {"anomaly": "both", "missing": "structured", "anomaly_rate": 0.02,
                                  "missing_rate": 0.05, "magnitude": [5, 10]},
                     "detector": {"hops": 1, "impute_iters": 2}},
        "experiments": [
            {"name": "E1_t", "grid": {"method": ["hash", "community"], "k": [2, 4], "depth": [0, 1, "required"]}},
            {"name": "E4b_t", "scenario": {"anomaly": "both", "missing": "structured", "anomaly_rate": 0.1,
                                            "missing_rate": 0.05, "magnitude": [5, 10],
                                            "placement": {"type": "boundary", "ref_method": "community",
                                                          "k": "grid", "min_cut_fraction": 0.0}},
             "grid": {"method": ["hash", "community"], "k": [4], "depth": [0, 1]}}]}
    out = tempfile.mkdtemp()
    orc.run_campaign(cfg, out, progress=lambda *_a, **_k: None)
    return Path(out) / "it" / "results.csv"


def test_end_to_end_report(campaign_results):
    out = Path(tempfile.mkdtemp())
    info = an.make_report(campaign_results, out, n_boot=200)
    for f in ("summary.csv", "comparisons.csv", "equivalence.csv", "report.md"):
        assert (out / f).exists(), f
    assert any(w.endswith("F3_pareto_bytes_vs_F1.png") for w in info["files"])
    pngs = list((out / "figures").glob("*.png"))
    assert len(pngs) >= 4 and all(p.stat().st_size > 5000 for p in pngs)
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "H1" in text and "H2" in text and "H8" in text
    df = an.load_results(campaign_results)
    req = df[df["depth"] >= df["required_depth"]]
    assert req["equivalent"].all() and f"{len(req)} de {len(req)}" in text
    json.loads((out / "loglog_fit.json").read_text())


def test_h2_excludes_adversarial_experiments(campaign_results):
    df = an.load_results(campaign_results)
    eq = an.equivalence_table(df, 0)
    assert eq["experiment"].astype(str).str.startswith("E4b").any()
    out = Path(tempfile.mkdtemp())
    an.make_report(campaign_results, out, n_boot=100)
    text = (out / "report.md").read_text(encoding="utf-8")
    n_h2 = int(text.split("**H2")[1].split(" de ")[1].split(" ")[0])
    normal = eq[(~eq["experiment"].astype(str).str.startswith("E4b")) & (eq["method"] != "hash")]
    assert n_h2 == len(normal)
