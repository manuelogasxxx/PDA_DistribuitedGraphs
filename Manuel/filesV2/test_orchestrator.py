"""Tests de orchestrator.py. Ejecutar:  pytest -q test_orchestrator.py"""

import copy
import json
import os
import time

import numpy as np
import pytest

import orchestrator as orc

QUIET = lambda *_a, **_k: None
QUALITY = ["d_f1", "d_auc", "cut_ratio", "halo_bytes", "auc", "mae", "equivalent"]


def base_cfg(**over):
    cfg = {
        "campaign": "t", "seeds": {"eval": {"range": [1, 2]}, "cal": {"range": [1001, 1001]}},
        "repeats": 1, "executor": "simulate", "order_seed": 7,
        "defaults": {"graph": {"mode": "hybrid", "n": 600, "k_neighbors": 8, "spread_factor": 0.2},
                     "scenario": {"anomaly": "both", "missing": "structured", "anomaly_rate": 0.02,
                                  "missing_rate": 0.05, "magnitude": [5, 10]},
                     "detector": {"hops": 1, "impute_iters": 2}},
        "experiments": [{"name": "E1", "grid": {"method": ["hash", "community"], "k": [4],
                                                  "depth": [0, "required"]}}],
    }
    cfg.update(over)
    return cfg


def run(cfg, out, **kw):
    return orc.run_campaign(cfg, out, progress=QUIET, **kw)


def rows_of(cfg, out):
    return orc.load_rows(cfg, out)[0]


# ----------------------------- plan -----------------------------

def test_seed_specs():
    assert orc._seeds(3) == [1, 2, 3]
    assert orc._seeds({"range": [5, 8]}) == [5, 6, 7, 8]
    assert orc._seeds({"list": [2, 9]}) == [2, 9]
    assert orc._seeds([4, 6]) == [4, 6]
    assert orc._seeds(None, 2) == [1, 2]
    with pytest.raises(ValueError):
        orc._seeds("x")


def test_expansion_counts_and_sweeps():
    cfg = base_cfg()
    d = orc.describe(cfg)[0]
    assert (d["variants"], d["jobs"], d["runs"]) == (1, 2, 8)
    cfg["experiments"][0]["sweep"] = {"detector.hops": [1, 2], "graph.spread_factor": [0.1, 0.3]}
    d = orc.describe(cfg)[0]
    assert (d["variants"], d["jobs"], d["runs"]) == (4, 8, 32)
    # detector completo como valor de barrido (pares atados)
    cfg["experiments"][0]["sweep"] = {"detector": [{"hops": 1, "impute_iters": 1}, {"hops": 2, "impute_iters": 2}]}
    assert orc.describe(cfg)[0]["variants"] == 2


def test_per_experiment_seed_override_and_grid_k_placement():
    cfg = base_cfg()
    cfg["experiments"][0]["seeds"] = {"eval": {"list": [3, 4, 5]}}
    assert orc.describe(cfg)[0]["jobs"] == 3
    cfg["experiments"].append({
        "name": "ADV", "scenario": {"anomaly": "both", "missing": "structured", "anomaly_rate": 0.1, "missing_rate": 0.05,
                                    "magnitude": [5, 10],
                                    "placement": {"type": "boundary", "ref_method": "community", "k": "grid",
                                                  "min_cut_fraction": 0.0}},
        "grid": {"method": ["hash"], "k": [3, 4], "depth": [0]}})
    variants = [v for v in orc.expand_variants(cfg) if v["experiment"] == "ADV"]
    assert sorted(v["variant"]["scenario"]["placement"]["k"] for v in variants) == [3, 4]
    assert all(v["variant"]["grid"]["k"] == [v["variant"]["scenario"]["placement"]["k"]] for v in variants)


def test_job_ids_are_stable_unique_and_content_addressed():
    cfg = base_cfg()
    a, b = orc.plan_jobs(cfg), orc.plan_jobs(cfg)
    assert [j["job_id"] for j in a] == [j["job_id"] for j in b]
    assert len({j["job_id"] for j in a}) == len(a)
    cfg2 = copy.deepcopy(cfg)
    cfg2["seeds"]["eval"] = {"range": [1, 3]}                  # añadir una semilla no cambia los ids existentes
    ids2 = {j["job_id"] for j in orc.plan_jobs(cfg2)}
    assert {j["job_id"] for j in a} <= ids2 and len(ids2) == len(a) + 1
    cfg3 = copy.deepcopy(cfg)
    cfg3["experiments"][0]["grid"]["depth"] = [0, 1]           # cambiar la rejilla cambia los ids
    assert not ({j["job_id"] for j in a} & {j["job_id"] for j in orc.plan_jobs(cfg3)})


def test_order_is_shuffled_but_reproducible():
    cfg = base_cfg()
    cfg["seeds"]["eval"] = {"range": [1, 12]}
    o1 = [j["job_id"] for j in orc.plan_jobs(cfg)]
    assert o1 == [j["job_id"] for j in orc.plan_jobs(cfg)]
    cfg["order_seed"] = 8
    assert o1 != [j["job_id"] for j in orc.plan_jobs(cfg)]
    assert o1 != sorted(o1)


def test_validation_errors():
    bad = base_cfg()
    bad["experiments"][0]["grid"]["method"] = ["inexistente"]
    with pytest.raises(ValueError):
        orc.validate_config(bad)
    bad = base_cfg()
    del bad["experiments"][0]["grid"]["k"]
    with pytest.raises(ValueError):
        orc.validate_config(bad)
    bad = base_cfg()
    bad["experiments"].append(copy.deepcopy(bad["experiments"][0]))
    with pytest.raises(ValueError):
        orc.validate_config(bad)
    bad = base_cfg()
    bad["experiments"][0]["grid"]["depth"] = [-1]
    with pytest.raises(ValueError):
        orc.validate_config(bad)
    with pytest.raises(ValueError):
        orc.validate_config({"experiments": []})


# ----------------------------- ejecución -----------------------------

def test_run_collect_and_row_content(tmp_path):
    cfg = base_cfg()
    info = run(cfg, tmp_path)
    assert info["ran"] == 2 and info["jobs_done"] == info["jobs_total"] == 2
    rows = rows_of(cfg, tmp_path)
    assert len(rows) == 8
    csv_path = tmp_path / "t" / "results.csv"
    assert csv_path.exists()
    for key in ("experiment", "job_id", "depth_spec", "tau", "tau_source", "d_f1", "k", "depth", "required_depth",
                "halo_msgs_max_worker", "anomaly_placement", "graph_seed"):
        assert key in rows[0], key
    req = [r for r in rows if r["depth_spec"] == "required"]
    assert req and all(r["depth"] == r["required_depth"] == 2 and r["equivalent"] for r in req)
    assert len({r["tau"] for r in rows}) == 1, "un solo tau por escenario"
    assert {r["graph_seed"] for r in rows} == {1, 2}


def test_manifest_and_tau_cache(tmp_path):
    cfg = base_cfg()
    run(cfg, tmp_path)
    man = json.loads((tmp_path / "t" / "manifest.json").read_text())
    assert man["config_hash"] and "numpy" in man["versions"] and man["hardware"]["cpu_count"] >= 1
    assert len(man["sessions"]) == 1 and man["config"]["campaign"] == "t"
    cache = json.loads((tmp_path / "t" / "tau_cache.json").read_text())
    assert len(cache) == 1 and list(cache.values())[0]["tau"] > 0 and list(cache.values())[0]["n_positives"] > 0


def test_resume_skips_done_jobs_and_keeps_tau_cache(tmp_path):
    cfg = base_cfg()
    info = run(cfg, tmp_path, limit=1)
    assert info["ran"] == 1 and info["jobs_done"] == 1
    assert orc.status(cfg, tmp_path)["E1"] == dict(done=1, total=2)
    assert not (tmp_path / "t" / "results.csv").exists(), "results.csv solo cuando la campaña está completa"
    rows_dir = tmp_path / "t" / "rows"
    first = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in rows_dir.glob("*.json")}
    cache_m = (tmp_path / "t" / "tau_cache.json").stat().st_mtime_ns
    time.sleep(0.02)
    info = run(cfg, tmp_path)
    assert info["ran"] == 1 and info["jobs_done"] == 2
    for name, (mt, content) in first.items():
        p = rows_dir / name
        assert p.stat().st_mtime_ns == mt and p.read_bytes() == content, "no se debe rehacer un trabajo terminado"
    assert (tmp_path / "t" / "tau_cache.json").stat().st_mtime_ns == cache_m, "tau no se recalcula"
    assert (tmp_path / "t" / "results.csv").exists()
    assert len(json.loads((tmp_path / "t" / "manifest.json").read_text())["sessions"]) == 2


def test_nothing_to_do_when_complete(tmp_path):
    cfg = base_cfg()
    run(cfg, tmp_path)
    assert run(cfg, tmp_path)["ran"] == 0


def test_orphan_temp_files_are_removed(tmp_path):
    cfg = base_cfg()
    rows_dir = tmp_path / "t" / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / "abc.json.tmp").write_text('{"truncado')
    run(cfg, tmp_path)
    assert not list(rows_dir.glob("*.tmp"))


def test_changed_config_does_not_mix_old_results(tmp_path):
    cfg = base_cfg()
    run(cfg, tmp_path)
    cfg2 = copy.deepcopy(cfg)
    cfg2["experiments"][0]["grid"]["depth"] = [0, 1, "required"]
    rows, info = orc.load_rows(cfg2, tmp_path)
    assert rows == [] and info["jobs_done"] == 0 and info["orphans"] == 2
    info = run(cfg2, tmp_path)
    assert info["jobs_done"] == 2 and info["orphans"] == 2
    assert len(rows_of(cfg2, tmp_path)) == 12 and not any(r["depth_spec"] == "required" and r["depth"] != 2
                                                           for r in rows_of(cfg2, tmp_path))


def test_deterministic_across_independent_runs(tmp_path):
    cfg = base_cfg()
    a, b = tmp_path / "a", tmp_path / "b"
    run(cfg, a)
    run(cfg, b)
    key = lambda r: (r["job_id"], r["method"], r["k"], r["depth"])
    ra = {key(r): r for r in rows_of(cfg, a)}
    rb = {key(r): r for r in rows_of(cfg, b)}
    assert ra.keys() == rb.keys()
    for k_ in ra:
        for q in QUALITY:
            assert ra[k_][q] == rb[k_][q], (k_, q)


def test_parallel_workers_match_serial(tmp_path):
    cfg = base_cfg()
    a, b = tmp_path / "serial", tmp_path / "par"
    run(cfg, a, workers=1)
    run(cfg, b, workers=2)
    key = lambda r: (r["job_id"], r["method"], r["k"], r["depth"])
    ra = {key(r): r for r in rows_of(cfg, a)}
    rb = {key(r): r for r in rows_of(cfg, b)}
    assert ra.keys() == rb.keys()
    assert all(ra[k_][q] == rb[k_][q] for k_ in ra for q in QUALITY)


def test_sweep_columns_and_variants_are_kept_apart(tmp_path):
    cfg = base_cfg()
    cfg["experiments"][0]["sweep"] = {"graph.spread_factor": [0.1, 0.4]}
    cfg["experiments"][0]["grid"] = {"method": ["community"], "k": [4], "depth": [0]}
    cfg["seeds"]["eval"] = {"range": [1, 1]}
    run(cfg, tmp_path)
    rows = rows_of(cfg, tmp_path)
    assert sorted({r["var_graph_spread_factor"] for r in rows}) == [0.1, 0.4]
    assert sorted({r["spread_factor"] for r in rows}) == [0.1, 0.4]
    assert len({r["tau"] for r in rows}) == 2, "cada escenario/variante calibra su propio tau"


def test_adversarial_placement_variant(tmp_path):
    cfg = base_cfg()
    cfg["experiments"] = [{
        "name": "ADV", "scenario": {"anomaly": "both", "missing": "structured", "anomaly_rate": 0.2,
                                    "missing_rate": 0.05, "magnitude": [5, 10],
                                    "placement": {"type": "boundary", "ref_method": "community", "k": "grid",
                                                  "min_cut_fraction": 0.0}},
        "grid": {"method": ["hash", "community"], "k": [3, 4], "depth": [0, 1]}}]
    cfg["seeds"]["eval"] = {"range": [1, 1]}
    info = run(cfg, tmp_path)
    assert info["jobs_done"] == info["jobs_total"] == 2
    rows = rows_of(cfg, tmp_path)
    assert len(rows) == 8
    for r in rows:
        assert r["anomaly_placement"] == f"boundary_community_k{r['k']}_mcf0"


def test_dry_run_estimate_matches_plan_counts():
    full = orc.load_config(os.path.join(os.path.dirname(orc.__file__), "configs", "campaign_full.json"))
    d = {x["experiment"]: x for x in orc.describe(full)}
    assert d["E0_equivalence"]["runs"] == 96 and d["E1_halo_ablation"]["runs"] == 4320
    assert (d["E2_scale_1k"]["runs"], d["E2_scale_10k"]["runs"], d["E2_scale_100k"]["runs"]) == (960, 960, 240)
    assert d["E3_overlap"]["runs"] == 1800 and d["E4_scenarios"]["runs"] == 2880 and d["E4b_adversarial"]["runs"] == 1440
    assert sum(x["runs"] for x in d.values()) == 12696
