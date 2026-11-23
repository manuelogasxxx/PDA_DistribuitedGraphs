#!/usr/bin/env python3
"""
orchestrator.py  (v1.0.0)  --  Paso 6: orquestador de campañas de experimentos.

Qué hace
  Lee una configuración (JSON o YAML), expande la rejilla en TRABAJOS (experimento x variante x semilla),
  calibra el umbral tau por escenario en semillas aparte, ejecuta los trabajos y guarda los resultados.
  Un trabajo construye el grafo y las señales UNA vez y recorre toda la rejilla (método x k x profundidad),
  de modo que todos los métodos se comparan sobre los mismos datos (comparación pareada).

Propiedades
  Reanudable     cada trabajo se guarda en su propio archivo con escritura ATÓMICA (se escribe a un
                 temporal y se renombra). Reanudar = saltar los trabajos cuyo archivo existe. Si el proceso
                 muere a mitad de un trabajo, ese trabajo simplemente se repite.
  Por contenido  el identificador de un trabajo es un hash de (experimento, variante, semilla, repeticiones,
                 modelo de red, ejecutor). Cambiar la configuración nunca mezcla resultados viejos con
                 nuevos: 'collect' solo incluye los trabajos del plan actual.
  Paralelo       --workers N reparte trabajos entre N procesos (por semilla). Con ejecutor "real" se fuerza
                 N=1 (los workers reales ya usan los núcleos).
  Reproducible   orden aleatorio con semilla fija, manifest con versiones, git y hardware, tau en caché.

Configuración (ver configs/demo.json y configs/campaign_full.json)
  "defaults": {graph, scenario, detector}   valores base
  "experiments": [{
      "name": "E1_halo",
      "graph"|"scenario"|"detector": {...},        sobrescribe los defaults
      "sweep": {"detector.hops": [1, 2]},          cada combinación es una VARIANTE (claves con puntos)
      "grid":  {"method": [...], "k": [...], "depth": [0, 1, "required"]},
      "seeds": {"eval": {"range": [1, 30]}}        opcional: sobrescribe las semillas globales
  }]
  scenario.placement = {"type": "boundary", "ref_method": "community", "k": "grid"|N,
                        "min_cut_fraction": 0.5}   prueba adversa E4b (anomalías en la frontera)
  Semillas: entero N (1..N), {"range": [a, b]} (inclusivo) o {"list": [...]}.

Uso:
  python orchestrator.py run configs/demo.json --out ./campaigns --workers 1
  python orchestrator.py run configs/demo.json --out ./campaigns --dry-run      # cuántos trabajos y tiempo aprox.
  python orchestrator.py status  configs/demo.json --out ./campaigns
  python orchestrator.py collect configs/demo.json --out ./campaigns            # -> results.csv
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from detect import DetectorConfig, run_centralized
from distributed import CostModel, prepare_reference, run_distributed, write_rows_csv
from iot_graph_generator import _versions, generate
from metrics import best_f1_threshold
from partition import _gv, boundary_nodes, louvain_labels, partition_graph
from signals import generate_signals

ORCH_VERSION = "1.0.0"
KNOWN_METHODS = {"centralized", "hash", "spatial", "community", "community_refined", "metis"}
SEC_PER_NODE_RUN = 2e-4          # heurística gruesa de segundos por corrida = 2e-4 * n (medido: 1k→0.13 s, 10k→1.5 s, 100k→25 s)


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def load_config(path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":"))


def _hash(obj, n=12) -> str:
    return hashlib.sha1(_canon(obj).encode()).hexdigest()[:n]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _seeds(spec, default=None) -> list[int]:
    if spec is None:
        spec = default
    if isinstance(spec, int):
        return list(range(1, spec + 1))
    if isinstance(spec, dict):
        if "range" in spec:
            a, b = spec["range"]
            return list(range(int(a), int(b) + 1))
        if "list" in spec:
            return [int(x) for x in spec["list"]]
    if isinstance(spec, list):
        return [int(x) for x in spec]
    raise ValueError(f"semillas inválidas: {spec!r}")


def _set_dotted(d: dict, key: str, value) -> None:
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate_config(cfg: dict) -> None:
    if not cfg.get("experiments"):
        raise ValueError("la configuración necesita al menos un experimento")
    names = [e.get("name") for e in cfg["experiments"]]
    if len(set(names)) != len(names) or None in names:
        raise ValueError("cada experimento necesita un 'name' único")
    for e in cfg["experiments"]:
        grid = e.get("grid", {})
        for key in ("method", "k", "depth"):
            if not grid.get(key):
                raise ValueError(f"{e['name']}: grid.{key} es obligatorio")
        bad = set(grid["method"]) - KNOWN_METHODS
        if bad:
            raise ValueError(f"{e['name']}: métodos desconocidos {sorted(bad)}")
        for d in grid["depth"]:
            if d != "required" and not (isinstance(d, int) and d >= 0):
                raise ValueError(f"{e['name']}: depth debe ser entero >= 0 o 'required'")


def expand_variants(cfg: dict) -> list[dict]:
    """Variantes = experimentos x combinaciones de 'sweep'. Cada variante lleva su 'var' (valores barridos)."""
    validate_config(cfg)
    defaults = cfg.get("defaults", {})
    variants = []
    for exp in cfg["experiments"]:
        base = _merge(defaults, {k: exp[k] for k in ("graph", "scenario", "detector") if k in exp})
        base["grid"] = copy.deepcopy(exp["grid"])
        sweep = exp.get("sweep", {})
        keys = list(sweep)
        for combo in (itertools.product(*[sweep[k] for k in keys]) if keys else [()]):
            v = copy.deepcopy(base)
            var = {}
            for key, val in zip(keys, combo):
                _set_dotted(v, key, val)
                var[key] = val
            pl = v.get("scenario", {}).get("placement")
            ks = v["grid"]["k"]
            kvals = [[k] for k in ks] if (pl and pl.get("k") == "grid") else [ks]
            for kk in kvals:
                vv = copy.deepcopy(v)
                vv["grid"]["k"] = kk
                if pl and pl.get("k") == "grid":
                    vv["scenario"]["placement"]["k"] = kk[0]
                var2 = dict(var)
                if pl and pl.get("k") == "grid":
                    var2["grid.k"] = kk[0]
                variants.append(dict(experiment=exp["name"], variant=vv, var=var2,
                                     seeds=_seeds(exp.get("seeds", {}).get("eval"),
                                                  cfg.get("seeds", {}).get("eval", 1))))
    return variants


def plan_jobs(cfg: dict) -> list[dict]:
    """Trabajos = variante x semilla de evaluación, en orden aleatorio reproducible."""
    repeats, executor = int(cfg.get("repeats", 1)), cfg.get("executor", "simulate")
    cost = cfg.get("cost", {"bandwidth_mbps": 1000, "latency_ms": 0.5})
    jobs = []
    for v in expand_variants(cfg):
        for seed in v["seeds"]:
            ident = dict(experiment=v["experiment"], variant=v["variant"], seed=seed,
                         repeats=repeats, executor=executor, cost=cost)
            jobs.append(dict(job_id=_hash(ident), experiment=v["experiment"], variant=v["variant"],
                             var=v["var"], seed=seed, repeats=repeats, executor=executor, cost=cost,
                             tau_key=tau_key(v["variant"], cfg)))
    random.Random(cfg.get("order_seed", 12345)).shuffle(jobs)
    return jobs


def tau_key(variant: dict, cfg: dict) -> str:
    ident = dict(graph=variant["graph"], scenario=variant["scenario"], detector=variant["detector"],
                 cal=_seeds(cfg.get("seeds", {}).get("cal"), {"range": [1001, 1003]}))
    return _hash(ident)


# ---------------------------------------------------------------------------
# Construcción de un caso y ejecución de un trabajo
# ---------------------------------------------------------------------------

def build_case(variant: dict, seed: int, labels=None):
    """Grafo + señales de una variante y semilla. Devuelve (g, sig, labels_louvain)."""
    gcfg = dict(variant["graph"])
    mode, n = gcfg.pop("mode", "hybrid"), gcfg.pop("n")
    g = generate(mode, n, seed=seed, **gcfg)
    sc = dict(variant["scenario"])
    placement = sc.pop("placement", None)
    if "magnitude" in sc:
        sc["magnitude"] = tuple(sc["magnitude"])
    if placement:
        ref, kref = placement["ref_method"], int(placement["k"])
        if "community" in ref and labels is None:
            labels = louvain_labels(_gv(g), seed)
        part, _ = partition_graph(g, ref, kref, seed=seed, **(dict(labels=labels) if "community" in ref else {}))
        mcf = float(placement.get("min_cut_fraction", 0.0))
        sc["anomaly_nodes"] = boundary_nodes(g, part, mcf)
        sc["anomaly_placement"] = f"boundary_{ref}_k{kref}_mcf{mcf:g}"
    return g, generate_signals(g, seed=seed, **sc), labels


def calibrate_tau(variant: dict, cfg: dict) -> dict:
    """tau que maximiza el F1 del CENTRALIZADO, mezclando las semillas de calibración (disjuntas de las de evaluación)."""
    det = DetectorConfig(**variant["detector"])
    s_all, y_all = [], []
    for sd in _seeds(cfg.get("seeds", {}).get("cal"), {"range": [1001, 1003]}):
        g, sig, _ = build_case(variant, sd)
        res = run_centralized(g, sig, det)
        s_all.append(res.scores_spatial[sig.mask])
        y_all.append(sig.labels[sig.mask])
    y = np.concatenate(y_all)
    if not y.any():
        raise ValueError("el escenario de calibración no tiene anomalías: no se puede calibrar tau")
    tau, f1 = best_f1_threshold(np.concatenate(s_all).astype(np.float64), y)
    return dict(tau=float(tau), f1_cal=float(f1), n_positives=int(y.sum()))


def run_job(spec: dict, executor_obj=None) -> tuple[list, float]:
    """Un trabajo: construye el caso una vez y recorre la rejilla. Devuelve (filas, segundos)."""
    t0 = time.perf_counter()
    variant, seed = spec["variant"], spec["seed"]
    grid = variant["grid"]
    g, sig, labels = build_case(variant, seed)
    if labels is None and any("community" in m for m in grid["method"]):
        labels = louvain_labels(_gv(g), seed)
    det = DetectorConfig(**variant["detector"])
    cost = CostModel(bandwidth_Bps=spec["cost"]["bandwidth_mbps"] * 1e6 / 8, latency_s=spec["cost"]["latency_ms"] / 1e3)
    ref = prepare_reference(g, sig, det, tau=spec["tau"], repeats=max(spec["repeats"], 1))
    rows = []
    for method, k, depth in itertools.product(grid["method"], grid["k"], grid["depth"]):
        d = det.halo_depth if depth == "required" else int(depth)
        opts = dict(labels=labels) if ("community" in method and labels is not None) else {}
        row = run_distributed(g, sig, ref, method, int(k), d, det, seed, cost, repeats=spec["repeats"],
                              partition_opts=opts, executor=executor_obj)
        row.update(experiment=spec["experiment"], job_id=spec["job_id"], depth_spec=str(depth),
                   tau_source="calibrated",
                   **{"var_" + k_.replace(".", "_"): (v if np.isscalar(v) or v is None else _canon(v))
                      for k_, v in spec["var"].items()})
        rows.append(row)
    return rows, time.perf_counter() - t0


def _worker_entry(spec):                           # función de nivel superior: picklable para ProcessPool
    rows, dt = run_job(spec)
    return spec["job_id"], rows, dt


# ---------------------------------------------------------------------------
# Almacenamiento
# ---------------------------------------------------------------------------

def _paths(cfg, out_dir):
    root = Path(out_dir) / cfg.get("campaign", "campaign")
    return root, root / "rows", root / "tau_cache.json", root / "manifest.json"


def _job_file(rows_dir: Path, job_id: str) -> Path:
    return rows_dir / f"{job_id}.json"


def save_job(rows_dir: Path, spec: dict, rows: list, seconds: float) -> None:
    payload = dict(job_id=spec["job_id"], experiment=spec["experiment"], seed=spec["seed"],
                   variant=spec["variant"], seconds=seconds, rows=rows)
    final = _job_file(rows_dir, spec["job_id"])
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    os.replace(tmp, final)                         # atómico: el archivo existe completo o no existe


def load_rows(cfg: dict, out_dir) -> tuple[list, dict]:
    """Filas de los trabajos del plan ACTUAL (ignora archivos huérfanos de otras configuraciones)."""
    _, rows_dir, _, _ = _paths(cfg, out_dir)
    wanted = {j["job_id"] for j in plan_jobs(cfg)}
    rows, found = [], set()
    for p in sorted(rows_dir.glob("*.json")) if rows_dir.exists() else []:
        if p.stem in wanted:
            rows.extend(json.loads(p.read_text(encoding="utf-8"))["rows"])
            found.add(p.stem)
    orphans = len([p for p in rows_dir.glob("*.json") if p.stem not in wanted]) if rows_dir.exists() else 0
    return rows, dict(jobs_done=len(found), jobs_total=len(wanted), orphans=orphans)


def status(cfg: dict, out_dir) -> dict:
    _, rows_dir, _, _ = _paths(cfg, out_dir)
    jobs = plan_jobs(cfg)
    per = {}
    for j in jobs:
        d = per.setdefault(j["experiment"], [0, 0])
        d[1] += 1
        d[0] += _job_file(rows_dir, j["job_id"]).exists()
    return {k: dict(done=v[0], total=v[1]) for k, v in per.items()}


# ---------------------------------------------------------------------------
# Campaña
# ---------------------------------------------------------------------------

def _git():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def describe(cfg: dict) -> list[dict]:
    """Resumen del plan: variantes, trabajos y corridas por experimento, con tiempo aproximado."""
    out = []
    for e in cfg["experiments"]:
        vs = [v for v in expand_variants(cfg) if v["experiment"] == e["name"]]
        jobs = sum(len(v["seeds"]) for v in vs)
        runs = sum(len(v["seeds"]) * len(v["variant"]["grid"]["method"]) * len(v["variant"]["grid"]["k"])
                   * len(v["variant"]["grid"]["depth"]) for v in vs)
        secs = sum(len(v["seeds"]) * len(v["variant"]["grid"]["method"]) * len(v["variant"]["grid"]["k"])
                   * len(v["variant"]["grid"]["depth"]) * SEC_PER_NODE_RUN * v["variant"]["graph"]["n"] for v in vs)
        out.append(dict(experiment=e["name"], variants=len(vs), jobs=jobs, runs=runs, est_hours=secs / 3600))
    return out


def run_campaign(cfg: dict, out_dir, workers: int = 1, limit: int | None = None, progress=print) -> dict:
    root, rows_dir, tau_path, man_path = _paths(cfg, out_dir)
    rows_dir.mkdir(parents=True, exist_ok=True)
    for tmp in rows_dir.glob("*.tmp"):               # restos de una ejecución interrumpida
        tmp.unlink()
    jobs = plan_jobs(cfg)
    pending = [j for j in jobs if not _job_file(rows_dir, j["job_id"]).exists()]
    cfg_hash = _hash(cfg, 16)
    manifest = json.loads(man_path.read_text()) if man_path.exists() else dict(
        campaign=cfg.get("campaign"), created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        orchestrator=ORCH_VERSION, versions=_versions(), git_commit=_git(),
        hardware=dict(platform=platform.platform(), cpu_count=os.cpu_count(),
                      affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None),
        env_blas={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        sessions=[])
    manifest["config"], manifest["config_hash"] = cfg, cfg_hash
    manifest["sessions"].append(dict(start_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     workers=workers, jobs_total=len(jobs), jobs_pending=len(pending)))
    man_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    progress(f"campaña '{cfg.get('campaign')}': {len(jobs)} trabajos, {len(jobs) - len(pending)} ya hechos, "
             f"{len(pending)} pendientes")
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return dict(ran=0, **load_rows(cfg, out_dir)[1])

    # 1) calibrar tau (una vez por escenario; caché en disco)
    cache = json.loads(tau_path.read_text()) if tau_path.exists() else {}
    seen = {}
    for j in pending:
        seen.setdefault(j["tau_key"], j["variant"])
    for key, variant in seen.items():
        if key not in cache:
            t0 = time.perf_counter()
            cache[key] = calibrate_tau(variant, cfg)
            tau_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            progress(f"  tau[{key}] = {cache[key]['tau']:.3f} (F1 cal. {cache[key]['f1_cal']:.3f}, "
                     f"{time.perf_counter() - t0:.0f}s)")
    for j in pending:
        j["tau"] = cache[j["tau_key"]]["tau"]

    # 2) ejecutar
    t_start, done = time.perf_counter(), 0
    executor_obj, pool = None, None
    if cfg.get("executor", "simulate") == "real":
        from runtime import WorkerPool
        if workers != 1:
            progress("ejecutor 'real': se fuerza --workers 1")
            workers = 1
        maxk = max(k for j in pending for k in j["variant"]["grid"]["k"])
        pool = WorkerPool(int(maxk))
        executor_obj = pool.executor
    specs = {j["job_id"]: j for j in pending}

    def finish(job_id, rows, dt):
        nonlocal done
        save_job(rows_dir, specs[job_id], rows, dt)
        done += 1
        el = time.perf_counter() - t_start
        progress(f"  [{done}/{len(pending)}] {specs[job_id]['experiment']} seed={specs[job_id]['seed']} "
                 f"({len(rows)} filas, {dt:.1f}s) | transcurrido {el:.0f}s, ETA {el / done * (len(pending) - done):.0f}s")

    try:
        if workers <= 1:
            for j in pending:
                rows, dt = run_job(j, executor_obj)
                finish(j["job_id"], rows, dt)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_worker_entry, j) for j in pending]
                for f in as_completed(futs):
                    finish(*f.result())
    finally:
        if pool is not None:
            pool.close()

    info = load_rows(cfg, out_dir)[1]
    if info["jobs_done"] == info["jobs_total"]:
        collect(cfg, out_dir, progress)
    return dict(ran=done, **info)


def collect(cfg: dict, out_dir, progress=print) -> Path:
    root, _, _, _ = _paths(cfg, out_dir)
    rows, info = load_rows(cfg, out_dir)
    path = root / "results.csv"
    write_rows_csv(rows, path)
    progress(f"results.csv: {len(rows)} filas de {info['jobs_done']}/{info['jobs_total']} trabajos"
             + (f" ({info['orphans']} archivos huérfanos ignorados)" if info["orphans"] else "") + f" -> {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Orquestador de campañas (Equipo T800)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status", "collect"):
        p = sub.add_parser(name)
        p.add_argument("config")
        p.add_argument("--out", default="./campaigns")
        if name == "run":
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--limit", type=int, default=None, help="ejecutar solo N trabajos pendientes")
            p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)

    if a.cmd == "run" and a.dry_run:
        print(f"{'experimento':22s} {'variantes':>9s} {'trabajos':>9s} {'corridas':>9s} {'horas aprox.':>12s}")
        tot = [0, 0, 0, 0.0]
        for d in describe(cfg):
            print(f"{d['experiment']:22s} {d['variants']:9d} {d['jobs']:9d} {d['runs']:9d} {d['est_hours']:12.2f}")
            tot = [tot[0] + d["variants"], tot[1] + d["jobs"], tot[2] + d["runs"], tot[3] + d["est_hours"]]
        print(f"{'TOTAL':22s} {tot[0]:9d} {tot[1]:9d} {tot[2]:9d} {tot[3]:12.2f}   (1 núcleo; heurística gruesa "
              f"{SEC_PER_NODE_RUN:g} s por nodo y corrida, sin incluir calibración)")
    elif a.cmd == "run":
        print(json.dumps(run_campaign(cfg, a.out, a.workers, a.limit)))
    elif a.cmd == "status":
        for k, v in status(cfg, a.out).items():
            print(f"{k:22s} {v['done']:5d}/{v['total']:<5d} {'✓' if v['done'] == v['total'] else ''}")
    else:
        collect(cfg, a.out)


if __name__ == "__main__":
    main()
