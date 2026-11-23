#!/usr/bin/env python3
"""
analysis.py  (v1.0.0)  --  Paso 6: análisis estadístico y figuras a partir de results.csv.

Uso:
  python analysis.py --results campaigns/campaign_full/results.csv --out analysis/

Genera en --out:
  summary.csv        media, desviación, mediana e IC 95 % bootstrap por (caso, método, k, profundidad)
  comparisons.csv    comparaciones pareadas entre métodos: Wilcoxon + Holm, delta de Cliff, Hodges-Lehmann
  equivalence.csv    TOST de la pérdida de F1 sin halo, con margen ±2.5 pp y ±0.5 pp
  loglog_fit.json    ajuste |ΔF1| = a * corte^b (con IC bootstrap del exponente)
  figures/*.png      F2 a F10 del plan (las que tengan datos)
  report.md          cifras principales y lista de archivos

Convenciones (las del plan de pruebas)
  * La unidad experimental es la SEMILLA (graph_seed): las comparaciones son pareadas por semilla.
  * ΔF1 se expresa en puntos porcentuales (d_f1_pp = 100 * d_f1).
  * Un "caso" es la combinación de parámetros de datos y detector; no se mezclan casos distintos.
  * speedup_model* es MODELADO (cota ideal); halo_bytes es medido.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CASE_COLS = ["experiment", "n", "spread_factor", "anomaly", "missing", "magnitude_lo", "magnitude_hi",
             "anomaly_rate", "anomaly_placement", "hops", "impute_iters"]
METRICS = ["d_f1_pp", "d_auc", "d_mae", "halo_mb", "replication_factor", "cut_ratio", "speedup_model",
           "speedup_model_nogather", "efficiency_model", "work_inflation"]
MARGIN_PP, STRICT_PP = 2.5, 0.5


# ---------------------------------------------------------------------------
# Estadística
# ---------------------------------------------------------------------------

def bootstrap_ci(x, stat: str = "mean", n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """IC percentil por bootstrap de la media o la mediana (remuestreo de semillas)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.mean(x[idx], axis=1) if stat == "mean" else np.median(x[idx], axis=1)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def holm(pvals) -> np.ndarray:
    """Corrección de Holm-Bonferroni (valores p ajustados, monótonos, tope 1). Respeta los NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pv[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def cliffs_delta(x, y) -> float:
    """δ de Cliff en [-1, 1]: P(x>y) - P(x<y). 0 = sin diferencia; ±1 = separación total."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def hodges_lehmann(d) -> float:
    """Estimador de Hodges-Lehmann: mediana de los promedios de Walsh de las diferencias pareadas."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def paired_wilcoxon(a, b) -> float:
    """Valor p del Wilcoxon de rangos con signo (pareado). Todas las diferencias nulas => p = 1."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return float("nan")
    if np.allclose(a, b, atol=1e-12):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def tost_ci(d, margin: float, alpha: float = 0.05) -> dict:
    """Equivalencia (TOST) para la media de diferencias pareadas `d` con margen ±margin.
    Equivalente = el IC de (1 - 2·alpha) (90 % con alpha=0.05) cae dentro de (-margin, +margin)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float(d.mean()) if n else np.nan, lo=np.nan, hi=np.nan, p_tost=np.nan,
                    equivalent=False)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    if se == 0:
        lo = hi = mean
        p = 0.0 if abs(mean) < margin else 1.0
    else:
        tcrit = stats.t.ppf(1 - alpha, n - 1)
        lo, hi = mean - tcrit * se, mean + tcrit * se
        p = max(stats.t.sf((mean + margin) / se, n - 1), stats.t.cdf((mean - margin) / se, n - 1))
    return dict(n=n, mean=float(mean), lo=float(lo), hi=float(hi), p_tost=float(p),
                equivalent=bool(lo > -margin and hi < margin))


def loglog_fit(x, y, n_boot: int = 2000, seed: int = 0) -> dict:
    """Ajuste y = a * x^b por mínimos cuadrados en log-log, con IC 95 % bootstrap del exponente."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(np.log(x)) == 0:
        return dict(n=int(len(x)), a=np.nan, b=np.nan, r2=np.nan, b_lo=np.nan, b_hi=np.nan)
    lx, ly = np.log(x), np.log(y)
    b, la = np.polyfit(lx, ly, 1)
    r2 = float(np.corrcoef(lx, ly)[0, 1] ** 2)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(lx[i]) > 0:
            bs.append(np.polyfit(lx[i], ly[i], 1)[0])
    lo, hi = (np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan))
    return dict(n=int(len(x)), a=float(np.exp(la)), b=float(b), r2=r2, b_lo=float(lo), b_hi=float(hi))


def pareto_front(cost, loss) -> np.ndarray:
    """Máscara de los puntos NO dominados (menor costo y menor pérdida son mejores)."""
    cost, loss = np.asarray(cost, dtype=float), np.asarray(loss, dtype=float)
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        dom = (cost <= cost[i]) & (loss <= loss[i]) & ((cost < cost[i]) | (loss < loss[i]))
        if dom.any():
            keep[i] = False
    return keep


# ---------------------------------------------------------------------------
# Datos y tablas
# ---------------------------------------------------------------------------

def load_results(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "experiment" not in df:
        df["experiment"] = "pilot"
    df["d_f1_pp"] = df["d_f1"] * 100.0
    df["halo_mb"] = df["halo_bytes"] / 1e6
    if "equivalent" in df:
        df["equivalent"] = df["equivalent"].astype(str).str.lower().eq("true")
    return df


def case_cols(df) -> list[str]:
    return [c for c in CASE_COLS if c in df.columns and df[c].nunique(dropna=False) >= 1]


def summarize(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    keys = case_cols(df) + ["method", "k", "depth"]
    rows = []
    for key, g in df.groupby(keys, dropna=False, sort=True):
        rec = dict(zip(keys, key))
        rec["n_seeds"] = g["graph_seed"].nunique()
        for m in [m for m in METRICS if m in g]:
            x = g[m].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x, "mean", n_boot)
            rec.update({f"{m}_mean": np.nanmean(x), f"{m}_sd": np.nanstd(x, ddof=1) if len(x) > 1 else 0.0,
                        f"{m}_median": np.nanmedian(x), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_methods(df: pd.DataFrame, metrics=("d_f1_pp", "d_mae", "halo_mb", "speedup_model")) -> pd.DataFrame:
    """Todas las parejas de métodos, por (caso, k, profundidad), pareadas por semilla. Holm por (experimento, métrica)."""
    keys = case_cols(df) + ["k", "depth"]
    out = []
    for key, g in df.groupby(keys, dropna=False, sort=True):
        for metric in [m for m in metrics if m in g]:
            piv = g.pivot_table(index="graph_seed", columns="method", values=metric, aggfunc="first")
            for m1, m2 in combinations(sorted(piv.columns), 2):
                a, b = piv[m1].to_numpy(), piv[m2].to_numpy()
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 2:
                    continue
                out.append(dict(zip(keys, key), metric=metric, method_a=m1, method_b=m2, n_pairs=int(ok.sum()),
                                mean_diff=float(np.mean(a[ok] - b[ok])), hodges_lehmann=hodges_lehmann(a[ok] - b[ok]),
                                cliffs_delta=cliffs_delta(a[ok], b[ok]), p_wilcoxon=paired_wilcoxon(a, b)))
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["p_holm"] = np.nan
    for _, idx in res.groupby(["experiment", "metric"]).groups.items():
        res.loc[idx, "p_holm"] = holm(res.loc[idx, "p_wilcoxon"].to_numpy())
    return res


def equivalence_table(df: pd.DataFrame, depth: int = 0) -> pd.DataFrame:
    """TOST de ΔF1 (pp) sin halo (o con la profundidad indicada), por caso, método y k."""
    keys = case_cols(df) + ["method", "k"]
    out = []
    for key, g in df[df["depth"] == depth].groupby(keys, dropna=False, sort=True):
        d = g.sort_values("graph_seed")["d_f1_pp"].to_numpy()
        t = tost_ci(d, MARGIN_PP)
        s = tost_ci(d, STRICT_PP)
        out.append(dict(zip(keys, key), depth=depth, n=t["n"], mean_pp=t["mean"], ci90_lo=t["lo"], ci90_hi=t["hi"],
                        p_tost_25=t["p_tost"], equivalent_25pp=t["equivalent"], equivalent_05pp=s["equivalent"]))
    return pd.DataFrame(out)


def network_sensitivity(df: pd.DataFrame, bandwidths_mbps=(100, 1000, 10000), latencies_ms=(0.1, 0.5, 5.0),
                        gather: bool = True) -> pd.DataFrame:
    """Recalcula el speedup MODELADO con otros parámetros de red a partir de los bytes, mensajes y cómputo medidos."""
    need = {"t_central", "halo_bytes_max_worker", "halo_msgs_max_worker", "gather_bytes", "t_compute_max", "k"}
    if not need <= set(df.columns):
        return pd.DataFrame()
    rows = []
    for bw in bandwidths_mbps:
        for lat in latencies_ms:
            B, L = bw * 1e6 / 8.0, lat / 1e3
            ex = np.where(df["k"] > 1, df["halo_bytes_max_worker"] / B + df["halo_msgs_max_worker"] * L, 0.0)
            ga = np.where(df["k"] > 1, df["gather_bytes"] / B + df["k"] * L, 0.0) if gather else 0.0
            t = ex + df["t_compute_max"] + ga
            tmp = df[case_cols(df) + ["method", "k", "depth", "graph_seed"]].copy()
            tmp["bandwidth_mbps"], tmp["latency_ms"] = bw, lat
            tmp["speedup"] = df["t_central"] / t
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _sel(df, prefix):
    return df[df["experiment"].astype(str).str.startswith(prefix)]


def fig_equivalence_vs_depth(df, out: Path):
    d = df[df["max_abs_diff_spatial"].notna()]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in d.groupby("method"):
        s = g.groupby("depth")["max_abs_diff_spatial"].median().clip(lower=1e-6)
        ax.plot(s.index, s.values, marker="o", label=m)
    ax.axhline(1e-3, ls="--", c="gray", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Profundidad de halo (saltos)")
    ax.set_ylabel("Diferencia máx. vs centralizado (mediana)")
    ax.legend()
    return _save(fig, out / "F2_equivalencia_vs_profundidad.png")


def fig_pareto(df, out: Path):
    d = df[df["halo_mb"].notna()]
    if d.empty:
        return None
    g = d.groupby(["method", "depth"]).agg(mb=("halo_mb", "mean"), f1=("d_f1_pp", "mean"),
                                           sd=("d_f1_pp", "std")).reset_index().fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m, s in g.groupby("method"):
        ax.errorbar(s["mb"], s["f1"], yerr=s["sd"], marker="o", capsize=2, label=m)
        for _, r in s.iterrows():
            ax.annotate(str(int(r["depth"])), (r["mb"], r["f1"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax.axhspan(-MARGIN_PP, MARGIN_PP, alpha=0.1, color="green", label=f"±{MARGIN_PP} pp")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlabel("Bytes de halo (MB, escala symlog); etiqueta = profundidad")
    ax.set_ylabel("ΔF1 vs centralizado (pp)")
    ax.legend(fontsize=8)
    return _save(fig, out / "F3_pareto_bytes_vs_F1.png")


def fig_loglog(df, out: Path):
    d = df[(df["depth"] == 0) & (df["method"] != "centralized")]
    fit = loglog_fit(d["cut_ratio"], d["d_f1_pp"].abs().clip(lower=1e-4))
    if not np.isfinite(fit["b"]):
        return None, fit
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m, g in d.groupby("method"):
        ax.scatter(g["cut_ratio"], g["d_f1_pp"].abs().clip(lower=1e-4), s=14, alpha=0.7, label=m)
    xs = np.geomspace(d["cut_ratio"].clip(lower=1e-4).min(), d["cut_ratio"].max(), 50)
    ax.plot(xs, fit["a"] * xs ** fit["b"], "k--", label=f"|ΔF1|={fit['a']:.1f}·corte^{fit['b']:.2f} (R²={fit['r2']:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Fracción de aristas cortadas"); ax.set_ylabel("|ΔF1| sin halo (pp)")
    ax.legend(fontsize=7)
    return _save(fig, out / "F4_perdida_vs_corte.png"), fit


def fig_speedup_vs_k(df, out: Path, name="F5_speedup_vs_k.png"):
    d = df[(df["depth"] == df["required_depth"]) & df["speedup_model"].notna()] if "required_depth" in df else df
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, col, title in zip(axes, ["speedup_model", "speedup_model_nogather"], ["con recolección", "sin recolección"]):
        for m, g in d.groupby("method"):
            s = g.groupby("k")[col].mean()
            ax.plot(s.index, s.values, marker="o", label=m)
        ax.plot(sorted(d["k"].unique()), sorted(d["k"].unique()), "k:", lw=1, label="ideal")
        ax.set_title(f"Speedup modelado ({title})"); ax.set_xlabel("k (workers)"); ax.set_ylabel("speedup")
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    axes[0].legend(fontsize=8)
    return _save(fig, out / name)


def fig_speedup_vs_n(df, out: Path):
    d = df[(df["depth"] == df["required_depth"]) & df["speedup_model"].notna()] if "required_depth" in df else df
    if d["n"].nunique() < 2:
        return None
    kk = d["k"].max()
    d = d[d["k"] == kk]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m, g in d.groupby("method"):
        s = g.groupby("n")["speedup_model"].mean()
        ax.plot(s.index, s.values, marker="o", label=m)
    ax.axhline(1.0, c="gray", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("n (nodos)"); ax.set_ylabel(f"Speedup modelado (k={kk})")
    ax.legend(fontsize=8)
    return _save(fig, out / "F6_speedup_vs_n.png")


def fig_overlap(df, out: Path):
    d = _sel(df, "E3")
    if d.empty or d["spread_factor"].nunique() < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    req = d[d["depth"] == d["required_depth"]]
    for m, g in d.groupby("method"):
        axes[0].plot(*zip(*sorted(g.groupby("spread_factor")["cut_ratio"].mean().items())), marker="o", label=m)
    for m, g in req.groupby("method"):
        axes[1].plot(*zip(*sorted(g.groupby("spread_factor")["halo_mb"].mean().items())), marker="o", label=m)
    for m, g in d[d["depth"] == 0].groupby("method"):
        axes[2].plot(*zip(*sorted(g.groupby("spread_factor")["d_f1_pp"].mean().items())), marker="o", label=m)
    for ax, t in zip(axes, ["Corte de aristas", "Halo a profundidad requerida (MB)", "ΔF1 sin halo (pp)"]):
        ax.set_xlabel("spread_factor (solape de comunidades)"); ax.set_title(t)
    axes[1].set_yscale("symlog", linthresh=0.5)
    axes[0].legend(fontsize=7)
    return _save(fig, out / "F7_solape_de_comunidades.png")


def fig_heatmap(df, out: Path, prefix="E4_", name="F8_mapa_calor_escenarios.png"):
    d = _sel(df, prefix)
    d = d[d["depth"] == 0]
    if d.empty:
        return None
    d = d.assign(escenario=d["anomaly"].astype(str) + " / " + d["missing"].astype(str) + " / mag"
                 + d["magnitude_lo"].astype(int).astype(str) + "-" + d["magnitude_hi"].astype(int).astype(str))
    piv = d.pivot_table(index="escenario", columns="method", values="d_f1_pp", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.6 * len(piv.columns) + 3, 0.45 * len(piv) + 2))
    vmax = max(abs(np.nanmin(piv.values)), abs(np.nanmax(piv.values)), 1e-6)
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right"); ax.set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="ΔF1 sin halo (pp)")
    return _save(fig, out / name)


def fig_boundary_breakdown(df, out: Path):
    d = df[(df["depth"] == 0) & df["d_auc_boundary"].notna()]
    if d.empty:
        return None
    g = d.groupby("method")[["d_auc_boundary", "d_auc_interior"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["d_auc_boundary"], 0.4, label="nodos de frontera")
    ax.bar(x + 0.2, g["d_auc_interior"].fillna(0), 0.4, label="nodos interiores")
    ax.set_xticks(x, g.index, rotation=20); ax.set_ylabel("ΔAUC sin halo"); ax.legend()
    return _save(fig, out / "F9_frontera_vs_interior.png")


def fig_network(df, out: Path):
    ns = network_sensitivity(df)
    if ns.empty:
        return None
    ns = ns[(ns["depth"] == ns.get("required_depth", ns["depth"]))] if "required_depth" in ns else ns
    kk = ns["k"].max()
    ns = ns[ns["k"] == kk]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for (m, lat), g in ns.groupby(["method", "latency_ms"]):
        s = g.groupby("bandwidth_mbps")["speedup"].mean()
        ax.plot(s.index, s.values, marker="o", ls={0.1: "-", 0.5: "--", 5.0: ":"}.get(lat, "-"),
                label=f"{m}, {lat:g} ms")
    ax.set_xscale("log"); ax.set_xlabel("Ancho de banda (Mbit/s)"); ax.set_ylabel(f"Speedup modelado (k={kk})")
    ax.legend(fontsize=6, ncol=2)
    return _save(fig, out / "F10_sensibilidad_de_red.png")


def fig_adversarial(df, out: Path):
    d = _sel(df, "E4b")
    d = d[d["depth"] == 0]
    if d.empty:
        return None
    d = d.assign(adversario=d["anomaly_placement"].astype(str))
    grp = d.groupby(["adversario", "method"])["d_f1_pp"]
    piv, err = grp.mean().unstack("method"), grp.std().unstack("method").fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot.bar(ax=ax, yerr=err, capsize=2)
    ax.axhspan(-MARGIN_PP, MARGIN_PP, alpha=0.1, color="green")
    ax.set_ylabel("ΔF1 sin halo (pp)"); ax.set_xlabel(""); ax.tick_params(axis="x", labelrotation=15)
    ax.legend(fontsize=7)
    return _save(fig, out / "F11_prueba_adversa.png")


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def make_report(results_csv, out_dir, n_boot: int = 2000) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(results_csv)
    written = []

    summ = summarize(df, n_boot)
    summ.to_csv(out / "summary.csv", index=False)
    comp = compare_methods(df)
    comp.to_csv(out / "comparisons.csv", index=False)
    eq = equivalence_table(df, 0)
    eq.to_csv(out / "equivalence.csv", index=False)
    written += ["summary.csv", "comparisons.csv", "equivalence.csv"]

    figs = out / "figures"
    produced = [fig_equivalence_vs_depth(df, figs), fig_pareto(_sel(df, "E1") if len(_sel(df, "E1")) else df, figs),
                fig_speedup_vs_k(df, figs), fig_speedup_vs_n(df, figs), fig_overlap(df, figs),
                fig_heatmap(df, figs), fig_heatmap(df, figs, "E4b", "F8b_mapa_calor_adversario.png"),
                fig_boundary_breakdown(df, figs), fig_network(df, figs), fig_adversarial(df, figs)]
    ll = fig_loglog(df, figs)
    fit = ll[1] if isinstance(ll, tuple) else None
    produced.append(ll[0] if isinstance(ll, tuple) else None)
    if fit is not None:
        (out / "loglog_fit.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
        written.append("loglog_fit.json")
    written += [str(p.relative_to(out)) for p in produced if p]

    # --- cifras principales ---
    lines = [f"# Informe de análisis\n", f"Filas: {len(df)} | experimentos: {', '.join(sorted(df['experiment'].astype(str).unique()))}\n"]
    if "equivalent" in df and "required_depth" in df:
        req = df[df["depth"] >= df["required_depth"]]
        lines.append(f"**H1 (equivalencia con halo suficiente):** {int(req['equivalent'].sum())} de {len(req)} corridas "
                     f"con profundidad ≥ requerida son equivalentes al centralizado.\n")
    if not eq.empty:
        adv = eq["experiment"].astype(str).str.startswith("E4b")
        ok = eq[(eq["method"] != "hash") & ~adv]
        if len(ok):
            lines.append(f"**H2 (±{MARGIN_PP} pp sin halo, anomalías no adversas, sin `hash`):** "
                         f"{int(ok['equivalent_25pp'].sum())} de {len(ok)} combinaciones (caso × método × k) dentro del margen; "
                         f"{int(ok['equivalent_05pp'].sum())} dentro del margen estricto ±{STRICT_PP} pp.\n")
        h = eq[adv & (eq["method"] != "hash")]
        if len(h):
            out_ = h[~h["equivalent_25pp"]]
            lines.append(f"**H8 (prueba adversa E4b, sin `hash`):** {len(out_)} de {len(h)} combinaciones salen del margen "
                         f"±{MARGIN_PP} pp sin halo.\n")
        lines.append(f"*Nota:* con pocas semillas (n < 10) los IC son amplios; los veredictos solo son definitivos con las "
                     f"30 semillas del plan.\n")
    if fit is not None and np.isfinite(fit["b"]):
        lines.append(f"**H3 (ley de potencia):** |ΔF1| ≈ {fit['a']:.2f}·corte^{fit['b']:.2f} "
                     f"(R²={fit['r2']:.2f}, IC95 exponente [{fit['b_lo']:.2f}, {fit['b_hi']:.2f}], n={fit['n']}). "
                     f"Exploratorio si los puntos forman solo dos grupos.\n")
    lines.append("\n## Archivos\n" + "\n".join(f"- `{w}`" for w in written))
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(rows=len(df), files=written, fit=fit)


def main() -> None:
    ap = argparse.ArgumentParser(description="Análisis estadístico y figuras (Equipo T800)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="./analysis")
    ap.add_argument("--boot", type=int, default=2000, help="remuestreos bootstrap")
    a = ap.parse_args()
    info = make_report(a.results, a.out, a.boot)
    print(f"{info['rows']} filas analizadas. Archivos en {a.out}:")
    for f in info["files"]:
        print("  ", f)


if __name__ == "__main__":
    main()
