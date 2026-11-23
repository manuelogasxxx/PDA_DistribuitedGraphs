#!/usr/bin/env python3
"""
runtime.py  (v1.0.0)  --  Paso 5: workers REALES (procesos del sistema operativo).

Qué es
  Ejecuta el mismo plan del paso 3/4 con k procesos reales. Cada worker recibe solo sus lecturas,
  intercambia el halo DIRECTAMENTE con los demás workers (una cola de entrada por worker) y calcula con
  detect.py. Lo que cambia respecto a distributed.py (simulación) es que ahora hay serialización real,
  transporte real y planificación real del sistema operativo.

Por qué multiprocessing y no Ray
  Ray no estaba disponible en el entorno de desarrollo (sin red) y no se escribe código que no se puede
  probar. El diseño no depende del backend: run_distributed(..., executor=pool.executor) acepta
  cualquier ejecutor que devuelva un SimResult, de modo que un backend de Ray puede añadirse después.

QUÉ SE MIDE (y qué valor tiene cada medida)
  bytes reales      tamaño de cada mensaje ya serializado (pickle). Siempre válido.
  tiempo de PARED   desde que se ordena la corrida hasta tener los resultados globales ensamblados
                    (rt_t_dist_wall). Solo es un speedup real si hay >= k núcleos libres; si k > núcleos
                    (rt_oversubscribed=True) los procesos se reparten el CPU y esta cifra NO es un speedup.
  tiempo de CPU     CPU consumida por cada worker durante la corrida (user+sys, incluye serialización y
                    E/S del transporte). No depende de cuántos núcleos haya. La estimación
                      rt_t_dist_cpu_est = max_q(CPU del worker q) + CPU de ensamblado en el agregador
                    es la ruta crítica con costos REALES y reparto ideal (sin esperas): cota inferior del
                    tiempo de pared con >= k núcleos. Es una ESTIMACIÓN, no un cronometraje en paralelo.
  Con 1 núcleo (como el entorno de desarrollo) solo son interpretables los bytes y los tiempos de CPU.

Hilos de BLAS: se fijan en 1 por proceso (OMP/OPENBLAS/MKL_NUM_THREADS) si numpy aún no se importó. Si
tu script importa numpy antes, exporta esas variables en el shell; si no, un worker multihilo y el
centralizado no son comparables. Se registra en rt_blas_threads.

Uso:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python runtime.py --graph ./data/hybrid_n10000_seed1 \
      --signals ./data/hybrid_n10000_seed1/signals_seed1_both_structured.npz \
      --methods hash community --k 2 4 8 --depth 0 2 --calibrate-ipc

Como librería:
  with WorkerPool(8) as pool:
      row = run_distributed(g, sig, ref, "community", k=8, depth=2, cfg=cfg, executor=pool.executor)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import pickle
import queue
import sys
import time
import traceback

import numpy as np

from detect import DetectorConfig
from distributed import (CostModel, SimResult, _worker_compute, prepare_reference, run_distributed,
                         write_rows_csv)
from iot_graph_generator import load_graph
from signals import load_signals

RUNTIME_VERSION = "1.0.0"
_PROTO = 5


# ---------------------------------------------------------------------------
# Proceso worker
# ---------------------------------------------------------------------------

def _execute(rank, st, inboxes, run_id):
    """Una corrida en un worker: enviar halo, recibir halo, calcular. Devuelve (bytes_resultado, stats)."""
    t_start, cpu_start = time.perf_counter(), time.process_time()
    w, Xo, Mo = st["plan"], st["X"], st["M"]
    T = Xo.shape[0]

    # A) El dueño extrae de SUS datos y envía (la cola serializa y escribe en un hilo aparte).
    sent = {}
    for dst, pos in st["send"].items():
        payload = pickle.dumps((Xo[:, pos], Mo[:, pos]), protocol=_PROTO)
        inboxes[dst].put((run_id, rank, payload))
        sent[dst] = len(payload)

    # B) Recibir el halo de quienes lo poseen.
    need, got, recv_bytes = set(w.recv_from), {}, {}
    inbox = inboxes[rank]
    while need:
        try:
            rid, src, payload = inbox.get(timeout=st["timeout"])
        except queue.Empty:
            raise TimeoutError(f"worker {rank}: halo de {sorted(need)} no llegó en {st['timeout']}s")
        if rid != run_id:
            continue
        got[src] = pickle.loads(payload)
        recv_bytes[src] = len(payload)
        need.discard(src)

    Xl = np.full((T, w.n_local), np.nan, dtype=np.float32)
    Ml = np.zeros((T, w.n_local), dtype=bool)
    Xl[:, :w.n_owned], Ml[:, :w.n_owned] = Xo, Mo
    for src, (Xm, Mm) in got.items():
        idx = w.n_owned + np.searchsorted(w.halo, w.recv_from[src])
        Xl[:, idx], Ml[:, idx] = Xm, Mm
    t_halo_ready = time.perf_counter() - t_start

    # C) Cómputo local. Se mide CPU y pared; con `repeats` se toma el mínimo de CPU.
    best, cpu_all = None, 0.0
    for _ in range(max(1, st["repeats"])):
        c0, w0 = time.process_time(), time.perf_counter()
        z_sp, z_tm, x_imp, phases = _worker_compute(Xl, Ml, w, st["cfg"], st["owned_only"], 1)
        c1, w1 = time.process_time() - c0, time.perf_counter() - w0
        cpu_all += c1
        if best is None or c1 < best[0]:
            best = (c1, w1, phases)
    cpu_compute, wall_compute, phases = best
    cpu_extra_repeats = cpu_all - cpu_compute          # repeticiones de cronometraje: no son trabajo de la corrida
    t_until_compute = t_halo_ready + wall_compute

    # D) Resultado.
    arrays = pickle.dumps((z_sp, z_tm, x_imp), protocol=_PROTO)
    stats = dict(sent=sent, recv=recv_bytes, t_halo_ready=t_halo_ready, t_until_compute=t_until_compute,
                 cpu_compute=cpu_compute, wall_compute=wall_compute, phases=phases,
                 cpu_total=time.process_time() - cpu_start - cpu_extra_repeats, result_bytes=len(arrays))
    return arrays, stats


def _worker_main(rank, ctrl_q, inboxes, result_q):
    state = None
    while True:
        msg = ctrl_q.get()
        kind = msg[0]
        if kind == "stop":
            return
        try:
            if kind == "setup":
                state = pickle.loads(msg[2])
                result_q.put(("ready", msg[1], rank, None))
            elif kind == "run":
                arrays, stats = _execute(rank, state, inboxes, msg[1])
                result_q.put(("result", msg[1], rank, (arrays, stats)))
            elif kind == "serve":                    # banco de pruebas del transporte: eco
                _, rid, reps = msg
                for _ in range(reps):
                    r, src, payload = inboxes[rank].get(timeout=60)
                    inboxes[src].put((r, rank, payload))
                result_q.put(("bench", rid, rank, None))
            elif kind == "ping":
                _, rid, partner, nbytes, reps = msg
                payload = bytes(nbytes)
                t0 = time.perf_counter()
                for _ in range(reps):
                    inboxes[partner].put((rid, rank, payload))
                    inboxes[rank].get(timeout=60)
                result_q.put(("bench", rid, rank, (time.perf_counter() - t0) / (2 * reps)))
        except BaseException:
            result_q.put(("error", msg[1] if len(msg) > 1 else -1, rank, traceback.format_exc()))


# ---------------------------------------------------------------------------
# Pool del lado del agregador
# ---------------------------------------------------------------------------

class WorkerPool:
    """k procesos persistentes (se crean una vez; cada corrida solo reenvía su parte de los datos)."""

    def __init__(self, size: int, start_method: str | None = None):
        method = start_method or ("fork" if sys.platform.startswith("linux") else "spawn")
        self.ctx = mp.get_context(method)
        self.size, self.start_method = size, method
        self.ctrl = [self.ctx.Queue() for _ in range(size)]
        self.inboxes = [self.ctx.Queue() for _ in range(size)]
        self.results = self.ctx.Queue()
        self.procs = [self.ctx.Process(target=_worker_main, daemon=True,
                                       args=(r, self.ctrl[r], self.inboxes, self.results))
                      for r in range(size)]
        t0 = time.perf_counter()
        for p in self.procs:
            p.start()
        self.t_startup = time.perf_counter() - t0
        self._run_id, self._broken = 0, False

    # -- utilidades --
    def _fail(self, msg):
        self._broken = True
        self.close(force=True)
        raise RuntimeError(msg)

    def _collect(self, kind, n, rid, timeout):
        out, deadline = {}, time.perf_counter() + timeout
        while len(out) < n:
            try:
                k, r, rank, payload = self.results.get(timeout=max(0.01, deadline - time.perf_counter()))
            except queue.Empty:
                self._fail(f"tiempo agotado esperando '{kind}' ({len(out)}/{n} workers)")
            if k == "error":
                self._fail(f"error en el worker {rank}:\n{payload}")
            if k == kind and r == rid:
                out[rank] = payload
        return out

    # -- una corrida --
    def run(self, X, M, plan, cfg: DetectorConfig, cost: CostModel = CostModel(), owned_only: bool = True,
            repeats: int = 1, timeout: float = 300.0) -> SimResult:
        if self._broken:
            raise RuntimeError("el pool falló antes; crea uno nuevo")
        k = plan.k
        if k > self.size:
            raise ValueError(f"el plan necesita {k} workers y el pool tiene {self.size}")
        X, M = np.asarray(X, dtype=np.float32), np.asarray(M, dtype=bool)
        T, n = X.shape
        self._run_id += 1
        rid = self._run_id

        # Reparto de los datos propios (los datos "nacen" en su dueño): se cronometra aparte.
        t0, scatter_bytes = time.perf_counter(), 0
        for q, w in enumerate(plan.workers):
            send = {}
            for dst, wd in enumerate(plan.workers):
                ids = wd.recv_from.get(q)
                if ids is not None and len(ids):
                    send[dst] = np.searchsorted(w.owned, ids)
            payload = pickle.dumps(dict(plan=w, send=send, X=X[:, w.owned], M=M[:, w.owned], cfg=cfg,
                                        owned_only=owned_only, repeats=repeats, timeout=timeout), protocol=_PROTO)
            scatter_bytes += len(payload)
            self.ctrl[q].put(("setup", rid, payload))
        self._collect("ready", k, rid, timeout)
        t_scatter = time.perf_counter() - t0

        # Corrida cronometrada.
        t0 = time.perf_counter()
        for q in range(k):
            self.ctrl[q].put(("run", rid))
        t_dispatch = time.perf_counter() - t0
        raw = self._collect("result", k, rid, timeout)
        t_arrived = time.perf_counter() - t0

        out_sp = np.zeros((T, n), dtype=np.float32)
        out_tm, out_im = np.zeros_like(out_sp), np.zeros_like(out_sp)
        cpu0 = time.process_time()
        for q, w in enumerate(plan.workers):
            z_sp, z_tm, x_im = pickle.loads(raw[q][0])
            out_sp[:, w.owned], out_tm[:, w.owned], out_im[:, w.owned] = z_sp, z_tm, x_im
        cpu_assembly = time.process_time() - cpu0
        t_assembled = time.perf_counter() - t0

        st = [raw[q][1] for q in range(k)]
        link_bytes = np.zeros((k, k), dtype=np.int64)
        link_msgs = np.zeros((k, k), dtype=np.int64)
        for q, s in enumerate(st):
            for dst, b in s["sent"].items():
                link_bytes[q, dst] += b
                link_msgs[q, dst] += 1
        recv_bytes, recv_msgs = link_bytes.sum(axis=0), link_msgs.sum(axis=0)
        gather_bytes = int(sum(s["result_bytes"] for s in st))
        t_worker = np.array([s["cpu_compute"] for s in st])      # CPU de cómputo: robusta a la contención
        t_comm_w = recv_bytes / cost.bandwidth_Bps + recv_msgs * cost.latency_s
        t_exchange = float(t_comm_w.max()) if k > 1 else 0.0
        t_gather = float(gather_bytes / cost.bandwidth_Bps + k * cost.latency_s) if k > 1 else 0.0
        cpu_max = max(s["cpu_total"] for s in st)
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)

        info = dict(
            k=k, depth=plan.depth, T=T,
            halo_bytes=int(link_bytes.sum()), halo_msgs=int(link_msgs.sum()),
            halo_bytes_max_worker=int(recv_bytes.max()), halo_msgs_max_worker=int(recv_msgs.max()),
            gather_bytes=gather_bytes,
            t_worker=t_worker, t_compute_max=float(t_worker.max()), t_compute_sum=float(t_worker.sum()),
            t_exchange_model=t_exchange, t_gather_model=t_gather,
            t_dist_model=t_exchange + float(t_worker.max()) + t_gather,
            t_dist_model_nogather=t_exchange + float(t_worker.max()),
            phase_mean={ph: float(np.mean([s["phases"][ph] for s in st]))
                        for ph in ("adjacency", "spatial", "temporal", "impute")},
            link_bytes=link_bytes, link_msgs=link_msgs,
            row_extra=dict(
                rt_t_dist_wall=t_assembled, rt_t_arrival_wall=t_arrived, rt_t_dispatch=t_dispatch,
                rt_t_scatter=t_scatter, rt_scatter_bytes=scatter_bytes,
                rt_t_exchange_wall=max(s["t_halo_ready"] for s in st),
                rt_t_until_compute_wall=max(s["t_until_compute"] for s in st) + t_dispatch,
                rt_cpu_max_worker=cpu_max, rt_cpu_sum=sum(s["cpu_total"] for s in st),
                rt_cpu_assembly=cpu_assembly, rt_t_dist_cpu_est=cpu_max + cpu_assembly,
                rt_msgs_max_worker=int(recv_msgs.max()),
                rt_halo_bytes_est=int(sum(len(w.halo) for w in plan.workers) * T * 5),   # modelo del paso 3
                rt_cpu_count=cores, rt_oversubscribed=bool(k > cores),
                rt_blas_threads=os.environ.get("OPENBLAS_NUM_THREADS", "?"),
                rt_start_method=self.start_method),
        )
        return SimResult(out_sp, out_tm, np.maximum(out_sp, out_tm), out_im, info)

    def executor(self, sig, plan, cfg, cost, owned_only, repeats):
        """Adaptador para run_distributed(..., executor=pool.executor)."""
        X = sig["X"] if isinstance(sig, dict) else sig.X
        M = sig["mask"] if isinstance(sig, dict) else sig.mask
        return self.run(X, M, plan, cfg, cost, owned_only, repeats)

    # -- transporte --
    def calibrate_ipc(self, sizes=(1_000, 10_000, 100_000, 1_000_000, 8_000_000), reps=(40, 40, 20, 10, 4),
                      trials: int = 5, timeout: float = 120.0) -> CostModel:
        """Mide el costo de un mensaje (ida) entre dos workers y estima latencia y ancho de banda del
        TRANSPORTE LOCAL del runtime (no de una red real).
          latencia       = mínimo entre los mensajes pequeños (<= 10 KB)
          ancho de banda = 1 / pendiente entre los dos tamaños mayores
        Ambas con el MÍNIMO de `trials` intentos por tamaño (el ruido del planificador solo suma tiempo)."""
        if self.size < 2:
            raise ValueError("hace falta un pool de al menos 2 workers")
        per = [[] for _ in sizes]
        for _ in range(trials):
            for i, (nbytes, r) in enumerate(zip(sizes, reps)):
                self._run_id += 1
                rid = self._run_id
                self.ctrl[1].put(("serve", rid, r))
                self.ctrl[0].put(("ping", rid, 1, nbytes, r))
                per[i].append(self._collect("bench", 2, rid, timeout)[0])
        med = [float(np.min(p)) for p in per]
        self.ipc_points = list(zip(sizes, med))
        latency = float(np.min([m for sz, m in self.ipc_points if sz <= 10_000]))
        slope = (med[-1] - med[-2]) / (sizes[-1] - sizes[-2])
        if slope <= 0:                                   # medición degenerada: ajuste global
            slope = max(float(np.polyfit(sizes, med, 1)[0]), 1e-12)
        return CostModel(bandwidth_Bps=float(1.0 / slope), latency_s=latency)

    # -- cierre --
    def close(self, force: bool = False):
        for q, p in zip(self.ctrl, self.procs):
            if p.is_alive() and not force:
                try:
                    q.put(("stop",))
                except Exception:
                    pass
        for p in self.procs:
            p.join(timeout=0.5 if not force else 0.0)
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _model_time(row, cost: CostModel) -> float:
    """Tiempo modelado con otro CostModel, a partir de los bytes y el cómputo medidos en la fila."""
    k = row["k"]
    if k == 1:
        return row["t_compute_max"]
    ex = row["halo_bytes_max_worker"] / cost.bandwidth_Bps + row["rt_msgs_max_worker"] * cost.latency_s
    ga = row["gather_bytes"] / cost.bandwidth_Bps + k * cost.latency_s
    return ex + row["t_compute_max"] + ga


def main() -> None:
    ap = argparse.ArgumentParser(description="Workers reales (procesos) sobre un plan de particiones (T800)")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--methods", nargs="+", default=["hash", "community"])
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--depth", type=int, nargs="+", default=None)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--impute-iters", type=int, default=2)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--calibrate-ipc", action="store_true", help="medir el transporte local y usarlo en el modelo")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    g, s = load_graph(a.graph), load_signals(a.signals)
    cfg = DetectorConfig(hops=a.hops, impute_iters=a.impute_iters)
    ref = prepare_reference(g, s, cfg, tau=a.tau, repeats=3)
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    print(f"núcleos disponibles: {cores} | centralizado: {ref.t_central:.3f}s | tau={ref.tau:.2f} | "
          f"halo requerido={cfg.halo_depth}")
    if max(a.k) > cores:
        print(f"AVISO: k máximo ({max(a.k)}) > núcleos ({cores}): 'speedup pared' NO es un speedup real; "
              f"interpreta bytes y la estimación por CPU.")

    with WorkerPool(max(a.k)) as pool:
        cost = CostModel()
        if a.calibrate_ipc:
            cost = pool.calibrate_ipc()
            print(f"transporte local calibrado: ancho de banda efectivo {cost.bandwidth_Bps / 1e6:.0f} MB/s, "
                  f"latencia {cost.latency_s * 1e6:.0f} µs por mensaje")
        print(f"\n{'método':10s} {'k':>2s} {'halo':>4s} {'MB halo':>8s} {'real/est.':>9s} {'pared(s)':>8s} "
              f"{'CPU est(s)':>10s} {'sp.pared':>8s} {'sp.CPU*':>8s} {'sp.modelo':>9s} {'err.mod':>8s} equiv")
        rows = []
        for method in a.methods:
            for k in a.k:
                for depth in (a.depth if a.depth is not None else [cfg.halo_depth]):
                    r = run_distributed(g, s, ref, method, k, depth, cfg, a.seed, cost, repeats=a.repeats,
                                        executor=pool.executor)
                    rows.append(r)
                    est = r["rt_halo_bytes_est"]
                    ratio = r["halo_bytes"] / est if est else 1.0
                    print(f"{method:10s} {k:2d} {depth:4d} {r['halo_bytes'] / 1e6:8.2f} {ratio:9.4f} "
                          f"{r['rt_t_dist_wall']:8.3f} {r['rt_t_dist_cpu_est']:10.3f} {r['speedup_real_wall']:8.2f} "
                          f"{r['speedup_cpu_est']:8.2f} {ref.t_central / r['t_dist_model']:9.2f} "
                          f"{100 * r['model_error_rel_cpu']:+7.1f}% {'SI' if r['equivalent'] else 'no'}")
    print("\n* sp.CPU = estimación con CPU real por worker (ruta crítica, reparto ideal). "
          "sp.pared = tiempo de pared medido (solo válido si núcleos >= k).")
    if a.csv:
        write_rows_csv(rows, a.csv)
        print(f"Filas guardadas en {a.csv}")


if __name__ == "__main__":
    main()
