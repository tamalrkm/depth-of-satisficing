"""
Stage 3 (parallel): NUMA-bound multi-process Stockfish-18 driver.

Same science as engine_analysis.py (per move, per depth win-prob trajectory from
UCI_ShowWDL), but built for the 2x EPYC box:
  - Threads=1 per engine, MANY engines (Lazy SMP scales sublinearly; we want aggregate
    throughput over independent positions). Default workers = physical cores.
  - each worker is pinned (CPU affinity) to ONE NUMA node so memory stays node-local
    (via os.sched_setaffinity; for strict membind, launch the whole
    program under `numactl`, but per-worker affinity already gives node-local first-touch).
  - ONE persistent engine per worker, reused across all its FENs (never respawned per pos).
  - FEN-dedup before queueing; each worker writes a shard to data/shards/, then we concat.

Output schema matches engine_analysis.py (data/depth_traj.parquet), one row per
(position, move, depth):
    pos_id, fen, move, depth, winprob, is_played, is_topk_final

Run:
    python src/run_engine.py --config config.yaml [--workers 64] [--limit 300]
"""
import argparse
import glob
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import chess
import chess.engine
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from engine_analysis import winprob_from_info

COLS = ["pos_id", "fen", "move", "depth", "winprob", "is_played", "is_topk_final"]
FLUSH_SECS = 120     # checkpoint each worker's buffer at least this often (resume granularity)
FLUSH_ROWS = 50000   # ...or once the buffer hits this many rows (caps work-at-risk + memory)


def parse_cpus(spec):
    """'0-31,64-95' -> [0,1,...,31,64,...,95]"""
    cpus = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def analyse_fen(engine, fen, depth, multipv, max_nodes=0):
    """move -> {depth: winprob} for the multi-PV-across-depths search.

    We do NOT re-query a played move that fell outside the top-K (that per-blunder re-search
    was the dominant cost). build_dataset.py instead treats such moves as no better than the
    worst kept candidate at each depth -- "as bad as the K-th move".

    `max_nodes` (>0) is a per-position safety cap: the search stops at `depth` OR `max_nodes`
    nodes, whichever comes first. A minority of positions (quiet endgames) never converge at
    this depth/multipv and would run for hours, permanently trapping the worker (it holds one
    persistent engine and walks its chunk sequentially). Capped positions simply return with
    max(depth) < `depth`; build_dataset.py flags them by their reached depth (no schema change)."""
    board = chess.Board(fen)
    traj = {}
    limit = (chess.engine.Limit(depth=depth, nodes=max_nodes) if max_nodes
             else chess.engine.Limit(depth=depth))
    with engine.analysis(board, limit, multipv=multipv) as analysis:
        for info in analysis:
            if "pv" not in info or "depth" not in info:
                continue
            mv = info["pv"][0].uci()
            try:
                wp = winprob_from_info(info)
            except Exception:
                continue
            traj.setdefault(mv, {})[info["depth"]] = wp

    final_depth = max((max(ds) for ds in traj.values()), default=0)
    topk_final = {mv for mv, ds in traj.items() if final_depth in ds}
    return traj, topk_final


def _start_engine(ecfg):
    engine = chess.engine.SimpleEngine.popen_uci(os.path.expanduser(ecfg["path"]))
    engine.configure({"Threads": ecfg["threads"], "Hash": ecfg["hash_mb"], "NumaPolicy": "none"})
    if ecfg.get("show_wdl", True):
        try:
            engine.configure({"UCI_ShowWDL": True})
        except chess.engine.EngineError:
            pass
    return engine


def _worker(task):
    wid, fen_items, node_cpus, ecfg, shard_dir, run_tag = task
    if node_cpus:
        try:
            os.sched_setaffinity(0, set(node_cpus))
        except OSError:
            pass
    engine = _start_engine(ecfg)

    buf, seq, failed, n_done = [], 0, 0, 0
    t0 = last_flush = time.time()

    def flush():
        nonlocal seq
        if not buf:
            return
        path = os.path.join(shard_dir, f"part_{run_tag}_{wid:03d}_{seq:04d}.parquet")
        pd.DataFrame(buf, columns=COLS).to_parquet(path, index=False)
        buf.clear()
        seq += 1

    for i, (fen, members) in enumerate(fen_items):
        traj = topk = None
        # Engines occasionally die mid-search under sustained load; restart and retry once,
        # then skip the position rather than killing the whole run.
        for _ in range(2):
            try:
                traj, topk = analyse_fen(engine, fen, ecfg["depth"], ecfg["multipv"],
                                         ecfg.get("max_nodes", 0))
                break
            except (chess.engine.EngineError, BrokenPipeError, OSError):
                try:
                    engine.close()
                except Exception:
                    pass
                engine = _start_engine(ecfg)
                traj = topk = None
        if traj is None:
            failed += 1
            continue
        for pos_id, played in members:
            for mv, ds in traj.items():
                is_played = (mv == played)
                in_topk = mv in topk
                for d, wp in ds.items():
                    buf.append((pos_id, fen, mv, d, wp, is_played, in_topk))
        n_done += 1
        if wid == 0 and i == min(49, len(fen_items) - 1):
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"[worker0] {i + 1} FENs in {time.time() - t0:.1f}s "
                  f"= {rate:.2f} FEN/s/worker; est aggregate ~{rate * _N_WORKERS:.1f} FEN/s",
                  flush=True)
        # checkpoint for resume: on a timer, or whenever the buffer grows large. The flush
        # check runs between positions; the node cap (max_nodes) bounds per-position time so a
        # worker can no longer sit inside one search for hours with completed work stranded.
        if buf and (time.time() - last_flush > FLUSH_SECS or len(buf) > FLUSH_ROWS):
            flush()
            last_flush = time.time()
    flush()
    try:
        engine.quit()
    except Exception:
        pass
    return n_done, failed


_N_WORKERS = 1  # set in main(); read by workers (fork inherits the module global)


def _done_fens(shard_dir):
    """FENs already analysed in any existing part file (resume support)."""
    done = set()
    for sp in glob.glob(os.path.join(shard_dir, "*.parquet")):
        try:
            done.update(pq.read_table(sp, columns=["fen"]).column("fen").to_pylist())
        except Exception:
            pass
    return done


def _concat_shards(shard_dir, out_path):
    """Concatenate every part file into depth_traj (dedup on pos_id/move/depth)."""
    parts = glob.glob(os.path.join(shard_dir, "*.parquet"))
    if not parts:
        pd.DataFrame(columns=COLS).to_parquet(out_path, index=False)
        return 0, 0
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    df = df.drop_duplicates(subset=["pos_id", "move", "depth"])
    df.to_parquet(out_path, index=False)
    return len(df), df.pos_id.nunique()


def main(cfg, workers, limit, depth=None, multipv=None, fresh=False, max_nodes=None):
    global _N_WORKERS
    d, e, p = cfg["data"], cfg["engine"], dict(cfg["parallel"])
    e = dict(e)
    if depth:
        e["depth"] = depth
    if multipv:
        e["multipv"] = multipv
    if max_nodes is not None:
        e["max_nodes"] = max_nodes
    sel = pd.read_parquet(d["selected"])
    if limit:
        sel = sel.iloc[:limit]

    shard_dir = p["shard_dir"]
    os.makedirs(shard_dir, exist_ok=True)
    if fresh:
        for f in glob.glob(os.path.join(shard_dir, "*.parquet")):
            os.remove(f)

    # FEN-dedup: group positions by FEN, keep their (pos_id, played) members.
    fen_groups = defaultdict(list)
    for r in sel.itertuples():
        fen_groups[r.fen].append((r.pos_id, r.played_uci))
    fen_items = list(fen_groups.items())
    print(f"{len(sel)} positions -> {len(fen_items)} unique FENs "
          f"(dedup saved {len(sel) - len(fen_items)})")

    # resume: drop FENs already present in existing part files
    done = _done_fens(shard_dir)
    if done:
        before = len(fen_items)
        fen_items = [(f, m) for f, m in fen_items if f not in done]
        print(f"resume: {len(done)} FENs already analysed; {len(fen_items)}/{before} remaining")

    nfen = len(fen_items)
    if nfen:
        workers = workers or p["workers"]
        workers = max(1, min(workers, nfen))
        _N_WORKERS = workers

        numa = p.get("numa", {})
        node_lists = []
        if numa.get("enabled", False):
            node_lists = [parse_cpus(numa["node0_cpus"]), parse_cpus(numa["node1_cpus"])]

        # round-robin FENs into `workers` chunks (balances heavy/light positions across workers)
        chunks = [[] for _ in range(workers)]
        for idx, item in enumerate(fen_items):
            chunks[idx % workers].append(item)

        run_tag = str(int(time.time()))
        tasks = [(wid, chunk,
                  node_lists[wid % len(node_lists)] if node_lists else None,
                  dict(e), shard_dir, run_tag)
                 for wid, chunk in enumerate(chunks)]

        cap = e.get("max_nodes", 0)
        print(f"launching {workers} workers (Threads={e['threads']}/engine, "
              f"depth={e['depth']}, multipv={e['multipv']}, "
              f"max_nodes={cap or 'none'}, "
              f"numa={'on' if node_lists else 'off'}); checkpoint every {FLUSH_SECS}s")
        t0 = time.time()
        with Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks),
                                desc="engine workers"))
        dt = time.time() - t0
        ndone = sum(nd for nd, _ in results)
        nfailed = sum(f for _, f in results)
        print(f"analysed {ndone} FENs in {dt:.1f}s = {ndone / max(dt, 1e-6):.2f} FEN/s aggregate"
              + (f"  ({nfailed} skipped after engine restarts)" if nfailed else ""))
    else:
        print("nothing to analyse (all FENs already done); rebuilding depth_traj from shards")

    nrows, npos = _concat_shards(shard_dir, d["depth_traj"])
    print(f"depth_traj now: {nrows} rows over {npos} positions -> {d['depth_traj']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=0, help="override cfg.parallel.workers")
    ap.add_argument("--limit", type=int, default=0, help="cap #positions (0 = all)")
    ap.add_argument("--depth", type=int, default=0, help="override cfg.engine.depth")
    ap.add_argument("--multipv", type=int, default=0, help="override cfg.engine.multipv")
    ap.add_argument("--fresh", action="store_true", help="wipe data/shards/ and start over (ignore resume)")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="override cfg.engine.max_nodes (per-position node cap; 0 = no cap)")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.workers, a.limit, a.depth, a.multipv, a.fresh,
         a.max_nodes)
