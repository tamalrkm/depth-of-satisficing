"""
Stage 8: synthetic agents that satisfice at a KNOWN depth (ground-truth for E6/Fig6).

Each agent plays every move by a soft-max over Stockfish depth-d_plant win-probs (inverse
temperature beta_gen). We then run the SAME pipeline (engine -> maia -> build_dataset) on
these games and check the model's estimated depth-of-satisficing recovers d_plant.

Two design choices that make the recovery a clean identifiability test, not a metadata
leak:
  * elo and clock are held CONSTANT across planted depths (NEUTRAL_ELO, classical), so the
    model cannot read depth off the context -- E[d] must come from the move/regret structure.
  * a per-position node cap (engine.max_nodes) mirrors the real run.

Generation is parallel: one persistent engine per worker process. Output is a
selected.parquet-schema table (so it slots straight into run_engine/maia/build_dataset via a
synthetic config), one row per decision, with an extra `planted_depth` column.

Run:
    python src/synthetic_agents.py --config config.yaml [--games 40] [--workers 64] \
        --out data/synthetic/selected.parquet
"""
import argparse
import math
import os
import random
from multiprocessing import Pool

import chess
import chess.engine
import pandas as pd
import yaml

NEUTRAL_ELO = 2000          # same for every agent: depth must not be readable from context
_ENG = None                 # per-process engine (set in _init)
_CFG = None


def _winprob(info):
    wdl = info.get("wdl")
    if wdl is not None:
        rel = wdl.relative
        return (rel.wins + 0.5 * rel.draws) / 1000.0
    return info["score"].relative.wdl(model="sf").expectation()


def _init(ecfg):
    global _ENG, _CFG
    _CFG = ecfg
    _ENG = chess.engine.SimpleEngine.popen_uci(os.path.expanduser(ecfg["path"]))
    _ENG.configure({"Threads": 1, "Hash": ecfg["hash_mb"], "NumaPolicy": "none"})
    try:
        _ENG.configure({"UCI_ShowWDL": True})
    except chess.engine.EngineError:
        pass


def _agent_move(board, d_plant, beta_gen, multipv, max_nodes, rng):
    """Soft-max over depth-d_plant win-probs (side-to-move POV)."""
    vals = {}
    lim = (chess.engine.Limit(depth=d_plant, nodes=max_nodes) if max_nodes
           else chess.engine.Limit(depth=d_plant))
    with _ENG.analysis(board, lim, multipv=multipv) as an:
        for info in an:
            if info.get("depth") == d_plant and "pv" in info:
                vals[info["pv"][0]] = _winprob(info)
    if not vals:
        return rng.choice(list(board.legal_moves))
    moves, ws = zip(*vals.items())
    z = [math.exp(beta_gen * w) for w in ws]
    s = sum(z)
    r, acc = rng.random() * s, 0.0
    for mv, zz in zip(moves, z):
        acc += zz
        if acc >= r:
            return mv
    return moves[-1]


def _play_game(task):
    """Play one game; return rows in selected.parquet schema (one per decision)."""
    d_plant, gidx, beta_gen, multipv, max_nodes, max_ply, min_ply = task
    rng = random.Random(1000 * d_plant + gidx)        # deterministic per (depth, game)
    board = chess.Board()
    game_id = f"syn_d{d_plant}_g{gidx}"
    hist, rows = [], []
    while not board.is_game_over() and board.ply() < max_ply:
        fen = board.fen()
        mv = _agent_move(board, d_plant, beta_gen, multipv, max_nodes, rng)
        uci = mv.uci()
        ply = board.ply()
        if ply >= min_ply:                            # skip opening book churn, like the real run
            rows.append(dict(
                pos_id=f"{game_id}:{ply}", game_id=game_id, ply=ply, fen=fen,
                side_to_move_elo=NEUTRAL_ELO, oppo_elo=NEUTRAL_ELO, time_class="classical",
                base_time=float("nan"), increment=float("nan"),
                clock_before=float("nan"), clock_after=float("nan"), time_spent=float("nan"),
                lichess_eval=float("nan"), played_uci=uci, player=game_id,
                hist_uci=" ".join(hist), source="synthetic", planted_depth=d_plant))
        hist.append(uci)
        board.push(mv)
    return rows


def main(cfg, games, workers, out):
    s = cfg["synthetic"]
    e = cfg["engine"]
    games = games or s["games_per_depth"]
    depths = s["planted_depths"]
    beta_gen = s["softmax_beta_gen"]
    multipv = e["multipv"]
    max_nodes = e.get("max_nodes", 0)
    max_ply = cfg["data"]["max_ply"]
    min_ply = cfg["data"]["min_ply"]
    tasks = [(d, g, beta_gen, multipv, max_nodes, max_ply, min_ply)
             for d in depths for g in range(games)]
    print(f"generating {len(tasks)} games ({len(depths)} depths x {games}) "
          f"on {workers} workers, node_cap={max_nodes or 'none'}, neutral_elo={NEUTRAL_ELO}")

    all_rows = []
    with Pool(workers, initializer=_init, initargs=(e,)) as pool:
        for i, rows in enumerate(pool.imap_unordered(_play_game, tasks, chunksize=1)):
            all_rows.extend(rows)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} games, {len(all_rows)} decisions", flush=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} decisions over {df.game_id.nunique()} games "
          f"({df.planted_depth.value_counts().sort_index().to_dict()}) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--games", type=int, default=0, help="games per planted depth (0=cfg)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default="data/synthetic/selected.parquet")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.games, a.workers, a.out)
