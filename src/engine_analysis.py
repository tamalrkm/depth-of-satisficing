"""
Stage 3 (implemented): Stockfish 18 multi-PV-across-depths analysis.

For each selected position we record, for every candidate move, its WIN PROBABILITY
at EVERY iterative-deepening depth d = 1..D, by streaming the engine's `info` lines.
Win probability comes from the engine's WDL output (UCI_ShowWDL), so regret is already
in outcome-probability units -- no logistic fit, no centipawn scaling.

Output schema (data/depth_traj.parquet), one row per (position, move, depth):
    pos_id        str    stable id of the position (e.g. game_id:ply)
    fen           str
    move          str    UCI move
    depth         int
    winprob       float  win probability for `move` at `depth`, side-to-move POV (0..1)
    is_played     bool   whether this move is the human's played move
    is_topk_final bool   in the engine's top-K at the deepest depth

Run:
    python src/engine_analysis.py --config config.yaml
"""
import argparse, yaml
import chess, chess.engine
import pandas as pd
from tqdm import tqdm


def winprob_from_info(info):
    """Win probability (side-to-move POV) from a streamed info dict."""
    wdl = info.get("wdl")
    if wdl is not None:
        rel = wdl.relative            # chess.engine.Wdl, per-mille, side-to-move POV
        return (rel.wins + 0.5 * rel.draws) / 1000.0
    # fallback if WDL unavailable: convert score to expectation
    return info["score"].relative.wdl(model="sf").expectation()


def analyse_position(engine, board, depth, multipv, played_uci):
    """Return list of rows: per (move, depth) winprob across the whole search."""
    # move -> {depth: winprob}
    traj = {}
    limit = chess.engine.Limit(depth=depth)
    with engine.analysis(board, limit, multipv=multipv) as analysis:
        for info in analysis:
            if "pv" not in info or "depth" not in info:
                continue
            mv = info["pv"][0].uci()
            d = info["depth"]
            try:
                wp = winprob_from_info(info)
            except Exception:
                continue
            traj.setdefault(mv, {})[d] = wp

    # ensure the played move has a trajectory even if it dropped out of top-K
    if played_uci is not None and played_uci not in traj:
        try:
            mv_obj = chess.Move.from_uci(played_uci)
            with engine.analysis(board, limit, root_moves=[mv_obj]) as a2:
                for info in a2:
                    if "pv" not in info or "depth" not in info:
                        continue
                    if info["pv"][0].uci() != played_uci:
                        continue
                    traj.setdefault(played_uci, {})[info["depth"]] = winprob_from_info(info)
        except Exception:
            pass

    final_depth = max((max(ds) for ds in traj.values()), default=0)
    topk_final = {mv for mv, ds in traj.items() if final_depth in ds}
    rows = []
    for mv, ds in traj.items():
        for d, wp in ds.items():
            rows.append(dict(move=mv, depth=d, winprob=wp,
                             is_played=(mv == played_uci),
                             is_topk_final=(mv in topk_final)))
    return rows


def main(cfg):
    e = cfg["engine"]
    engine = chess.engine.SimpleEngine.popen_uci(e["path"])
    engine.configure({"Threads": e["threads"], "Hash": e["hash_mb"], "NumaPolicy": "none"})
    if e.get("show_wdl", True):
        try:
            engine.configure({"UCI_ShowWDL": True})
        except chess.engine.EngineError:
            print("warning: engine did not accept UCI_ShowWDL")

    sel = pd.read_parquet(cfg["data"]["selected"])
    out_rows = []
    for r in tqdm(sel.itertuples(), total=len(sel), desc="SF18 multi-PV"):
        board = chess.Board(r.fen)
        rows = analyse_position(engine, board, e["depth"], e["multipv"], r.played_uci)
        for row in rows:
            row.update(pos_id=r.pos_id, fen=r.fen)
            out_rows.append(row)

    engine.quit()
    df = pd.DataFrame(out_rows)
    df.to_parquet(cfg["data"]["depth_traj"], index=False)
    print(f"wrote {len(df)} rows -> {cfg['data']['depth_traj']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
