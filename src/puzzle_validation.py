"""External validation: does engine-derived item difficulty predict HUMAN-calibrated difficulty?

The paper's item-response section treats each position as a test item whose difficulty is its
critical depth. That claim has so far been internal -- difficulty and ability both come from our
own pipeline. The Lichess puzzle database supplies an independent human-calibrated difficulty for
millions of positions: each puzzle carries a Glicko rating estimated from the successes and
failures of rated solvers, computed by Lichess with no reference to any engine-depth analysis.

This script (1) draws a rating-stratified sample of well-calibrated puzzles, writing the
`selected.parquet` that src/run_engine.py consumes, and (2) after the engine pass, derives per
puzzle:
    crit_depth   deepest depth at which the apparent-best move differs from the full-depth best
                 (the paper's item difficulty b_j)
    emerge_depth first depth from which the solution is best and remains best to full depth
    shallow_reg  mean regret of the solution at depth <= 4 (how wrong it looks to a quick glance)
    sw_solution  swing of the solution move; total_swing  sum |swing| over candidates
and relates them to the puzzle's human rating, controlling for the two trivial determinants of
puzzle difficulty: how many moves the solver must find, and whether it is a mate.

Stage 1:  uv run python -m src.puzzle_validation sample --out DIR [--per-band 100]
Stage 2:  uv run python src/run_engine.py --config DIR/config.yaml
Stage 3:  uv run python -m src.puzzle_validation analyse --out DIR
"""
import argparse, os
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

PUZZLES = os.path.expanduser("~/projects/chess-guru/data/lichess_puzzles.parquet")
MIN_PLAYS, MAX_RD = 500, 80          # well-calibrated puzzles only
BANDS = list(range(800, 2600, 200))


def sample(out, per_band, seed=17):
    import chess
    os.makedirs(f"{out}/shards", exist_ok=True)
    p = pd.read_parquet(PUZZLES)
    w = p[(p.nbplays >= MIN_PLAYS) & (p.rating_dev <= MAX_RD)].copy()
    w["solver_moves"] = w.moves.str.split().str.len() // 2
    rng = np.random.default_rng(seed)
    picks = []
    for lo in BANDS:
        g = w[(w.rating >= lo) & (w.rating < lo + 200)]
        n = min(per_band, len(g))
        picks.append(g.iloc[rng.choice(len(g), n, replace=False)])
    s = pd.concat(picks, ignore_index=True)
    # the position shown to the solver is AFTER the opponent's first move; solution is the next
    pos = [(lambda b, mv: (b.push_uci(mv[0]) or b.fen(), mv[1]))(chess.Board(f), m.split())
           for f, m in zip(s.fen, s.moves)]
    s["pos_fen"] = [a for a, _ in pos]; s["solution"] = [b for _, b in pos]
    pd.DataFrame({"pos_id": s.puzzle_id, "fen": s.pos_fen, "played_uci": s.solution}) \
        .to_parquet(f"{out}/selected.parquet", index=False)
    s.drop(columns=["fen", "moves"]).to_parquet(f"{out}/meta.parquet", index=False)
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["data"]["selected"] = f"{out}/selected.parquet"
    cfg["data"]["depth_traj"] = f"{out}/depth_traj.parquet"
    cfg["parallel"]["shard_dir"] = f"{out}/shards"
    yaml.safe_dump(cfg, open(f"{out}/config.yaml", "w"))
    print(f"sampled {len(s)} puzzles over {len(BANDS)} rating bands -> {out}/selected.parquet")


def features(out):
    dt = pd.read_parquet(f"{out}/depth_traj.parquet")
    meta = pd.read_parquet(f"{out}/meta.parquet").set_index("puzzle_id")
    rows = []
    for pid, g in dt.groupby("pos_id"):
        piv = g.pivot_table(index="move", columns="depth", values="winprob").ffill(axis=1)
        depths = sorted(piv.columns)
        if not depths or depths[-1] < 8:
            continue
        D = depths[-1]
        played = g.loc[g.is_played, "move"]
        if played.empty or played.iloc[0] not in piv.index:
            continue
        sol = played.iloc[0]
        delta = piv.max(0) - piv                       # regret per move per depth
        best_move = piv.idxmax(0)
        mis = [d for d in depths if best_move[d] != best_move[D]]
        emerge = next((d for d in depths if all(best_move[dd] == sol for dd in depths if dd >= d)), D + 1)
        shallow = [delta.loc[sol, d] for d in depths if d <= 4]
        sw = delta.sub(delta[D], axis=0).sum(1)
        rows.append(dict(puzzle_id=pid, crit_depth=max(mis) if mis else depths[0],
                         emerge_depth=emerge, shallow_reg=float(np.nanmean(shallow)) if shallow else np.nan,
                         sw_solution=float(sw[sol]), total_swing=float(sw.abs().sum()),
                         sol_is_final_best=int(best_move[D] == sol)))
    f = pd.DataFrame(rows).set_index("puzzle_id").join(meta)
    f["mate"] = f.themes.str.contains("mate").astype(int)
    f["endgame"] = f.themes.str.contains("endgame").astype(int)
    return f.dropna(subset=["crit_depth", "emerge_depth", "shallow_reg", "sw_solution"])


def _ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    XtXi = np.linalg.inv(X.T @ X); Xu = X * r[:, None]
    se = np.sqrt(np.diag(XtXi @ (Xu.T @ Xu) @ XtXi))
    print(f"   R2={1 - r.var() / y.var():.3f}")
    for n, bi, si in zip(names, b[1:], se[1:]):
        print(f"     {n:14s} {bi:+8.1f}  (SE {si:.1f}, z={bi / si:+.1f})")


def analyse(out):
    f = features(out)
    f.to_parquet(f"{out}/features.parquet")
    print(f"puzzles: {len(f)}   (engine full-depth best == puzzle solution: {f.sol_is_final_best.mean():.2f})")
    FEATS = ["crit_depth", "emerge_depth", "shallow_reg", "sw_solution", "total_swing"]
    print("\nSpearman with the puzzle's human rating:")
    for c in FEATS + ["solver_moves"]:
        r, p = spearmanr(f[c], f.rating); print(f"  {c:13s} rho={r:+.3f}  p={p:.1e}")
    z = lambda c: ((f[c] - f[c].mean()) / (f[c].std() + 1e-12)).to_numpy()
    y = f.rating.to_numpy()
    print("\nOLS puzzle rating ~ standardised predictors (HC1 SE):")
    print("  controls only (solution length, mate, endgame):")
    _ols(y, [z("solver_moves"), f.mate, f.endgame], ["solver_moves", "mate", "endgame"])
    print("  + engine depth features:")
    _ols(y, [z("solver_moves"), f.mate, f.endgame, z("crit_depth"), z("emerge_depth"), z("shallow_reg")],
         ["solver_moves", "mate", "endgame", "crit_depth", "emerge_depth", "shallow_reg"])
    C = np.column_stack([np.ones(len(f)), z("solver_moves"), f.mate.to_numpy(), f.endgame.to_numpy()])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    print("\npartial Spearman, controlling solution length + mate/endgame theme:")
    for c in FEATS:
        r, p = spearmanr(res(z(c)), res(y)); print(f"  {c:13s} rho={r:+.3f}  p={p:.1e}")
    print("\nwithin solution-length strata (no residualisation):")
    for k, g in f.groupby("solver_moves"):
        if len(g) < 60: continue
        r1, p1 = spearmanr(g.crit_depth, g.rating)
        print(f"  solver_moves={k}  n={len(g):5d}  crit_depth rho={r1:+.3f} (p={p1:.1e})")
    nm = f[f.mate == 0]
    print(f"\nnon-mate puzzles only (n={len(nm)}):")
    for c in FEATS:
        r, p = spearmanr(nm[c], nm.rating); print(f"  {c:13s} rho={r:+.3f}  p={p:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "analyse"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-band", type=int, default=100)
    a = ap.parse_args()
    sample(a.out, a.per_band) if a.stage == "sample" else analyse(a.out)
