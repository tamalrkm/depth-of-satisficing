"""Discriminant validity: is 'depth' just error MAGNITUDE in disguise?

The paper's claim is that skill is legible in *at what depth* a flaw becomes visible, not in
*how large* the flaw is. A sceptical reader will ask whether the two are the same thing. This
script answers model-free, straight from the played-move regret trajectories:

  1. exposure depth = the first grid depth at which the played move's regret reaches half of
     its final-depth value (defined for errors, final regret >= 0.05);
  2. within FIXED bins of final-depth regret (error magnitude), regress exposure depth on
     rating (plus ply), with player-clustered (CR0) SEs;
  3. report the raw rank correlation between error magnitude and exposure depth.

If depth carries skill information at fixed magnitude, and magnitude and exposure depth are
essentially uncorrelated, then depth is a distinct dimension, not a re-description of size.

Run:  uv run python -m src.discriminant_depth [--tensor data/train.pt]
"""
import argparse
import numpy as np
import pandas as pd
import torch

GRID = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
BINS = [(0.05, 0.10, "mistake  (0.05-0.10)"), (0.10, 0.20, "blunder  (0.10-0.20)"),
        (0.20, 1.01, "blunder  (>0.20)")]


def cr0_slope(g, xcol, ycol):
    X = np.column_stack([np.ones(len(g)), g[xcol].to_numpy() / 1000.0, g["ply"].to_numpy() / 100.0])
    yv = g[ycol].to_numpy()
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    codes = pd.factorize(g["player"].to_numpy())[0]
    agg = np.zeros((codes.max() + 1, X.shape[1]))
    np.add.at(agg, codes, X * r[:, None])
    se = np.sqrt(np.diag(XtXi @ (agg.T @ agg) @ XtXi))
    return b[1], se[1]


def load(tensor):
    blob = torch.load(tensor, map_location="cpu", weights_only=False)
    delta, y, dmask, meta = blob["delta"], blob["y"], blob["depth_mask"], blob["meta"]
    n = len(y)
    pr = delta[torch.arange(n), y].numpy()                        # [N, D] played-move regret
    last = (dmask > 0).float().cumsum(1).argmax(1).numpy()        # last valid depth column
    final = pr[np.arange(n), last]
    half = 0.5 * final
    idx = np.array([np.argmax(pr[i, :last[i] + 1] >= half[i]) if final[i] > 0 else -1
                    for i in range(n)])
    exp_depth = np.where(idx >= 0, GRID[np.clip(idx, 0, len(GRID) - 1)], np.nan)
    return pd.DataFrame({
        "elo": np.asarray(meta["elo"], float), "tc": meta["time_class"], "src": meta["source"],
        "player": meta["player"], "ply": np.asarray(meta["ply"], float),
        "final": final, "exp_depth": exp_depth})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="data/train.pt")
    ap.add_argument("--controls", default="classical,rapid")
    a = ap.parse_args()
    df = load(a.tensor)
    df = df[(df["src"] == "online") & df["tc"].isin(a.controls.split(","))]
    err = df[df["final"] >= 0.05].copy()
    print(f"online {a.controls} errors (final regret >= 0.05): {len(err):,}")
    rho = err[["final", "exp_depth"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman(error magnitude, exposure depth) = {rho:+.3f}   <- ~0 means distinct dimensions\n")
    print(f"{'error magnitude':22s} {'n':>7s} {'<1600':>7s} {'1600-1999':>10s} {'2000+':>7s}   slope per +1000 Elo")
    for lo, hi, lab in BINS:
        g = err[(err["final"] >= lo) & (err["final"] < hi)]
        m = lambda l, h: g[(g["elo"] >= l) & (g["elo"] < h)]["exp_depth"].mean()
        s, se = cr0_slope(g, "elo", "exp_depth")
        print(f"{lab:22s} {len(g):>7,} {m(0,1600):>7.2f} {m(1600,2000):>10.2f} {m(2000,4000):>7.2f}   "
              f"{s:+.2f} plies (SE {se:.2f}, z={s/se:+.1f})")


if __name__ == "__main__":
    main()
