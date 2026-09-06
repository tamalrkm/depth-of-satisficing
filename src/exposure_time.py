"""Model-free E3 analogue: within player, are SLOWER decisions' errors exposed DEEPER?

Uses the exposure depth of src.discriminant_depth (first grid depth at which an error's regret
reaches half its final value). Within player (player fixed effects via de-meaning), regresses
exposure depth on log time-on-move, controlling for the error's final magnitude, the position's
total swing (context[3]) and move number, with player-clustered SEs. Also a paired version:
each player's errors split at their own median time; mean exposure depth slow - fast.
Online classical+rapid, errors only.   Run:  uv run python -m src.exposure_time
"""
import numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon

GRID = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])


def main():
    blob = torch.load("data/train.pt", map_location="cpu", weights_only=False)
    delta, y, dmask, ctx, meta = blob["delta"], blob["y"], blob["depth_mask"], blob["context"], blob["meta"]
    n = len(y); pr = delta[torch.arange(n), y].numpy()
    last = (dmask > 0).float().cumsum(1).argmax(1).numpy(); final = pr[np.arange(n), last]
    idx = np.array([np.argmax(pr[i, :last[i] + 1] >= 0.5 * final[i]) if final[i] > 0 else -1 for i in range(n)])
    df = pd.DataFrame({"player": meta["player"], "tc": meta["time_class"], "src": meta["source"],
                       "ply": np.asarray(meta["ply"], float), "final": final,
                       "exp": np.where(idx >= 0, GRID[np.clip(idx, 0, 10)], np.nan),
                       "swing": ctx[:, 3].numpy(), "pos_id": meta["pos_id"]})
    ts = pd.read_parquet("data/selected.parquet", columns=["pos_id", "time_spent"]).set_index("pos_id")
    df["t"] = ts.reindex(df["pos_id"])["time_spent"].to_numpy()
    df = df[(df["src"] == "online") & df["tc"].isin(["classical", "rapid"]) & (df["final"] >= 0.05)
            & np.isfinite(df["t"]) & (df["t"] > 0)].copy()
    df["lt"] = np.log(df["t"])
    cnt = df.groupby("player")["exp"].transform("size"); df = df[cnt >= 10].copy()
    print(f"online classical+rapid errors with time, players with >=10 errors: {len(df):,} decisions, {df.player.nunique():,} players")

    # (1) within-player regression with covariates, player-clustered SE
    for col in ["exp", "lt", "final", "swing", "ply"]:
        df[col + "_dm"] = df[col] - df.groupby("player")[col].transform("mean")
    X = np.column_stack([df["lt_dm"], df["final_dm"], df["swing_dm"], df["ply_dm"] / 100]); yv = df["exp_dm"].to_numpy()
    b, *_ = np.linalg.lstsq(X, yv, rcond=None); r = yv - X @ b; XtXi = np.linalg.inv(X.T @ X)
    codes = pd.factorize(df["player"])[0]; agg = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(agg, codes, X * r[:, None])
    se = np.sqrt(np.diag(XtXi @ (agg.T @ agg) @ XtXi))
    print("\n(1) exposure depth ~ log(time) + error size + position swing + ply, player fixed effects, clustered SE")
    for name, bi, si in zip(["log time (per e-fold)", "error size", "position swing", "ply/100"], b, se):
        print(f"   {name:22s} {bi:+.3f} plies  (SE {si:.3f}, z={bi/si:+.1f})")

    # (2) paired: per player, errors split at own median time; slow - fast exposure depth
    print("\n(2) paired within-player: mean exposure depth (slow errors) - (fast errors)")
    rng = np.random.default_rng(0)
    for t in ["classical", "rapid", "both"]:
        g = df if t == "both" else df[df["tc"] == t]; diffs = []
        for p, sub in g.groupby("player"):
            if len(sub) < 10: continue
            med = sub["t"].median(); f, s = sub[sub["t"] <= med]["exp"], sub[sub["t"] > med]["exp"]
            if len(f) >= 3 and len(s) >= 3: diffs.append(s.mean() - f.mean())
        d = np.array(diffs); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"   {t:10s} players={len(d):>5d}  diff={d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  frac>0={(d>0).mean():.2f}  Wilcoxon p={wilcoxon(d).pvalue:.1e}")


if __name__ == "__main__":
    main()
