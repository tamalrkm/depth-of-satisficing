"""Instant-good-move quality by rating (Regan test).

Ken Regan's proposed test of the depth-of-satisficing claim: among moves played
*instantly* (little to no search) in positions with genuine choice, and when the
player is NOT in time trouble, is move quality roughly equal across ratings? The
depth account predicts that the rating gap in quality collapses for instant moves
(everyone is then relying on pattern, not search) and reopens once players have
time to search at depth.

We measure quality with the project's engine yardstick, not a rating-dependent one:
  - played-move regret at final depth  delta_{played,D}  (0 = engine-best; lower better)
  - best-move match rate                (regret == 0)      -- accuracy analogue
  - blunder rate                         (regret >= 0.10)
These are monotone proxies for Regan's IPR; his C++ pipeline gives the true IPR, and
this script exists so the two code bases test the *same* operational definitions.

Design choices that a referee (or Ken) will check first:
  * REAL CHOICE ONLY. We keep a decision only if the position is non-forced -- at
    least two engine-reasonable moves (final-depth regret <= 0.05) AND at least one
    available move that is a genuine mistake (regret >= 0.10). This removes forced
    recaptures / only-moves, where "instant good move" is trivially equal for all.
  * NOT IN TIME TROUBLE. Both time buckets require a healthy clock
    (clock_before >= max(30 s, 0.15 * base_time)); instant moves are then fast by
    choice, not compulsion.
  * INSTANT = 0 < time_spent <= T_INSTANT (default 1 s); the open lower bound drops
    premoves/clock-rounding zeros. DELIBERATE = time_spent >= T_SLOW (default 10 s).
  * Online (clocked) games only; OTB/broadcast rows lack reliable per-move clocks.

Run:  uv run python -m src.instant_moves            # primary month (data/train.pt)
      uv run python -m src.instant_moves --tensor data/repl06/train.pt --sel data/repl06/selected.parquet
"""
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from . import figstyle

REASONABLE = 0.05     # regret <= this  -> an engine-reasonable move
BLUNDER = 0.10        # regret >= this  -> a real mistake is on the board
T_INSTANT = 1.0       # s: played essentially without search
T_SLOW = 10.0         # s: deliberated
BANDS = [(0, 1400), (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 3500)]
BAND_LABELS = ["<1400", "1400-1599", "1600-1799", "1800-1999", "2000-2199", "2200+"]


def final_depth_regret(delta, depth_mask, y):
    """Played-move regret and the full candidate regret vector, both read at each row's
    deepest VALID depth (capped positions stop short of the grid end)."""
    n, m, d = delta.shape
    last = (depth_mask > 0).float().cumsum(1).argmax(1)          # index of last valid depth col
    rows = torch.arange(n)
    cand = delta[rows, :, last]                                  # [N, M] regret of every candidate
    played = cand[rows, y]                                       # [N]    regret of the played move
    return played.numpy(), cand.numpy()


def load(tensor_path, sel_path):
    blob = torch.load(tensor_path, map_location="cpu", weights_only=False)
    delta, mask, dmask, y = blob["delta"], blob["move_mask"], blob["depth_mask"], blob["y"]
    meta = blob["meta"]
    played_reg, cand = final_depth_regret(delta, dmask, y)
    mask_np = mask.numpy() > 0

    cand_masked = np.where(mask_np, cand, np.nan)
    n_reasonable = np.nansum(cand_masked <= REASONABLE, axis=1)
    max_avail = np.nanmax(np.where(mask_np, cand, -np.inf), axis=1)

    df = pd.DataFrame({
        "pos_id": meta["pos_id"],
        "elo": np.asarray(meta["elo"], dtype=float),
        "source": meta["source"],
        "played_reg": played_reg,
        "n_reasonable": n_reasonable.astype(int),
        "max_avail": max_avail,
    })
    sel = pd.read_parquet(sel_path, columns=[
        "pos_id", "time_spent", "clock_before", "base_time", "increment"]).set_index("pos_id")
    df = df.join(sel, on="pos_id")
    return df


def prepare(df, t_instant=T_INSTANT, t_slow=T_SLOW):
    df = df[df["source"] == "online"].copy()
    df = df.dropna(subset=["time_spent", "clock_before", "base_time"])
    # genuine decision: >1 reasonable move AND a real mistake available
    real = (df["n_reasonable"] >= 2) & (df["max_avail"] >= BLUNDER)
    # healthy clock (not time trouble) for both buckets
    healthy = df["clock_before"] >= np.maximum(30.0, 0.15 * df["base_time"])
    df = df[real & healthy].copy()
    df["bucket"] = np.where(
        (df["time_spent"] > 0) & (df["time_spent"] <= t_instant), "instant",
        np.where(df["time_spent"] >= t_slow, "deliberate", "other"))
    df = df[df["bucket"] != "other"].copy()
    df["band"] = pd.cut(df["elo"], bins=[b[0] for b in BANDS] + [BANDS[-1][1]],
                        labels=BAND_LABELS, right=False)
    return df


def summarise(df):
    rows = []
    for band in BAND_LABELS:
        for bucket in ["instant", "deliberate"]:
            g = df[(df["band"] == band) & (df["bucket"] == bucket)]
            if len(g) == 0:
                continue
            rows.append({
                "band": band, "bucket": bucket, "n": len(g),
                "mean_regret": g["played_reg"].mean(),
                "accuracy": (g["played_reg"] <= 1e-6).mean(),
                "blunder_rate": (g["played_reg"] >= BLUNDER).mean(),
            })
    return pd.DataFrame(rows)


def band_slope(tab, metric):
    """Least-squares slope of a metric across rating bands (band index as x)."""
    out = {}
    for bucket in ["instant", "deliberate"]:
        t = tab[tab["bucket"] == bucket]
        x = np.array([BAND_LABELS.index(b) for b in t["band"]], float)
        yv = t[metric].to_numpy()
        if len(x) >= 2:
            out[bucket] = np.polyfit(x, yv, 1)[0]
    return out


def adjusted_rating_slope(df, bucket):
    """Difficulty-adjusted effect of rating on played-move regret, within a time bucket.
    OLS  regret ~ 1 + elo/1000 + max_avail + n_reasonable + ply-proxy, so the rating
    coefficient is read at matched position difficulty (time spent is endogenous to
    difficulty, so the raw instant-vs-deliberate levels are confounded; the cross-rating
    slope within a bucket, at fixed difficulty, is the clean quantity). Plain OLS here;
    the confirmatory version should cluster SEs by player, as in E4."""
    g = df[df["bucket"] == bucket]
    y = g["played_reg"].to_numpy()
    X = np.column_stack([
        np.ones(len(g)),
        g["elo"].to_numpy() / 1000.0,
        g["max_avail"].to_numpy(),
        g["n_reasonable"].to_numpy().astype(float),
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(g) - X.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta[1], se[1], len(g)      # regret change per +1000 Elo, at fixed difficulty


def figure(tab, path):
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    styles = {"instant": (figstyle.TC_COLORS["blitz"], "instant (<=1 s)"),
              "deliberate": (figstyle.INK, "deliberate (>=10 s)")}
    for j, (metric, ylab, lo) in enumerate([
            ("mean_regret", "mean regret of played move\n(win probability; lower = better)", True),
            ("accuracy", "engine-best match rate", False)]):
        a = ax[j]
        for bucket, (col, lab) in styles.items():
            t = tab[tab["bucket"] == bucket]
            x = [BAND_LABELS.index(b) for b in t["band"]]
            a.plot(x, t[metric], "-o", color=col, lw=2.2, ms=5, zorder=4, label=lab)
        a.set_xticks(range(len(BAND_LABELS)))
        a.set_xticklabels(BAND_LABELS, rotation=35, ha="right", fontsize=7)
        a.set_xlabel("player rating band (Elo)")
        a.set_ylabel(ylab)
        figstyle.panel_label(a, "ab"[j])
        if j == 0:
            a.legend(frameon=False, fontsize=8, loc="best")
    figstyle.save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="data/train.pt")
    ap.add_argument("--sel", default="data/selected.parquet")
    ap.add_argument("--t-instant", type=float, default=T_INSTANT)
    ap.add_argument("--out", default="paper/figs/figR_instant_moves.png")
    args = ap.parse_args()
    df = prepare(load(args.tensor, args.sel), t_instant=args.t_instant)
    print(f"real-choice, healthy-clock, online decisions: {len(df):,}")
    print("  bucket counts:", df["bucket"].value_counts().to_dict())
    tab = summarise(df)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(tab.to_string(index=False))

    print("\ndifficulty-adjusted rating slope (regret change per +1000 Elo, at matched difficulty):")
    slopes = {}
    for bucket in ["instant", "deliberate"]:
        b, se, n = adjusted_rating_slope(df, bucket)
        slopes[bucket] = b
        print(f"  {bucket:11s}  {b:+.4f}  (SE {se:.4f}, n={n:,})")
    if slopes.get("instant") and slopes.get("deliberate"):
        print(f"  -> deliberation amplifies the skill gap by "
              f"{slopes['deliberate']/slopes['instant']:.1f}x "
              f"(more negative = quality improves faster with rating)")

    figure(tab, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
