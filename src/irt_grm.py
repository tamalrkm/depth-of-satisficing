"""
Item-side IRT layer (graded-response / explanatory).

Each decision is an ITEM. Engine-anchored item properties (no free per-item params -- handles the
one-player-per-position sparsity, in the spirit of an LLTM / explanatory IRT):
  difficulty  b_j  = critical depth (deepest misleading grid depth), in plies.
  discrimination   = position swing magnitude.
Ordered response (error severity) from the played move's full-depth regret (win-prob):
  0 best (<=.02) | 1 inaccuracy (.02-.05) | 2 mistake (.05-.10) | 3 blunder (>.10).
Ability = within-pool rating z (external).

Explanatory proportional-odds GRM:  severity ~ ability + difficulty + swing + ability:swing.
Expect: ability<0 (abler -> milder errors), difficulty>0, ability:swing<0 (ability discriminates
MORE in high-swing items = discrimination). Plus the test-information picture: information
concentrates in high-swing items, on the ply scale shared with the depth-of-satisficing ability.

    uv run python src/irt_grm.py
"""
import numpy as np
import pandas as pd
import torch
import yaml
from statsmodels.miscmodels.ordinal_model import OrderedModel

import figstyle
import matplotlib.pyplot as plt

FIG = "paper/figs_irt"


def main(cfg_path="config.yaml"):
    import os
    os.makedirs(FIG, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path))
    tag = "_chesscom" if "chesscom" in cfg_path else ("_2026_05" if "2026_05" in cfg_path else "")
    print(f"POOL: {cfg_path}  (figure tag='{tag}')")
    grid = np.array(cfg["model"]["depth_grid"])
    blob = torch.load(cfg["data"]["train_tensor"])
    meta = blob["meta"]
    delta = blob["delta"].numpy(); mm = blob["move_mask"].numpy().astype(bool)
    dmask = blob.get("depth_mask")
    dmask = (dmask.numpy() if dmask is not None else np.ones((delta.shape[0], delta.shape[2]))).astype(bool)
    y = blob["y"].numpy(); N, M, D = delta.shape
    elo = np.array(meta["elo"], float); tc = np.array(meta["time_class"]); src = np.array(meta["source"])
    tsw = blob["context"].numpy()[:, 3]                      # position swing magnitude (discrimination)

    de = delta.copy(); de[~mm] = np.inf
    lv = np.clip(dmask.sum(1).astype(int) - 1, 0, D - 1); ar = np.arange(N)
    final_best = de[ar, :, lv].argmin(1)
    best_at_d = de.argmin(1)
    valid = np.arange(D)[None, :] <= lv[:, None]
    mismatch = (best_at_d != final_best[:, None]) & valid
    any_mis = mismatch.any(1)
    last_mis = (D - 1) - mismatch[:, ::-1].argmax(1)
    b_j = np.where(any_mis, grid[np.clip(last_mis, 0, D - 1)], float(grid[0])).astype(float)  # critical depth

    reg = de[ar, y, lv]                                      # played full-depth regret
    sev = np.zeros(N, int)
    sev[reg > 0.02] = 1; sev[reg > 0.05] = 2; sev[reg > 0.10] = 3
    pool = np.where(src == "broadcast", "otb", tc)
    z = np.full(N, np.nan)
    for p in np.unique(pool):
        s = pool == p; z[s] = (elo[s] - elo[s].mean()) / (elo[s].std() + 1e-9)

    dist = np.bincount(sev, minlength=4) / N
    print(f"N={N:,}  severity dist  best={dist[0]:.3f} inacc={dist[1]:.3f} mistake={dist[2]:.3f} blunder={dist[3]:.3f}")
    print(f"item difficulty b_j (critical depth): mean {b_j.mean():.1f} plies, range [{b_j.min():.0f},{b_j.max():.0f}]")

    # ---------- explanatory proportional-odds GRM (subsample for the MLE) ----------
    s = np.isfinite(z)
    zs = (z - np.nanmean(z)) / np.nanstd(z)
    bs = (b_j - b_j.mean()) / b_j.std()
    ws = (np.log1p(tsw) - np.log1p(tsw).mean()) / np.log1p(tsw).std()
    X = pd.DataFrame({"ability": zs, "difficulty": bs, "swing": ws, "ability_x_swing": zs * ws})[s]
    Y = sev[s]
    rng = np.random.default_rng(0)
    take = rng.choice(np.where(s)[0].size, size=min(200000, s.sum()), replace=False)
    res = OrderedModel(Y[take], X.to_numpy()[take], distr="logit").fit(method="bfgs", disp=False)
    names = ["ability", "difficulty", "swing", "ability_x_swing"]
    print("\nexplanatory proportional-odds GRM (severity; higher = worse):")
    for i, nm in enumerate(names):
        print(f"   {nm:16s} beta={res.params[i]:+.3f}  p={res.pvalues[i]:.1e}")
    print("   expect: ability<0, difficulty>0, ability_x_swing<0 (ability discriminates more in high-swing items)")

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (a) information: P(error = not best) vs ability, by swing tercile -> discrimination grows with swing
    err = (sev >= 1).astype(float)
    swt = np.digitize(tsw, np.quantile(tsw, [1/3, 2/3]))
    zq = np.quantile(z[s], np.linspace(0, 1, 9)); zc = 0.5 * (zq[:-1] + zq[1:])
    for t, lab in [(0, "low swing"), (1, "mid swing"), (2, "high swing")]:
        m = s & (swt == t); bid = np.clip(np.digitize(z[m], zq[1:-1]), 0, 7)
        rate = np.array([err[m][bid == k].mean() if (bid == k).sum() > 30 else np.nan for k in range(8)])
        ax[0].plot(zc, rate, marker="o", label=lab)
    ax[0].set_xlabel("ability (within-pool rating z)"); ax[0].set_ylabel("P(error: not best move)")
    ax[0].set_title("(a) discrimination grows with swing"); ax[0].legend(); ax[0].grid(alpha=.3)
    figstyle.panel_label(ax[0], "a")

    # (b) graded-response operating curves: P(each severity category) vs ability
    for k, lab, col in [(0, "best", "#0072B2"), (1, "inaccuracy", "#56B4E9"),
                        (2, "mistake", "#E69F00"), (3, "blunder", "#D55E00")]:
        rate = np.array([(sev[s][np.clip(np.digitize(z[s], zq[1:-1]), 0, 7) == j] == k).mean()
                         if (np.clip(np.digitize(z[s], zq[1:-1]), 0, 7) == j).sum() > 30 else np.nan
                         for j in range(8)])
        ax[1].plot(zc, rate, marker="o", color=col, label=lab)
    ax[1].set_xlabel("ability (within-pool rating z)"); ax[1].set_ylabel("P(severity category)")
    ax[1].set_title("(b) graded-response operating curves"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    figstyle.panel_label(ax[1], "b")

    # (c) one ruler: item difficulty distribution on the ply scale (shared with depth-of-satisficing)
    ax[2].hist(b_j, bins=np.array(grid) - 0.5 if False else np.arange(grid.min()-1, grid.max()+2, 2),
               color="#0072B2", alpha=.8, edgecolor="white")
    ax[2].set_xlabel("critical depth (plies)"); ax[2].set_ylabel("number of positions (items)")
    ax[2].set_title("(c) items on the depth ruler"); ax[2].grid(alpha=.3, axis="y")
    figstyle.panel_label(ax[2], "c")
    fig.tight_layout(); p = f"{FIG}/irt_grm{tag}.png"; figstyle.save(fig, p)
    print(f"\n-> {p}")

    # group-level ability validity (sidesteps low per-player reliability): severity falls with rating band.
    # Use the largest pool present (chess.com has no 'classical').
    counts = {p: int(((pool == p) & np.isfinite(z)).sum()) for p in np.unique(pool)}
    big = max(counts, key=counts.get)
    cl = (pool == big) & np.isfinite(z)
    print(f"\nGROUP validity: mean severity by within-pool rating quintile ({big}, n={cl.sum():,}):")
    if cl.sum() > 100:
        q = np.digitize(z[cl], np.quantile(z[cl], [.2, .4, .6, .8]))
        print("   " + "  ".join(f"Q{k+1}={sev[cl][q==k].mean():.3f}" for k in range(5)))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
