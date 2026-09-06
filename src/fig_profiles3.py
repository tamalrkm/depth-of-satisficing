"""Fig 5 (3-axis): per-player profile without the time-elasticity axis, which is computed from
the posterior and inherits its prior (rating-blind audit, Sep 2026). Reads data/player_profiles.csv
written by analyze.e5; reports nested-CV rating recovery (depth-only vs 3-axis), the elite
ordering by depth alone vs by the 3-axis ridge prediction, and draws the 3-panel figure for the
six broadcast players with the most analysed decisions.  Run: uv run python -m src.fig_profiles3
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figstyle
from analyze import _ridge_cv

FEATS = ["depth", "trap", "disc"]


def ridge_predict(X, t, Xq, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs, Xqs = (X - mu) / sd, (Xq - mu) / sd
    A = np.column_stack([np.ones(len(Xs)), Xs]); Aq = np.column_stack([np.ones(len(Xqs)), Xqs])
    R = lam * np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ t); return Aq @ w


def discordant(score, rating):
    n = len(score); return sum((score[i] - score[j]) * (rating[i] - rating[j]) < 0 for i in range(n) for j in range(i + 1, n)), n * (n - 1) // 2


def main():
    P = pd.read_csv("data/player_profiles.csv"); t = P["zrating"].to_numpy(); X = P[FEATS].to_numpy()
    print(f"players={len(P)}   nested-CV R2: depth-only={_ridge_cv(X, t, [0]):.3f}  3-axis={_ridge_cv(X, t, [0, 1, 2]):.3f}")
    el = P[P["elite"] == 1].sort_values("n", ascending=False).head(6).copy()
    el["pred"] = ridge_predict(X, t, el[FEATS].to_numpy())
    r = el["rating"].to_numpy()
    d1, tot = discordant(el["depth"].to_numpy(), r); d3, _ = discordant(el["pred"].to_numpy(), r)
    print(f"elite six: discordant pairs vs true rating -- depth alone {d1}/{tot}, 3-axis profile {d3}/{tot}")
    z = {f: (P[f].mean(), P[f].std() + 1e-9) for f in FEATS}
    el = el.sort_values("rating", ascending=True).reset_index(drop=True)
    for _, row in el.iloc[::-1].iterrows():
        print(f"  {row['name'][:22]:22s} {int(row['rating'])}  " + "  ".join(f"{f}={(row[f]-z[f][0])/z[f][1]:+.2f}" for f in FEATS) + f"  pred={row['pred']:+.2f}")
    surname = lambda s: max(s.split(), key=len).title()
    ylabels = [f"{surname(r_['name'])}  ({int(r_['rating'])})" for _, r_ in el.iterrows()]
    titles = [("depth", "depth of\nsatisficing"), ("trap", "trap\nsusceptibility"), ("disc", "deep-discovery\nrate")]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), sharey=True, constrained_layout=True); yy = np.arange(len(el))
    for axi, (f, ttl) in zip(axes, titles):
        zs = np.array([(r_[f] - z[f][0]) / z[f][1] for _, r_ in el.iterrows()])
        figstyle.zero_line(axi, axis="x"); axi.hlines(yy, 0, zs, color=figstyle.ACCENT_LIGHT, lw=1.4, zorder=2)
        axi.plot(zs, yy, "o", color=figstyle.ACCENT, markersize=7.5, zorder=3, **figstyle.ring())
        axi.set_title(ttl, fontsize=8.5); axi.set_xlabel("z vs population", fontsize=7.5)
        lim = max(1.0, np.abs(zs).max() * 1.25); axi.set_xlim(-lim, lim); axi.grid(axis="y", visible=False)
    axes[0].set_yticks(yy); axes[0].set_yticklabels(ylabels, fontsize=8)
    figstyle.save(fig, "paper/figs/fig5_player_profiles3.png"); print("wrote paper/figs/fig5_player_profiles3.png")


if __name__ == "__main__":
    main()
