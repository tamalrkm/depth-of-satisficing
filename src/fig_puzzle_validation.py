"""Figure: engine-derived item difficulty against an independent human criterion.

(a) Mean human puzzle rating (Lichess Glicko from millions of rated solve attempts) by the
    engine's critical depth for the item, with player-free bootstrap CIs over puzzles.
(b) Person side: solve rate of Maia-3 simulated solvers across a ladder of ratings, split by
    the item's critical depth. Deeper items sit lower for a pattern model at every strength.
Run:  uv run python -m src.fig_puzzle_validation [--out data/puzzles]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figstyle

rng = np.random.default_rng(0)


def ci(v, n=2000):
    if len(v) < 10: return np.nan, np.nan, np.nan
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/puzzles"); a = ap.parse_args()
    d = pd.read_parquet(f"{a.out}/features_with_maia.parquet")
    m = pd.read_parquet(f"{a.out}/maia_solvers.parquet")
    d["cd"] = d.crit_depth.clip(upper=5)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)

    # (a) human difficulty vs engine critical depth
    xs, ms, los, his, ns = [], [], [], [], []
    for cd, g in d.groupby("cd"):
        mu, lo, hi = ci(g.rating.to_numpy())
        if np.isnan(mu): continue
        xs.append(cd); ms.append(mu); los.append(lo); his.append(hi); ns.append(len(g))
    ax[0].fill_between(xs, los, his, color=figstyle.ACCENT_LIGHT, alpha=0.5, lw=0, zorder=2)
    ax[0].plot(xs, ms, "-o", color=figstyle.ACCENT, lw=2.2, ms=6, zorder=3, **figstyle.ring())
    for x, y_, n in zip(xs, ms, ns):
        ax[0].annotate(f"n={n:,}", (x, y_), textcoords="offset points", xytext=(0, -14),
                       ha="center", fontsize=6.5, color=figstyle.INK2)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([f"{int(x)}" + ("+" if x == 5 else "") for x in xs])
    ax[0].set_xlabel("engine critical depth of the item (plies)")
    ax[0].set_ylabel("human puzzle rating\n(Glicko from rated solvers)")
    figstyle.panel_label(ax[0], "a")

    # (b) person side: Maia solver ladder by item depth
    mm = m.merge(d[["cd"]], left_on="puzzle_id", right_index=True)
    grp = mm.groupby(["cd", "solver_rating"]).top1.mean().reset_index()
    ramp = figstyle.BAND_RAMP if hasattr(figstyle, "BAND_RAMP") else None
    cds = sorted(grp.cd.unique())
    for i, cd in enumerate(cds):
        g = grp[grp.cd == cd].sort_values("solver_rating")
        col = ramp[int(i * (len(ramp) - 1) / max(len(cds) - 1, 1))] if ramp else None
        ax[1].plot(g.solver_rating, g.top1, "-o", ms=4, lw=2.0, color=col, zorder=3,
                   label=f"{int(cd)}" + ("+" if cd == 5 else ""))
    ax[1].set_xlabel("simulated solver rating (Maia-3)")
    ax[1].set_ylabel("solve rate (solution is top move)")
    ax[1].legend(frameon=False, fontsize=7, title="item critical depth", title_fontsize=7,
                 loc="lower right", ncol=2)
    figstyle.panel_label(ax[1], "b")
    figstyle.save(fig, "paper/figs/fig_puzzle_validation.png")
    print("wrote paper/figs/fig_puzzle_validation.png")


if __name__ == "__main__":
    main()
