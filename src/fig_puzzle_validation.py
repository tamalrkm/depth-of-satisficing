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
    sw = pd.read_parquet(f"{a.out}/swing_decomposition.parquet")
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)

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

    # (b) two faces of swing by human rating band
    sw["band"] = (sw.rating // 200) * 200
    bx, dd, dlo, dhi, te = [], [], [], [], []
    for b, g in sw.groupby("band"):
        mu, lo, hi = ci(g.deep_discovery.to_numpy().astype(float))
        if np.isnan(mu): continue
        bx.append(b + 100); dd.append(mu); dlo.append(lo); dhi.append(hi); te.append(g.trap_edge.mean())
    ax[1].fill_between(bx, dlo, dhi, color=figstyle.ACCENT_LIGHT, alpha=0.5, lw=0, zorder=2)
    ax[1].plot(bx, dd, "-o", color=figstyle.ACCENT, lw=2.2, ms=5, zorder=3, label="solution is a deep discovery\n(share of puzzles)")
    ax2 = ax[1].twinx()
    ax2.plot(bx, te, "-s", color=figstyle.TC_COLORS["blitz"], lw=2.0, ms=4.5, zorder=3, label="trap's shallow edge over solution\n(win prob., depth <= 4)")
    ax2.axhline(0, color=figstyle.GRID, lw=1); ax2.set_ylabel("trap's shallow edge (win probability)", color=figstyle.TC_COLORS["blitz"])
    ax2.tick_params(axis="y", colors=figstyle.TC_COLORS["blitz"]); ax2.grid(False)
    ax[1].set_xlabel("human puzzle rating (200-point bands)"); ax[1].set_ylabel("share with a deep-discovery solution")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.8, loc="upper left")
    figstyle.panel_label(ax[1], "b")

    # (c) person side: Maia solver ladder by item depth
    mm = m.merge(d[["cd"]], left_on="puzzle_id", right_index=True)
    grp = mm.groupby(["cd", "solver_rating"]).top1.mean().reset_index()
    ramp = figstyle.BAND_RAMP if hasattr(figstyle, "BAND_RAMP") else None
    cds = sorted(grp.cd.unique())
    for i, cd in enumerate(cds):
        g = grp[grp.cd == cd].sort_values("solver_rating")
        col = ramp[int(i * (len(ramp) - 1) / max(len(cds) - 1, 1))] if ramp else None
        ax[2].plot(g.solver_rating, g.top1, "-o", ms=4, lw=2.0, color=col, zorder=3,
                   label=f"{int(cd)}" + ("+" if cd == 5 else ""))
    ax[2].set_xlabel("simulated solver rating (Maia-3)")
    ax[2].set_ylabel("solve rate (solution is top move)")
    ax[2].legend(frameon=False, fontsize=7, title="item critical depth", title_fontsize=7,
                 loc="lower right", ncol=2)
    figstyle.panel_label(ax[2], "c")
    figstyle.save(fig, "paper/figs/fig_puzzle_validation.png")
    print("wrote paper/figs/fig_puzzle_validation.png")


if __name__ == "__main__":
    main()
