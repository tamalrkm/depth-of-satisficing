"""Replacement Fig. 2: depth of satisficing vs rating under the rating-blind pooled estimator.
Reads the per-player tables written by src.diag_player_depth (data/diag/*.parquet).
  a) 2025-09, fully rating-blind (fixed-1950 Maia, no rating in context, flat depth prior):
     per-band mean E[d] by control with player-bootstrap 95% CIs.
  b) Replication: Spearman(rating, E[d]) with 95% player-bootstrap CI, by control x month
     (replication months use their own tensors: rating-conditioned Maia, no rating in
     context, flat depth prior -- which on 2025-09 gives the same answer as fully blind).
Run:  uv run python -m src.fig_player_depth
"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import figstyle
from analyze import band_of, BANDS, BAND_MID

TC = ["classical", "rapid", "blitz", "bullet"]
rng = np.random.default_rng(0)


def band_means(df):
    out = []
    for b in range(len(BANDS)):
        v = df.loc[df["band"] == b, "Ed"].to_numpy()
        if len(v) < 10: out.append((np.nan, np.nan, np.nan)); continue
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
        out.append((v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    return np.array(out)


def rho_ci(df):
    r = df[["elo", "Ed"]].to_numpy(); n = len(r)
    if n < 30: return np.nan, np.nan, np.nan
    bs = [spearmanr(r[i, 0], r[i, 1]).correlation for i in (rng.integers(0, n, n) for _ in range(2000))]
    return spearmanr(r[:, 0], r[:, 1]).correlation, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    blind = "data/diag/player_depth_2025-09_blind.parquet"
    months = {os.path.basename(p)[13:20]: p for p in sorted(glob.glob("data/diag/player_depth_20*.parquet")) if "blind" not in p}
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
    # (a) primary, fully blind
    if os.path.exists(blind):
        d = pd.read_parquet(blind); d["band"] = [band_of(e) for e in d["elo"]]
        ends = []
        for t in TC:
            m = band_means(d[d["tc"] == t]); ok = ~np.isnan(m[:, 0]); xs = np.array(BAND_MID)[ok]
            col = figstyle.TC_COLORS[t]
            ax[0].fill_between(xs, m[ok, 1], m[ok, 2], color=col, alpha=0.18, lw=0, zorder=2)
            ax[0].plot(xs, m[ok, 0], marker="o", color=col, zorder=3, **figstyle.ring())
            ends.append((xs[-1], m[ok, 0][-1], t + (" (control)" if t == "bullet" else ""), col))
        ends.sort(key=lambda e: e[1]); last = -1e9
        for ex, ey, lab, col in ends:
            ly = max(ey, last + 0.35)
            figstyle.direct_label(ax[0], ex, ly, lab, col if lab.startswith(("classical", "rapid")) else figstyle.INK2,
                                  dx=6, weight="bold" if lab.startswith(("classical", "rapid")) else "regular"); last = ly
        ax[0].set_xlim(right=BAND_MID[-1] + 90)
        ax[0].set_xlabel("rating (per-control bands)"); ax[0].set_ylabel("per-player depth, flat prior (plies)")
        ax[0].set_title("2025-09, no rating information in the model")
    figstyle.panel_label(ax[0], "a")
    # (b) replication: rho by control x month
    keys = (["2025-09*"] if os.path.exists(blind) else []) + sorted(months)
    tabs = ([("2025-09*", pd.read_parquet(blind))] if os.path.exists(blind) else []) + [(k, pd.read_parquet(months[k])) for k in sorted(months)]
    xoff = np.linspace(-0.3, 0.3, max(len(tabs), 1))
    for j, (k, d) in enumerate(tabs):
        for i, t in enumerate(TC):
            r, lo, hi = rho_ci(d[d["tc"] == t])
            if np.isnan(r): continue
            ax[1].errorbar(i + xoff[j], r, yerr=[[r - lo], [hi - r]], fmt="o", ms=4.5, lw=1.6, capsize=2.5,
                           color=figstyle.TC_COLORS[t], alpha=0.55 + 0.45 * (j == 0), zorder=3)
    figstyle.zero_line(ax[1]); ax[1].set_xticks(range(len(TC))); ax[1].set_xticklabels(TC)
    ax[1].set_ylabel("Spearman(rating, per-player depth)"); ax[1].set_title("replication across months")
    ax[1].text(0.02, 0.97, "points, left to right within each control:\n" + ", ".join(k for k, _ in tabs)
               + "\n(* fully rating-blind; others: no rating in the depth prior)",
               transform=ax[1].transAxes, fontsize=6.8, color=figstyle.INK2, va="top", ha="left")
    figstyle.panel_label(ax[1], "b")
    figstyle.save(fig, "paper/figs/fig2_player_depth_blind.png"); print("wrote paper/figs/fig2_player_depth_blind.png")


if __name__ == "__main__":
    main()
