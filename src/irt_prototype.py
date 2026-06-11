"""
IRT prototype: is depth-of-satisficing an item-response model on the PLY scale?

Per decision (position j, player p), from the engine depth trajectories already in train.pt:
  item difficulty  b_j  = position CRITICAL DEPTH: shallowest grid depth at which the
                          final-depth-best move is already the apparent best move.
  response   solve_pj   = player played a near-best move (final-depth regret <= EPS).
  ability               = within-pool rating z (external, non-circular).

Tests (decide whether IRT is worth making the backbone):
  S1  ICC      : P(solve) vs ability, stratified by b_j band -> ordered logistic curves.
  S2  INVARIANCE: per-player theta (the b_j at which P(solve)=0.5, on the ply scale) estimated
                  from classical-only vs rapid-only decisions should AGREE (pool-invariant
                  person measurement = the cross-pool currency). Compare to raw pool ratings.
  S3  DISCRIM=SWING: higher-swing positions have steeper solve-vs-ability slopes.

    uv run python src/irt_prototype.py
"""
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression

import figstyle
import matplotlib.pyplot as plt

EPS = 0.02          # near-best: <=2% win-prob regret at full depth
FIG = "paper/figs_irt"


def fit_logit(x, ybin):
    """1-feature logistic; return (slope, intercept) or None if degenerate."""
    if len(ybin) < 30 or ybin.min() == ybin.max() or np.std(x) < 1e-9:
        return None
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    m.fit(x.reshape(-1, 1), ybin)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def main():
    import os
    os.makedirs(FIG, exist_ok=True)
    cfg = yaml.safe_load(open("config.yaml"))
    grid = np.array(cfg["model"]["depth_grid"])
    blob = torch.load(cfg["data"]["train_tensor"])
    meta = blob["meta"]
    delta = blob["delta"].numpy()                 # [N,M,D] regret per candidate per depth
    move_mask = blob["move_mask"].numpy().astype(bool)
    dmask = blob.get("depth_mask")
    dmask = (dmask.numpy() if dmask is not None else np.ones((delta.shape[0], delta.shape[2]))).astype(bool)
    y = blob["y"].numpy()
    N, M, D = delta.shape
    elo = np.array(meta["elo"], float); tc = np.array(meta["time_class"]); src = np.array(meta["source"])
    player = np.array(meta["player"]); swing = np.array(meta["swing"])

    de = delta.copy()
    de[~move_mask] = np.inf                        # ignore padded candidates
    lv = dmask.sum(1).astype(int) - 1              # last valid depth index
    lv = np.clip(lv, 0, D - 1)
    ar = np.arange(N)

    final_best = de[ar, :, lv].argmin(1)           # [N] best move at full depth
    best_at_d = de.argmin(1)                        # [N,D] apparent best at each depth
    valid = np.arange(D)[None, :] <= lv[:, None]
    mismatch = (best_at_d != final_best[:, None]) & valid   # [N,D] position is misleading at d
    any_mis = mismatch.any(1)
    last_mis = (D - 1) - mismatch[:, ::-1].argmax(1)        # DEEPEST misleading depth
    b_j = np.where(any_mis, grid[np.clip(last_mis, 0, D - 1)], float(grid[0])).astype(float)
    # ^ difficulty = how deep you must search to stop being fooled (captures persistent traps)

    played_reg = de[ar, y, lv]                       # played move full-depth regret
    solve = (played_reg <= EPS).astype(int)

    # ability = within-pool rating z (broadcast = its own 'otb' pool)
    pool = np.where(src == "broadcast", "otb", tc)
    z = np.full(N, np.nan)
    for p in np.unique(pool):
        m = pool == p
        z[m] = (elo[m] - elo[m].mean()) / (elo[m].std() + 1e-9)

    print(f"N={N:,}  solve-rate={solve.mean():.3f}  b_j range [{b_j.min():.0f},{b_j.max():.0f}] "
          f"mean {b_j.mean():.1f}  (EPS={EPS})")

    # ---------- S1: ICC ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    bands = [("easy (b<=6)", b_j <= 6), ("medium (8-12)", (b_j >= 8) & (b_j <= 12)), ("hard (b>=14)", b_j >= 14)]
    zq = np.quantile(z[np.isfinite(z)], np.linspace(0, 1, 11))
    zc = 0.5 * (zq[:-1] + zq[1:])
    print("\nS1 ICC  P(solve) by ability decile x difficulty band:")
    for name, msel in bands:
        s = msel & np.isfinite(z)
        binid = np.clip(np.digitize(z[s], zq[1:-1]), 0, 9)
        rate = np.array([solve[s][binid == k].mean() if (binid == k).sum() > 20 else np.nan for k in range(10)])
        ax[0].plot(zc, rate, marker="o", label=f"{name}  (n={int(s.sum()):,})")
        print(f"   {name:14s} solve@low={np.nanmin(rate):.2f} solve@high={np.nanmax(rate):.2f}")
    ax[0].set_xlabel("ability (within-pool rating z)"); ax[0].set_ylabel("P(play near-best)")
    ax[0].set_title("(a) ICCs by position critical depth"); ax[0].legend(); ax[0].grid(alpha=.3)
    figstyle.panel_label(ax[0], "a")
    # quantify: logistic solve ~ z + b_j
    s = np.isfinite(z)
    X = np.column_stack([z[s], b_j[s]])
    lr = LogisticRegression(C=1e6, max_iter=300).fit(X, solve[s])
    print(f"   logit(solve) ~ z + b_j:  beta_ability={lr.coef_[0,0]:+.3f} (>0 expected)  "
          f"beta_difficulty={lr.coef_[0,1]:+.3f} (<0 expected)")

    # ---------- S2: person-parameter invariance (difficulty-adjusted ability residual) ----------
    # Explanatory-Rasch person estimate that needs few decisions: theta_p = mean over the player's
    # decisions of (solve - E[solve | b_j]).  E[solve|b_j] from a population logistic on b_j.
    s = np.isfinite(b_j)
    pop = LogisticRegression(C=1e6, max_iter=300).fit(b_j[s].reshape(-1, 1), solve[s])
    exp_solve = pop.predict_proba(b_j.reshape(-1, 1))[:, 1]
    resid = solve - exp_solve                                   # ability signal per decision

    def theta_resid(selmask, mindec):
        out = {}
        for p in np.unique(player[selmask]):
            m = selmask & (player == p)
            if m.sum() >= mindec:
                out[p] = resid[m].mean()
        return out

    # (i) split-half reliability: odd vs even decisions of each player (is theta a stable trait?)
    idx = np.arange(N)
    th_odd = theta_resid((idx % 2 == 1), 25); th_even = theta_resid((idx % 2 == 0), 25)
    comm = sorted(set(th_odd) & set(th_even))
    if len(comm) >= 30:
        a = np.array([th_odd[p] for p in comm]); b = np.array([th_even[p] for p in comm])
        rel = pearsonr(a, b)[0]
        print(f"\nS2a SPLIT-HALF reliability of theta (odd vs even decisions, {len(comm)} players): r={rel:+.3f}")
    # (ii) cross-pool invariance where overlap allows
    th_cl = theta_resid((pool == "classical"), 25); th_ra = theta_resid((pool == "rapid"), 25)
    common = sorted(set(th_cl) & set(th_ra))
    print(f"S2b CROSS-POOL: players with theta in BOTH classical & rapid (>=25 each) = {len(common)}")
    if len(common) >= 20:
        a = np.array([th_cl[p] for p in common]); b = np.array([th_ra[p] for p in common])
        r = pearsonr(a, b)
        rcl = np.array([elo[(player == p) & (pool == "classical")].mean() for p in common])
        rra = np.array([elo[(player == p) & (pool == "rapid")].mean() for p in common])
        rr = pearsonr(rcl, rra)
        print(f"   theta_classical vs theta_rapid: r={r[0]:+.3f}   (raw pool ratings: r={rr[0]:+.3f})")
        ax[1].scatter(a, b, s=14, alpha=.5)
        ax[1].set_xlabel(r"$\theta$ classical (ability resid)"); ax[1].set_ylabel(r"$\theta$ rapid")
        ax[1].set_title(f"(b) cross-pool ability\nr={r[0]:.2f} ({len(common)} players)")
    else:
        print("   still too few overlapping players; cross-pool invariance NOT testable on this sample")
        if len(comm) >= 30:
            ax[1].scatter(a if False else [th_odd[p] for p in comm], [th_even[p] for p in comm], s=10, alpha=.4)
            ax[1].set_xlabel(r"$\theta$ odd decisions"); ax[1].set_ylabel(r"$\theta$ even")
            ax[1].set_title(f"(b) split-half reliability r={rel:.2f}")
    figstyle.panel_label(ax[1], "b")

    # ---------- S3: discrimination = swing ----------
    tsw = blob["context"].numpy()[:, 3]              # total position swing magnitude
    swq = np.quantile(tsw, np.linspace(0, 1, 9))
    slopes, mids = [], []
    for k in range(8):
        m = (tsw >= swq[k]) & (tsw < swq[k + 1]) & np.isfinite(z)
        f = fit_logit(z[m], solve[m])
        if f:
            slopes.append(f[0]); mids.append(0.5 * (swq[k] + swq[k + 1]))
    print("\nS3 DISCRIMINATION vs SWING (solve~ability slope by swing octile):")
    print("   swing:", " ".join(f"{m:.2f}" for m in mids))
    print("   slope:", " ".join(f"{s:.2f}" for s in slopes))
    if len(slopes) >= 4:
        rho = spearmanr(mids, slopes).correlation
        print(f"   Spearman(swing, discrimination) = {rho:+.3f}  (>0 = swing is item discrimination)")
        ax[2].plot(mids, slopes, marker="o")
        ax[2].set_xlabel("position swing magnitude"); ax[2].set_ylabel("ICC slope (discrimination)")
        ax[2].set_title(f"(c) swing = discrimination\nSpearman={rho:+.2f}")
        figstyle.panel_label(ax[2], "c")
    fig.tight_layout()
    p = f"{FIG}/irt_prototype.png"; figstyle.save(fig, p)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
