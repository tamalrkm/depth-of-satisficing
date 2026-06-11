"""
Diagnostics:
  (A) H5 robustness -- single-decision rating R^2 for swing-down (traps) vs swing-up, under a
      2x2 of method choices [raw cross-pool rating vs within-pool z] x [in-sample vs nested-CV],
      on BOTH the 2025-09 primary and the 2026-05 replication. Pins down whether the primary's
      "traps 6x more informative" was a cross-pool / in-sample artifact or a real sample effect.
  (B) Blitz H1 -- why depth~rating vanished in blitz on 2026-05: per-rating-band coverage
      (decisions, players) and per-band E[d], blitz vs rapid (which replicated).

    uv run python src/diag.py
"""
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from analyze import load, player_split, fit, dhat_over, _ridge_cv, band_of, BAND_MID, BANDS
from replicate import pool_of, znorm_within

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def r2_insample(X, t):
    A = np.column_stack([np.ones(len(t)), X])
    w, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ w
    return 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def partA_h5(cfg, label):
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]
    elo = np.array(meta["elo"], float); tc = np.array(meta["time_class"]); src = np.array(meta["source"])
    sw = np.array(meta["swing"])
    pr = delta[np.arange(len(y)), y].numpy()                 # [N,D] played-move regret
    z = znorm_within(elo, pool_of(tc, src))
    down, up = sw == "down", sw == "up"
    dims = list(range(pr.shape[1]))
    print(f"\n##### Part A: H5 swing-down vs swing-up rating R^2  --  {label} "
          f"(down n={int(down.sum()):,}, up n={int(up.sum()):,}) #####")
    print(f"  {'method':38s} {'R2_down':>9s} {'R2_up':>9s}  direction")
    variants = [
        ("raw cross-pool elo, IN-SAMPLE (full vec)", elo, r2_insample),
        ("raw cross-pool elo, nested-CV (full vec)", elo, lambda X, t: _ridge_cv(X, t, dims)),
        ("within-pool z,      IN-SAMPLE (full vec)", z,   r2_insample),
        ("within-pool z,      nested-CV (full vec)", z,   lambda X, t: _ridge_cv(X, t, dims)),
    ]
    for name, target, fn in variants:
        rd = fn(pr[down], target[down]); ru = fn(pr[up], target[up])
        arrow = "down>up" if rd > ru else "up>down"
        ratio = rd / ru if ru > 1e-9 else float("inf")
        print(f"  {name:38s} {rd:>9.4f} {ru:>9.4f}  {arrow} ({ratio:.1f}x)" if np.isfinite(ratio)
              else f"  {name:38s} {rd:>9.4f} {ru:>9.4f}  {arrow}")
    # single deep-regret feature, raw elo in-sample (likeliest 'primary' operationalisation)
    deep = pr[:, -1:]
    rd = r2_insample(deep[down], elo[down]); ru = r2_insample(deep[up], elo[up])
    print(f"  {'raw elo, IN-SAMPLE, DEEP-regret only':38s} {rd:>9.4f} {ru:>9.4f}  "
          f"{'down>up' if rd>ru else 'up>down'}")


def partB_blitz(cfg, label):
    mc = cfg["model"]
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]; players = np.array(meta["player"]); elo = np.array(meta["elo"]); tc = np.array(meta["time_class"])
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    vidx = np.where(va)[0]
    print(f"\n##### Part B: blitz vs rapid H1 coverage & per-band E[d]  --  {label} #####")
    model = fit(mc, DEV, delta, logq, mask, dmask, ctx, y, tr)
    dh = dhat_over(model, DEV, delta, logq, mask, dmask, ctx, y, vidx)
    vtc, vpl, velo = tc[vidx], players[vidx], elo[vidx]
    for t in ["rapid", "blitz"]:
        m = vtc == t
        rho = spearmanr(velo[m], dh[m]).correlation
        print(f"\n  {t}: held-out n={int(m.sum())}, players={len(set(vpl[m]))}, "
              f"rating [{velo[m].min():.0f}, {velo[m].max():.0f}] mean={velo[m].mean():.0f} sd={velo[m].std():.0f}  Spearman={rho:+.3f}")
        print(f"    {'band':>10s} {'n':>6s} {'players':>8s} {'meanElo':>8s} {'E[d]':>7s}")
        for bi in range(len(BANDS)):
            bsel = m & np.array([band_of(e) == bi for e in velo])
            if bsel.sum() == 0:
                continue
            print(f"    {str(BANDS[bi]):>10s} {int(bsel.sum()):>6d} {len(set(vpl[bsel])):>8d} "
                  f"{velo[bsel].mean():>8.0f} {dh[bsel].mean():>7.2f}")


if __name__ == "__main__":
    primary = yaml.safe_load(open("config.yaml"))
    repl = yaml.safe_load(open("config_2026_05.yaml"))
    partA_h5(primary, "2025-09 PRIMARY")
    partA_h5(repl, "2026-05 REPLICATION")
    partB_blitz(repl, "2026-05 REPLICATION")
