"""
Follow-up: among OTB-classical NON-best played moves, does the engine/fusion recover
swing-DOWN traps (low regret shallow, high regret deep) better than plain blunders
(bad at every depth)? The satisficing mechanism predicts: yes for traps (the depth mixture
sees them looking best at shallow depth), ~no/negative for blunders.

Trap = played move's regret RISES with depth: sw = sum_d (delta_d - delta_D) < 0  (ICMLA 2015).
Also reported with an explicit shallow-vs-deep regret split for interpretability.
"""
import sys
import numpy as np
import torch
import yaml

from analyze import load, player_split, fit, maia_raw
from maia_otb_match import fusion_topk, maia_topk

REAL = 0.02   # real (non-dust) error: >=2% win-prob lost at full depth


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    mc = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = np.array(mc["depth_grid"])
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]
    players = np.array(meta["player"]); src = np.array(meta["source"]); tc = np.array(meta["time_class"])
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])

    N = len(y); yy = y.numpy(); de = delta.numpy(); dm = dmask.numpy(); D = de.shape[2]
    lv = (dm.sum(1).astype(int) - 1).clip(0, D - 1)
    ptraj = de[np.arange(N), yy, :]                       # [N, D] played-move regret per grid depth
    deep = ptraj[np.arange(N), lv]                        # full-depth regret of played move
    sw = ((ptraj - deep[:, None]) * dm).sum(1)           # ICMLA swing (valid depths): <0 = trap
    shallow = ptraj[:, grid <= 8].mean(1)                # mean regret at shallow depths (<=8 ply)

    otb = (src == "broadcast") & (tc == "classical")
    nonbest = deep > 1e-9
    real = deep >= REAL
    print(f"depth grid={list(grid)}  D={D}  device={dev}")

    nb = va & otb & nonbest
    re = va & otb & real
    print(f"OTB-classical held-out non-best={int(nb.sum()):,}  real-error(≥{REAL:.0%})={int(re.sum()):,}")
    print(f"  among non-best:  swing-DOWN traps (sw<0) = {100*(sw[nb]<0).mean():.0f}%   "
          f"plain (sw≥0) = {100*(sw[nb]>=0).mean():.0f}%")
    print(f"  among real-errors: traps = {100*(sw[re]<0).mean():.0f}%   plain = {100*(sw[re]>=0).mean():.0f}%")

    print("\nfitting fusion (alpha,beta free) on TRAIN split ...")
    fusion = fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr)
    print(f"  alpha={fusion.alpha.item():.3f}  beta={fusion.beta.item():.3f}")

    def rep(sel, label):
        idx = np.where(va & otb & sel)[0]
        if len(idx) == 0:
            print(f"  {label}: (empty)"); return
        mk = maia_topk(logq, mask, y, idx); fk = fusion_topk(fusion, dev, delta, logq, mask, dmask, ctx, y, idx)
        print(f"  {label:48s} n={len(idx):6,}  "
              f"Maia {100*mk[1].mean():4.1f}/{100*mk[3].mean():4.1f}  "
              f"Fus {100*fk[1].mean():4.1f}/{100*fk[3].mean():4.1f}  "
              f"Δtop1={100*(fk[1].mean()-mk[1].mean()):+4.1f}  Δtop3={100*(fk[3].mean()-mk[3].mean()):+4.1f}")

    print("\n=== by ICMLA swing sign (top1/top3, Δ = fusion−Maia) ===")
    rep(nonbest & (sw < 0),  "non-best  TRAP  (swing-down, sw<0)")
    rep(nonbest & (sw >= 0), "non-best  PLAIN (sw≥0)")
    rep(real & (sw < 0),     "real-err  TRAP  (swing-down, sw<0)")
    rep(real & (sw >= 0),    "real-err  PLAIN (sw≥0)")

    print("\n=== explicit shallow-vs-deep regret (real errors only) ===")
    rep(real & (shallow <= 0.01), "real-err  looked-GOOD shallow (≤1%) → trap")
    rep(real & (shallow >= 0.02), "real-err  looked-BAD  shallow (≥2%) → blunder")

    # player-clustered bootstrap of the paired Δtop1 (fusion − Maia) for the two key buckets
    rng = np.random.default_rng(0)
    pl = np.array(meta["player"])
    def boot(sel, label):
        idx = np.where(va & otb & sel)[0]
        mh = maia_topk(logq, mask, y, idx)[1].astype(float)
        fh = fusion_topk(fusion, dev, delta, logq, mask, dmask, ctx, y, idx)[1].astype(float)
        d = fh - mh
        gp = pl[idx]
        uniq = np.unique(gp)
        by = {p: np.where(gp == p)[0] for p in uniq}
        means = []
        for _ in range(1000):
            samp = rng.choice(uniq, len(uniq), replace=True)
            rows = np.concatenate([by[p] for p in samp])
            means.append(d[rows].mean())
        lo, hi = np.percentile(means, [2.5, 97.5])
        print(f"  {label:42s} n={len(idx):5,}  Δtop1={100*d.mean():+5.2f} pts  "
              f"95% CI [{100*lo:+5.2f}, {100*hi:+5.2f}]  players={len(uniq)}")
    print("\n=== player-clustered bootstrap (1000x) of Δtop1 ===")
    boot(real & (shallow <= 0.01), "genuine trap (looked-good shallow)")
    boot(real & (shallow >= 0.02), "plain blunder (looked-bad shallow)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
