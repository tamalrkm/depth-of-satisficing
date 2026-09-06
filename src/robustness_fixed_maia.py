"""Robustness: does 'depth rises with rating' survive a fully RATING-BLIND model?

Rating enters the model at two points: (1) the Maia-3 pattern prior is conditioned on the
side-to-move's rating; (2) context[0] is the normalised rating. A sceptic will ask whether the
depth->rating relationship (E2) is partly manufactured by that conditioning. This script refits
the fusion model with BOTH removed:
  - the prior from a Maia-3 pass run at ONE fixed rating for every position
    (config_fixed1950.yaml; 1950 = online sample median), and
  - context[0] zeroed at fit time (the same idiom as the elo-free E5 refit),
then recomputes E2 (held-out depth vs rating, Spearman within time control, per-band means)
exactly as analyze.e2 does. For comparison it also runs the same elo-free fit on the ORIGINAL
tensor (rating-conditioned prior, context elo zeroed), isolating the prior's contribution.

Run (after the fixed-rating Maia pass + build_dataset):
    uv run python -m src.robustness_fixed_maia
"""
import argparse
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from .analyze import fit, dhat_over, player_split, band_of, BANDS

PAPER = {"classical": +0.40, "rapid": +0.66, "blitz": +0.45, "bullet": -0.04}  # reported E2


def e2_rating_blind(tensor_path, cfg, label):
    mc = cfg["model"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(tensor_path, map_location="cpu", weights_only=False)
    delta, logq, mask, ctx, y = (blob[k] for k in ["delta", "logq", "move_mask", "context", "y"])
    dmask = blob.get("depth_mask", torch.ones(delta.shape[0], delta.shape[2]))
    meta = blob["meta"]; players = np.array(meta["player"]); elo = np.array(meta["elo"])
    tc = np.array(meta["time_class"])
    ctx2 = ctx.clone(); ctx2[:, 0] = 0.0                       # rating-blind context
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    vidx = np.where(va)[0]
    print(f"\n[{label}] fitting (context elo zeroed) ...")
    model = fit(mc, dev, delta, logq, mask, dmask, ctx2, y, tr)
    dh = dhat_over(model, dev, delta, logq, mask, dmask, ctx2, y, vidx)
    vb = np.array([band_of(e) for e in elo[vidx]]); vtc = tc[vidx]; velo = elo[vidx]
    out = {}
    print(f"  {'control':10s} {'n':>7s} {'rho':>7s} {'paper':>7s}   per-band E[d]")
    for t in ["classical", "rapid", "blitz", "bullet"]:
        s = vtc == t
        rho = spearmanr(velo[s], dh[s]).correlation
        means = [dh[s & (vb == b)].mean() if (s & (vb == b)).sum() >= 50 else np.nan
                 for b in range(len(BANDS))]
        out[t] = rho
        print(f"  {t:10s} {s.sum():>7,} {rho:>+7.3f} {PAPER[t]:>+7.2f}   "
              + " ".join(f"{m:.2f}" for m in means if not np.isnan(m)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fixed-config", default="config_fixed1950.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); cfgf = yaml.safe_load(open(a.fixed_config))
    r_orig = e2_rating_blind(cfg["data"]["train_tensor"], cfg,
                             "A: rating-conditioned Maia prior, context elo zeroed")
    r_fix = e2_rating_blind(cfgf["data"]["train_tensor"], cfgf,
                            "B: FIXED-rating Maia prior (1950), context elo zeroed = fully rating-blind")
    print("\nSummary: depth-vs-rating Spearman within control")
    print(f"  {'control':10s} {'paper':>7s} {'A elo-free ctx':>15s} {'B fully blind':>14s}")
    for t in ["classical", "rapid", "blitz", "bullet"]:
        print(f"  {t:10s} {PAPER[t]:>+7.2f} {r_orig[t]:>+15.3f} {r_fix[t]:>+14.3f}")
    print("\nReading: if B keeps the slow-control positives and the bullet null, the depth->rating\n"
          "relationship is not an artefact of the rating-conditioned prior.")


if __name__ == "__main__":
    main()
