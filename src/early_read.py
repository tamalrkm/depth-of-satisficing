"""
Early-read analysis: Maia-alone vs fusion (top-1, cross-entropy) and E[d] by source.

This is the headline strengthening test on the broadcast (elite-OTB) stratum: a searchless
state-only model (Maia) should fail hardest where deep deliberation matters most, and the
depth-aware fusion should recover the gap. If E[d] is also high on the broadcast stratum,
the recovery is attributable to depth (not merely OOD).

Run after build_dataset + train:
    python src/early_read.py --config config.yaml [--model data/model.pt]
"""
import argparse

import numpy as np
import torch
import yaml

from model import SatisficingModel

NEG = -1e9


def stratum_stats(name, sel, maia_match, ce_maia, fusion_match, ce_fusion, dhat, elo):
    n = int(sel.sum())
    if n == 0:
        return
    print(f"=== {name}  n={n}  median Elo={int(np.median(elo[sel]))} ===")
    print(f"  Maia-alone  top-1: {maia_match[sel].mean():.3f}   CE (nats): {ce_maia[sel].mean():.3f}")
    print(f"  Fusion      top-1: {fusion_match[sel].mean():.3f}   CE (nats): {ce_fusion[sel].mean():.3f}")
    print(f"  Δ (fusion - Maia)  top-1: {fusion_match[sel].mean() - maia_match[sel].mean():+.3f}"
          f"   ΔCE: {ce_fusion[sel].mean() - ce_maia[sel].mean():+.3f}")
    print(f"  inferred E[d]:     mean={dhat[sel].mean():.2f}  sd={dhat[sel].std():.2f}")


def main(cfg, model_path):
    blob = torch.load(cfg["data"]["train_tensor"], weights_only=False)
    delta, logq, mask, ctx, y = (blob[k] for k in
                                 ["delta", "logq", "move_mask", "context", "y"])
    dmask = blob.get("depth_mask")
    meta = blob["meta"]
    src = np.array(meta["source"])
    elo = np.array(meta["elo"])
    time_class = np.array(meta["time_class"])

    ckpt = torch.load(model_path, weights_only=False)
    mc = cfg["model"]
    model = SatisficingModel(mc["depth_grid"], ctx.shape[-1], mc["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Maia-alone = softmax over logq masked to legal moves
    mask_bool = mask.bool()
    logq_masked = logq.masked_fill(~mask_bool, NEG)
    maia_top = logq_masked.argmax(dim=1)
    maia_match = (maia_top == y).float().numpy()
    log_z = torch.logsumexp(logq_masked, dim=1)
    ce_maia = (-(logq.gather(1, y.unsqueeze(1)).squeeze(1) - log_z)).numpy()

    # Fusion = the trained model
    with torch.no_grad():
        p_i, _pi, _pd = model(delta, logq, mask, ctx, dmask)
        dhat, _r = model.depth_of_satisficing(delta, logq, mask, ctx, y, dmask)
    fusion_top = p_i.argmax(dim=1)
    fusion_match = (fusion_top == y).float().numpy()
    ce_fusion = (-(p_i.gather(1, y.unsqueeze(1)).squeeze(1).clamp_min(1e-12).log())).numpy()
    dhat = dhat.numpy()

    print(f"decisions: {len(src)}  alpha={model.alpha.item():.3f}  beta={model.beta.item():.3f}")

    for stratum in sorted(set(src.tolist())):
        sel = (src == stratum)
        stratum_stats(stratum, sel, maia_match, ce_maia, fusion_match, ce_fusion, dhat, elo)

    # also: online classical (the deep-online subset) for comparison with elite-OTB classical
    online_classical = (src == "online") & (time_class == "classical")
    if online_classical.any():
        print()
        stratum_stats("online classical", online_classical,
                      maia_match, ce_maia, fusion_match, ce_fusion, dhat, elo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="data/model.pt")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.model)
