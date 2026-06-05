"""
Smoke check: load the trained model + tensors and confirm depth_of_satisficing returns
sane values (finite, inside the depth grid, non-degenerate spread). Prints E[d] summary
and a coarse E[d]-vs-Elo trend (the paper's headline direction, not a result on smoke data).

Run:
    python src/smoke_check.py --config config.yaml
"""
import argparse

import numpy as np
import torch
import yaml

from model import SatisficingModel


def main(cfg):
    blob = torch.load(cfg["data"]["train_tensor"])
    delta, logq, mask, ctx, y = (blob[k] for k in
                                 ["delta", "logq", "move_mask", "context", "y"])
    dmask = blob.get("depth_mask")
    grid = cfg["model"]["depth_grid"]

    ckpt = torch.load("data/model.pt", weights_only=False)
    model = SatisficingModel(grid, ctx.shape[-1], cfg["model"]["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dhat, r = model.depth_of_satisficing(delta, logq, mask, ctx, y, dmask)
    dhat = dhat.numpy()
    lo, hi = min(grid), max(grid)

    print(f"decisions: {len(dhat)}")
    print(f"E[d]: mean={dhat.mean():.2f}  sd={dhat.std():.2f}  "
          f"min={dhat.min():.2f}  max={dhat.max():.2f}  (grid {lo}..{hi})")
    print(f"alpha={model.alpha.item():.3f}  beta={model.beta.item():.3f}")

    elo = np.array(blob["meta"]["elo"])
    if len(np.unique(elo)) > 1:
        rho = np.corrcoef(elo, dhat)[0, 1]
        print(f"corr(E[d], Elo) = {rho:+.3f}  (sign is the headline direction; "
              f"magnitude meaningless on random smoke games)")

    ok = (np.isfinite(dhat).all() and dhat.min() >= lo - 1e-3 and dhat.max() <= hi + 1e-3
          and dhat.std() > 1e-3)
    print("SMOKE CHECK:", "PASS" if ok else "FAIL (degenerate / out-of-grid depth)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
