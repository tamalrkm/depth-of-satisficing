"""
Q: On OTB-classical decisions where the played move is NOT the engine's best move,
   (a) how accurate is Maia-3 at predicting the played move (move-match), and
   (b) does adding the engine analysis (fusion) recover MORE played moves?

Method: held-out-by-player split (same seed/frac as analyze.e4); fusion refit on the TRAIN
split (no in-sample leak), Maia-3 used raw. Move-match = top-1 (and top-3) over the legal
candidate set. "Not the best move" = the played move has positive engine regret at the
deepest VALID depth (depth_mask-aware).
"""
import sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from analyze import load, player_split, fit, maia_raw

REAL_ERR = 0.02   # >=2% win-prob lost = a "real" (non-dust) sub-optimal move


@torch.no_grad()
def fusion_topk(model, dev, delta, logq, mask, dmask, ctx, y, idx, ks=(1, 3)):
    ds = TensorDataset(delta[idx], logq[idx], mask[idx], dmask[idx], ctx[idx], y[idx])
    hits = {k: [] for k in ks}
    for d, q, m, dm, c, t in DataLoader(ds, batch_size=8192):
        p_i, _, _ = model.forward(d.to(dev), q.to(dev), m.to(dev), c.to(dev), dm.to(dev))
        order = p_i.argsort(dim=1, descending=True)            # [B, M]
        t = t.to(dev).unsqueeze(1)
        for k in ks:
            hits[k].append((order[:, :k] == t).any(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in hits.items()}


@torch.no_grad()
def maia_topk(logq, mask, y, idx, ks=(1, 3)):
    lg = logq[idx].clone()
    lg[~mask[idx].bool()] = -1e9
    order = lg.argsort(dim=1, descending=True)
    t = y[idx].unsqueeze(1)
    return {k: (order[:, :k] == t).any(1).numpy() for k in ks}


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    mc = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]
    players = np.array(meta["player"])
    src = np.array(meta["source"])
    tc = np.array(meta["time_class"])

    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])

    # played move's regret at the deepest VALID depth (depth_mask: prefix of 1s then 0s)
    dm = dmask.numpy()
    D = delta.shape[2]
    lv = (dm.sum(1).astype(int) - 1).clip(0, D - 1)            # last valid depth idx per row
    de = delta.numpy()
    played_reg = de[np.arange(len(y)), y.numpy(), lv]          # [N] full-depth regret of played

    otb_cl = (src == "broadcast") & (tc == "classical")
    print(f"device={dev}  total decisions={len(y):,}  broadcast={int((src=='broadcast').sum()):,}  "
          f"OTB-classical={int(otb_cl.sum()):,}")
    print(f"held-out players={len(set(players[va]))}  OTB-classical held-out={int((va & otb_cl).sum()):,}")

    print("\nfitting fusion (alpha,beta free) on TRAIN split ...")
    fusion = fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr)
    print(f"  fitted: alpha={fusion.alpha.item():.3f}  beta={fusion.beta.item():.3f}")

    def report(sel, label):
        idx = np.where(va & otb_cl & sel)[0]
        if len(idx) == 0:
            print(f"{label}: (empty)"); return
        mk = maia_topk(logq, mask, y, idx)
        fk = fusion_topk(fusion, dev, delta, logq, mask, dmask, ctx, y, idx)
        mn, _ = maia_raw(logq, mask, y, idx)
        print(f"\n{label}  (n={len(idx):,})")
        print(f"  Maia-3   top1={100*mk[1].mean():5.1f}%   top3={100*mk[3].mean():5.1f}%   NLL={mn.mean():.3f}")
        print(f"  Fusion   top1={100*fk[1].mean():5.1f}%   top3={100*fk[3].mean():5.1f}%")
        print(f"  Δ(fusion−maia) top1={100*(fk[1].mean()-mk[1].mean()):+5.1f} pts   "
              f"top3={100*(fk[3].mean()-mk[3].mean()):+5.1f} pts")

    # fraction of OTB-classical decisions where the human played the engine-best move
    vsel = va & otb_cl
    frac_best = float((played_reg[vsel] <= 1e-9).mean())
    print(f"\nOTB-classical held-out: played == engine-best in {100*frac_best:.1f}% of decisions "
          f"(non-best in {100*(1-frac_best):.1f}%)")

    report(np.ones(len(y), bool), "ALL OTB-classical")
    report(played_reg <= 1e-9, "played IS engine-best (regret=0)")
    report(played_reg > 1e-9, "played NOT engine-best (any positive regret)")
    report(played_reg >= REAL_ERR, f"played NOT best — real error (regret ≥ {REAL_ERR:.0%} win-prob)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
