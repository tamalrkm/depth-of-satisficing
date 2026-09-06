"""Non-circular E2: per-PLAYER depth from the move likelihood alone, flat prior over depth.

Per decision the move evidence about depth is weak (the posterior update is ~uncorrelated with
rating), but a player contributes many decisions. Here we pool them: with the fitted (alpha,
beta) from an ELO-FREE-context model, compute log P(y_i | d) for every decision on the depth
grid, sum over a player's decisions within a time control, and take the flat-prior posterior
over d. No rating enters any prior. Then: Spearman(player rating, player E[d]) within control.
Runs on the original tensor (rating-conditioned Maia) and, if present, the fixed-rating-Maia
tensor (fully rating-blind).  Run:  uv run python -m src.diag_player_depth
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from analyze import fit, player_split, band_of, BANDS  # noqa: E402

TC = ["classical", "rapid", "blitz", "bullet"]; MIN_DEC = 30


@torch.no_grad()
def logp_y_given_d(model, dev, delta, logq, mask, dmask, ctx, y):
    ds = TensorDataset(delta, logq, mask, dmask, ctx, y); out = []
    for d, q, m, dm, c, t in DataLoader(ds, batch_size=8192):
        _, _, p = model.forward(d.to(dev), q.to(dev), m.to(dev), c.to(dev), dm.to(dev))
        py = p.gather(1, t.to(dev).view(-1, 1, 1).expand(-1, 1, p.shape[-1])).squeeze(1)   # [B, D]
        lp = py.clamp_min(1e-9).log()
        lp = lp.masked_fill(dm.to(dev) == 0, 0.0)          # unreached depths contribute nothing
        out.append(lp.cpu())
    return torch.cat(out).numpy()


def run(tensor, cfg, label):
    mc = cfg["model"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(tensor, map_location="cpu", weights_only=False)
    delta, logq, mask, ctx, y = (blob[k] for k in ["delta", "logq", "move_mask", "context", "y"])
    dmask = blob.get("depth_mask", torch.ones(delta.shape[0], delta.shape[2]))
    meta = blob["meta"]; players = np.array(meta["player"]); elo = np.array(meta["elo"]); tc = np.array(meta["time_class"])
    src = np.array(meta["source"])
    ctx2 = ctx.clone(); ctx2[:, 0] = 0.0
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    print(f"\n[{label}] fitting elo-free model for (alpha, beta) ...")
    model = fit(mc, dev, delta, logq, mask, dmask, ctx2, y, tr)
    print(f"  alpha={float(model.alpha):.3f} beta={float(model.beta):.3f}")
    lp = logp_y_given_d(model, dev, delta, logq, mask, dmask, ctx2, y)        # [N, D]
    grid = np.array(mc["depth_grid"], float)
    df = pd.DataFrame({"player": players, "elo": elo, "tc": tc, "src": src, "val": va})
    df = df[df["src"] == "online"]
    for scope, sub in [("all players", df), ("held-out players only", df[df["val"]])]:
        print(f"  --- {scope}: flat-prior per-(player, control) E[d], cells with >= {MIN_DEC} decisions")
        print(f"  {'control':10s} {'players':>8s} {'rho(rating, E[d])':>18s}   per-band mean E[d]")
        for t in TC:
            g = sub[sub["tc"] == t]
            cells = g.groupby("player").indices
            rows = []
            for p, idx in cells.items():
                if len(idx) < MIN_DEC: continue
                L = lp[g.index.values[idx]].sum(0); L -= L.max(); w = np.exp(L); w /= w.sum()
                rows.append((g["elo"].values[idx].mean(), (w * grid).sum()))
            if len(rows) < 30: print(f"  {t:10s} {len(rows):>8d}   (too few)"); continue
            r = np.array(rows); rho = spearmanr(r[:, 0], r[:, 1]).correlation
            bands = [band_of(e) for e in r[:, 0]]
            bm = [r[np.array(bands) == b, 1].mean() if (np.array(bands) == b).sum() >= 10 else np.nan for b in range(len(BANDS))]
            print(f"  {t:10s} {len(rows):>8d} {rho:>+18.3f}   " + " ".join(f"{v:.2f}" if not np.isnan(v) else "-" for v in bm))


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    run(cfg["data"]["train_tensor"], cfg, "ORIGINAL tensor: rating-conditioned Maia, elo-free context, flat depth prior")
    if os.path.exists("data/train_fixed1950.pt"):
        cfgf = yaml.safe_load(open("config_fixed1950.yaml"))
        run("data/train_fixed1950.pt", cfgf, "FIXED-1950 Maia tensor: fully rating-blind, flat depth prior")


if __name__ == "__main__":
    main()
