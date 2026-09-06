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


def run(tensor, cfg, label, out_parquet=None):
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
    rng = np.random.default_rng(0); saved = []
    def boot_rho(r):
        bs = [spearmanr(r[i, 0], r[i, 1]).correlation for i in (rng.integers(0, len(r), len(r)) for _ in range(2000))]
        return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    for scope, sub in [("all players", df), ("held-out players only", df[df["val"]])]:
        print(f"  --- {scope}: flat-prior per-(player, control) E[d], cells with >= {MIN_DEC} decisions")
        print(f"  {'control':10s} {'players':>8s} {'rho [95% CI]':>24s}   per-band mean E[d]")
        for t in TC:
            g = sub[sub["tc"] == t]
            cells = g.groupby("player").indices
            rows = []
            for p, idx in cells.items():
                if len(idx) < MIN_DEC: continue
                L = lp[g.index.values[idx]].sum(0); L -= L.max(); w = np.exp(L); w /= w.sum()
                rows.append((g["elo"].values[idx].mean(), (w * grid).sum()))
            if len(rows) < 30: print(f"  {t:10s} {len(rows):>8d}   (too few)"); continue
            r = np.array(rows); rho = spearmanr(r[:, 0], r[:, 1]).correlation; lo, hi = boot_rho(r)
            bands = [band_of(e) for e in r[:, 0]]
            bm = [r[np.array(bands) == b, 1].mean() if (np.array(bands) == b).sum() >= 10 else np.nan for b in range(len(BANDS))]
            print(f"  {t:10s} {len(rows):>8d} {rho:>+8.3f} [{lo:+.3f},{hi:+.3f}]   " + " ".join(f"{v:.2f}" if not np.isnan(v) else "-" for v in bm))
            if scope == "all players":
                saved += [{"label": label, "tc": t, "elo": e, "Ed": d_} for e, d_ in r]
    if out_parquet and saved:
        pd.DataFrame(saved).to_parquet(out_parquet, index=False); print(f"  saved per-player table -> {out_parquet}")


MONTHS = {"2025-09": "data/train.pt", "2026-04": "data/repl04/train.pt",
          "2026-05": "data/repl/train.pt", "2026-06": "data/repl06/train.pt"}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--all-months", action="store_true"); a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml")); os.makedirs("data/diag", exist_ok=True)
    if os.path.exists("data/train_fixed1950.pt"):
        run("data/train_fixed1950.pt", yaml.safe_load(open("config_fixed1950.yaml")),
            "2025-09 FULLY RATING-BLIND (fixed-1950 Maia, no rating in context, flat depth prior)",
            "data/diag/player_depth_2025-09_blind.parquet")
    for m, t in (MONTHS.items() if a.all_months else list(MONTHS.items())[:1]):
        if os.path.exists(t):
            run(t, cfg, f"{m} (rating-conditioned Maia, no rating in context, flat depth prior)",
                f"data/diag/player_depth_{m}.parquet")


if __name__ == "__main__":
    main()
