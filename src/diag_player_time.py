"""Non-circular E3: within-player, flat-prior. Does the depth that best explains a player's
SLOW decisions exceed the depth that best explains their FAST decisions?

No prior over depth is used. With (alpha, beta) from an elo-free fit, log P(y_i | d) is
computed per decision; for each (player, control) with >= MIN_DEC valid-timed decisions, the
decisions are split at the player's own median time-on-move, the log-likelihoods are summed
within each half, and the flat-prior E[d] of each half is compared (slow - fast), paired
within player. Run:  uv run python -m src.diag_player_time
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, yaml
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset
from analyze import fit, player_split  # noqa: E402

TC = ["classical", "rapid", "blitz", "bullet"]; MIN_DEC = 40


@torch.no_grad()
def logp_y_given_d(model, dev, delta, logq, mask, dmask, ctx, y):
    ds = TensorDataset(delta, logq, mask, dmask, ctx, y); out = []
    for d, q, m, dm, c, t in DataLoader(ds, batch_size=8192):
        _, _, p = model.forward(d.to(dev), q.to(dev), m.to(dev), c.to(dev), dm.to(dev))
        py = p.gather(1, t.to(dev).view(-1, 1, 1).expand(-1, 1, p.shape[-1])).squeeze(1)
        out.append(py.clamp_min(1e-9).log().masked_fill(dm.to(dev) == 0, 0.0).cpu())
    return torch.cat(out).numpy()


def flat_Ed(L, grid):
    L = L - L.max(); w = np.exp(L); w /= w.sum(); return float((w * grid).sum())


def main():
    cfg = yaml.safe_load(open("config.yaml")); mc = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(cfg["data"]["train_tensor"], map_location="cpu", weights_only=False)
    delta, logq, mask, ctx, y = (blob[k] for k in ["delta", "logq", "move_mask", "context", "y"])
    dmask = blob.get("depth_mask", torch.ones(delta.shape[0], delta.shape[2]))
    meta = blob["meta"]; players = np.array(meta["player"]); tc = np.array(meta["time_class"])
    src = np.array(meta["source"]); pid = np.array(meta["pos_id"])
    ctx2 = ctx.clone(); ctx2[:, 0] = 0.0
    tr, _ = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    print("fitting elo-free model for (alpha, beta) ..."); model = fit(mc, dev, delta, logq, mask, dmask, ctx2, y, tr)
    lp = logp_y_given_d(model, dev, delta, logq, mask, dmask, ctx2, y)
    grid = np.array(mc["depth_grid"], float)
    ts = pd.read_parquet(cfg["data"]["selected"], columns=["pos_id", "time_spent"]).set_index("pos_id").reindex(pid)["time_spent"].to_numpy()
    df = pd.DataFrame({"player": players, "tc": tc, "src": src, "t": ts, "i": np.arange(len(y))})
    df = df[(df["src"] == "online") & np.isfinite(df["t"]) & (df["t"] > 0)]

    print(f"\nWithin-player, flat-prior: E[d](slow half) - E[d](fast half), split at each player's median time-on-move")
    print(f"  {'control':10s} {'players':>8s} {'mean diff':>10s} {'95% CI (player bootstrap)':>26s} {'frac>0':>7s} {'Wilcoxon p':>11s}   E[d] fast / slow")
    rng = np.random.default_rng(0)
    for t in TC + ["all"]:
        g = df if t == "all" else df[df["tc"] == t]
        diffs, ef, es = [], [], []
        for p, sub in g.groupby("player"):
            if len(sub) < MIN_DEC: continue
            med = sub["t"].median(); fast = sub[sub["t"] <= med]["i"].values; slow = sub[sub["t"] > med]["i"].values
            if len(fast) < MIN_DEC // 3 or len(slow) < MIN_DEC // 3: continue
            a, b = flat_Ed(lp[fast].sum(0), grid), flat_Ed(lp[slow].sum(0), grid)
            diffs.append(b - a); ef.append(a); es.append(b)
        if len(diffs) < 30: print(f"  {t:10s} {len(diffs):>8d}   (too few)"); continue
        d = np.array(diffs); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"  {t:10s} {len(d):>8d} {d.mean():>+10.3f} [{np.percentile(bs,2.5):>+8.3f}, {np.percentile(bs,97.5):>+8.3f}] "
              f"{(d>0).mean():>7.2f} {wilcoxon(d).pvalue:>11.1e}   {np.mean(ef):.2f} / {np.mean(es):.2f}")


if __name__ == "__main__":
    main()
