"""Diagnostic for E3: is 'clock-free depth tracks thinking time' the prior or the move evidence?

The E3 model zeroes only context[1] (REMAINING clock). Time-class one-hot (context[4:8]) and
ply (context[2]) stay in the depth prior, and both co-vary with time spent (median move time
11/5/3/1 s across classical/rapid/blitz/bullet). So a pooled correlation between posterior
depth and time spent can come from the prior alone. This script decomposes it.

Fits (original tensor, rating-conditioned Maia):
  FULL      : paper's full context
  CLOCKFREE : context[1]=0                       (E3's model)
  TIMEBLIND : context[1]=0 and context[4:8]=0    (no clock, no time class)
On held-out players with valid time-on-move, reports Spearman(time spent, ...) for the
posterior, the prior mean, and the update (posterior - prior): pooled (as the paper), by phase
(as the paper), and WITHIN each time class; a within-player version (both de-meaned per
player); and held-out NLL FULL vs CLOCKFREE (does the clock in the prior improve prediction?).
Run:  uv run python -m src.diag_time_prior
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from analyze import fit, per_decision, dhat_over, player_split  # noqa: E402

TC = ["classical", "rapid", "blitz", "bullet"]


@torch.no_grad()
def prior_mean_over(model, dev, delta, logq, mask, dmask, ctx, idx):
    ds = TensorDataset(delta[idx], logq[idx], mask[idx], dmask[idx], ctx[idx]); out = []
    for d, q, m, dm, c in DataLoader(ds, batch_size=8192):
        _, pi, _ = model.forward(d.to(dev), q.to(dev), m.to(dev), c.to(dev), dm.to(dev))
        out.append((pi * model.depth_vals).sum(-1).cpu())
    return torch.cat(out).numpy()


def cluster_boot_mean(x, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.unique(groups, return_inverse=True)[1]
    sums = np.bincount(codes, weights=x); cnts = np.bincount(codes); G = len(sums)
    est = [sums[p].sum() / cnts[p].sum() for p in (rng.integers(0, G, G) for _ in range(n))]
    return x.mean(), np.percentile(est, 2.5), np.percentile(est, 97.5)


def demean_by(x, g):
    s = pd.Series(x); return (s - s.groupby(g).transform("mean")).to_numpy()


def main():
    cfg = yaml.safe_load(open("config.yaml")); mc = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(cfg["data"]["train_tensor"], map_location="cpu", weights_only=False)
    delta, logq, mask, ctx, y = (blob[k] for k in ["delta", "logq", "move_mask", "context", "y"])
    dmask = blob.get("depth_mask", torch.ones(delta.shape[0], delta.shape[2]))
    meta = blob["meta"]; players = np.array(meta["player"]); tc = np.array(meta["time_class"])
    ply = np.array(meta["ply"]); pid = np.array(meta["pos_id"])
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"]); vidx = np.where(va)[0]
    sel = pd.read_parquet(cfg["data"]["selected"], columns=["pos_id", "time_spent"]).set_index("pos_id")
    ts = sel.reindex(pid[vidx])["time_spent"].to_numpy()
    ok = np.isfinite(ts) & (ts > 0)

    c_full = ctx
    c_clk = ctx.clone(); c_clk[:, 1] = 0.0
    c_tb = c_clk.clone(); c_tb[:, 4:8] = 0.0
    models = {}
    for name, c in [("FULL", c_full), ("CLOCKFREE", c_clk), ("TIMEBLIND", c_tb)]:
        print(f"fitting {name} ..."); models[name] = (fit(mc, dev, delta, logq, mask, dmask, c, y, tr), c)

    res = {}
    for name, (m, c) in models.items():
        post = dhat_over(m, dev, delta, logq, mask, dmask, c, y, vidx)
        prior = prior_mean_over(m, dev, delta, logq, mask, dmask, c, vidx)
        res[name] = (post, prior, post - prior)
    vtc, vpl, vply = tc[vidx], players[vidx], ply[vidx]
    lt = np.log(np.where(ok, ts, 1.0))

    def rho(v, s): return spearmanr(lt[s], v[s]).correlation
    for name in ["CLOCKFREE", "TIMEBLIND"]:
        post, prior, upd = res[name]
        print(f"\n[{name}] Spearman(log time spent, .) on held-out decisions")
        print(f"  {'slice':16s} {'n':>7s} {'posterior':>10s} {'prior':>8s} {'update':>8s}")
        slices = [("pooled (paper)", ok),
                  ("opening", ok & (vply <= 24)), ("middlegame", ok & (vply > 24) & (vply <= 60)),
                  ("endgame", ok & (vply > 60))] + [(f"within {t}", ok & (vtc == t)) for t in TC]
        for lab, s in slices:
            print(f"  {lab:16s} {s.sum():>7,} {rho(post,s):>+10.3f} {rho(prior,s):>+8.3f} {rho(upd,s):>+8.3f}")
        # within-player: de-mean both per player (players with >=20 valid held-out decisions)
        cnt = pd.Series(vpl[ok]).value_counts(); keep = ok & (pd.Series(vpl).map(cnt).fillna(0).to_numpy().astype(int) >= 20)   # parenthesised: & binds tighter than >=
        for lab, s in [("within-player (all)", keep)] + [(f"within-player, {t}", keep & (vtc == t)) for t in TC]:
            if s.sum() < 500: continue
            g = vpl[s]
            print(f"  {lab:24s} {s.sum():>7,}  post {spearmanr(demean_by(lt[s],g), demean_by(post[s],g)).correlation:+.3f}"
                  f"  prior {spearmanr(demean_by(lt[s],g), demean_by(prior[s],g)).correlation:+.3f}"
                  f"  update {spearmanr(demean_by(lt[s],g), demean_by(upd[s],g)).correlation:+.3f}")

    print("\nHeld-out NLL: does the CLOCK in the depth prior improve move prediction? (NLL_clockfree - NLL_full)")
    nf, _ = per_decision(models["FULL"][0], dev, delta, logq, mask, dmask, c_full, y, vidx)
    nc, _ = per_decision(models["CLOCKFREE"][0], dev, delta, logq, mask, dmask, c_clk, y, vidx)
    d = nc - nf; m_, lo, hi = cluster_boot_mean(d, vpl)
    print(f"  overall    {m_:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    for t in TC:
        s = vtc == t; m_, lo, hi = cluster_boot_mean(d[s], vpl[s]); print(f"  {t:10s} {m_:+.5f} [{lo:+.5f}, {hi:+.5f}]")


if __name__ == "__main__":
    main()
