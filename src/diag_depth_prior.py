"""Diagnostic: how much of 'depth rises with rating' (E2) is the learned rating prior, and is
that prior supported by held-out move prediction?

Fits the paper's FULL model (rating in context) and an ELO-FREE-context model on the original
tensor (rating-conditioned Maia in both). On held-out players, within each time control:
  * Spearman(rating, posterior E[d])        -- the statistic the paper reports
  * Spearman(rating, prior mean depth)      -- the learned prior pi_d(c) alone, before the move
  * Spearman(rating, posterior - prior)     -- the per-decision move-evidence update
  * held-out NLL, FULL vs ELO-FREE          -- does pi_d(rating) improve move prediction?
                                              (player-clustered bootstrap CI on the difference)
  * mean prior depth by rating band (FULL)  -- what the model learned, band by band.
Run:  uv run python -m src.diag_depth_prior
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from analyze import fit, per_decision, dhat_over, player_split, band_of, BANDS  # noqa: E402

TC = ["classical", "rapid", "blitz", "bullet"]


@torch.no_grad()
def prior_mean_over(model, dev, delta, logq, mask, dmask, ctx, idx):
    ds = TensorDataset(delta[idx], logq[idx], mask[idx], dmask[idx], ctx[idx])
    out = []
    for d, q, m, dm, c in DataLoader(ds, batch_size=8192):
        _, pi, _ = model.forward(d.to(dev), q.to(dev), m.to(dev), c.to(dev), dm.to(dev))
        out.append((pi * model.depth_vals).sum(-1).cpu())
    return torch.cat(out).numpy()


def cluster_boot_mean(x, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    codes, uniq = np.unique(groups, return_inverse=True)[1], np.unique(groups)
    sums = np.bincount(codes, weights=x); cnts = np.bincount(codes)
    G = len(uniq); est = []
    for _ in range(n):
        pick = rng.integers(0, G, G)
        est.append(sums[pick].sum() / cnts[pick].sum())
    return x.mean(), np.percentile(est, 2.5), np.percentile(est, 97.5)


def main():
    cfg = yaml.safe_load(open("config.yaml")); mc = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(cfg["data"]["train_tensor"], map_location="cpu", weights_only=False)
    delta, logq, mask, ctx, y = (blob[k] for k in ["delta", "logq", "move_mask", "context", "y"])
    dmask = blob.get("depth_mask", torch.ones(delta.shape[0], delta.shape[2]))
    meta = blob["meta"]; players = np.array(meta["player"]); elo = np.array(meta["elo"])
    tc = np.array(meta["time_class"])
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"]); vidx = np.where(va)[0]
    ctx_free = ctx.clone(); ctx_free[:, 0] = 0.0

    print("fitting FULL (rating in context) ...");   m_full = fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr)
    print("fitting ELO-FREE context ...");            m_free = fit(mc, dev, delta, logq, mask, dmask, ctx_free, y, tr)

    post_full = dhat_over(m_full, dev, delta, logq, mask, dmask, ctx, y, vidx)
    prior_full = prior_mean_over(m_full, dev, delta, logq, mask, dmask, ctx, vidx)
    post_free = dhat_over(m_free, dev, delta, logq, mask, dmask, ctx_free, y, vidx)
    prior_free = prior_mean_over(m_free, dev, delta, logq, mask, dmask, ctx_free, vidx)
    nll_full, _ = per_decision(m_full, dev, delta, logq, mask, dmask, ctx, y, vidx)
    nll_free, _ = per_decision(m_free, dev, delta, logq, mask, dmask, ctx_free, y, vidx)
    ve, vtc, vpl = elo[vidx], tc[vidx], players[vidx]
    vb = np.array([band_of(e) for e in ve])

    print("\n(1) Decomposing the paper's depth-vs-rating correlation (held-out, within control)")
    print(f"  {'control':10s} {'post(FULL)':>11s} {'prior(FULL)':>12s} {'update(FULL)':>13s} | {'post(FREE)':>11s} {'update(FREE)':>13s}")
    for t in TC:
        s = vtc == t
        r = lambda v: spearmanr(ve[s], v[s]).correlation
        print(f"  {t:10s} {r(post_full):>+11.3f} {r(prior_full):>+12.3f} {r(post_full-prior_full):>+13.3f} | "
              f"{r(post_free):>+11.3f} {r(post_free-prior_free):>+13.3f}")
    print("  post = posterior E[d]; prior = mean of pi_d(c) before seeing the move; update = post - prior")

    print("\n(2) Does a rating-dependent depth prior improve HELD-OUT move prediction?")
    print("    (NLL_free - NLL_full, nats; >0 means rating in the depth prior helps; player-clustered 95% CI)")
    d = nll_free - nll_full
    m, lo, hi = cluster_boot_mean(d, vpl)
    print(f"  overall     {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]   n={len(d):,}")
    for t in TC:
        s = vtc == t; m, lo, hi = cluster_boot_mean(d[s], vpl[s])
        print(f"  {t:10s}  {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]   n={s.sum():,}")

    print("\n(3) What the FULL model learned: mean PRIOR depth by rating band, within control")
    print(f"  {'control':10s} " + " ".join(f"{b[0]:>5d}" for b in BANDS))
    for t in TC:
        s = vtc == t
        row = [prior_full[s & (vb == b)].mean() if (s & (vb == b)).sum() >= 50 else np.nan for b in range(len(BANDS))]
        print(f"  {t:10s} " + " ".join(f"{v:5.2f}" if not np.isnan(v) else "    -" for v in row))


if __name__ == "__main__":
    main()
