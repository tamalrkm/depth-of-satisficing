"""
Stage 7: the results (E1-E6 of the manuscript).

Implemented so far:
  E4/Fig4  held-out move prediction: fusion vs state-only (Maia-3, beta=0) and search-only
           (alpha=0). Held-out cross-entropy (nats) + top-1 match, overall and stratified by
           time_class x swing_class -> the pre-registered differential interaction.

Models are re-fit here under one fixed player split so train/val are identical across
predictors (a fair comparison; does not depend on a previously saved model.pt).

Run:
    python src/analyze.py --config config.yaml --result e4
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import TensorDataset, DataLoader

from model import SatisficingModel

FREEZE = -30.0   # softplus(-30) ~ 1e-13 ~ 0: freezes a gate (beta or alpha) effectively off
FIGDIR = "paper/figs"
BANDS = [(1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000),
         (2000, 2200), (2200, 2400), (2400, 2600), (2600, 4000)]
BAND_MID = [1300, 1500, 1700, 1900, 2100, 2300, 2500, 2650]


def player_split(players, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(players)))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val = set(uniq[:n_val])
    is_val = np.array([p in val for p in players])
    return ~is_val, is_val


def load(cfg):
    blob = torch.load(cfg["data"]["train_tensor"])
    keys = ["delta", "logq", "move_mask", "context", "y"]
    delta, logq, mask, ctx, y = (blob[k] for k in keys)
    dmask = blob.get("depth_mask")
    if dmask is None:
        dmask = torch.ones(delta.shape[0], delta.shape[2])
    return blob, delta, logq, mask, dmask, ctx, y


def fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr, freeze_beta=False, freeze_alpha=False):
    """Fit a SatisficingModel on the train rows; optionally freeze a gate off."""
    model = SatisficingModel(mc["depth_grid"], ctx.shape[-1], mc["hidden"]).to(dev)
    if freeze_beta:
        model._beta.data.fill_(FREEZE); model._beta.requires_grad_(False)
    if freeze_alpha:
        model._alpha.data.fill_(FREEZE); model._alpha.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=mc["lr"])
    idx = np.where(tr)[0]
    ds = TensorDataset(delta[idx], logq[idx], mask[idx], dmask[idx], ctx[idx], y[idx])
    dl = DataLoader(ds, batch_size=mc["batch_size"], shuffle=True)
    for _ in range(mc["epochs"]):
        for d, q, m, dm, c, t in dl:
            loss, _ = model.loss(d.to(dev), q.to(dev), m.to(dev), c.to(dev), t.to(dev),
                                 mc["entropy_reg"], dm.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def per_decision(model, dev, delta, logq, mask, dmask, ctx, y, idx):
    """Held-out per-decision NLL (nats) and top-1 correctness."""
    ds = TensorDataset(delta[idx], logq[idx], mask[idx], dmask[idx], ctx[idx], y[idx])
    nll, top1 = [], []
    for d, q, m, dm, c, t in DataLoader(ds, batch_size=8192):
        p_i, _, _ = model.forward(d.to(dev), q.to(dev), m.to(dev), c.to(dev), dm.to(dev))
        py = p_i.gather(1, t.to(dev).unsqueeze(1)).squeeze(1).clamp_min(1e-9)
        nll.append((-py.log()).cpu()); top1.append((p_i.argmax(1) == t.to(dev)).cpu())
    return torch.cat(nll).numpy(), torch.cat(top1).numpy()


@torch.no_grad()
def maia_raw(logq, mask, y, idx):
    """Uncalibrated Maia-3 alone: softmax over the legal-masked policy logits."""
    lg = logq[idx].clone()
    lg[~mask[idx].bool()] = -1e9
    z = torch.logsumexp(lg, dim=1)
    nll = -(lg.gather(1, y[idx].unsqueeze(1)).squeeze(1) - z)
    top1 = (lg.argmax(1) == y[idx])
    return nll.numpy(), top1.numpy()


def e4(cfg):
    mc = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]
    players = meta["player"]
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    vidx = np.where(va)[0]
    print(f"device={dev}  decisions={len(y)}  held-out={va.sum()} "
          f"({len(set(np.array(players)[va]))} players)")

    print("fitting fusion (alpha,beta free)...")
    fusion = fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr)
    print("fitting state-only (beta=0, Maia + learned depth-free temperature)...")
    state = fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr, freeze_beta=True)
    print("fitting search-only (alpha=0, regret mixture, no Maia)...")
    search = fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr, freeze_alpha=True)

    f_nll, f_t1 = per_decision(fusion, dev, delta, logq, mask, dmask, ctx, y, vidx)
    s_nll, s_t1 = per_decision(state, dev, delta, logq, mask, dmask, ctx, y, vidx)
    r_nll, r_t1 = per_decision(search, dev, delta, logq, mask, dmask, ctx, y, vidx)
    m_nll, m_t1 = maia_raw(logq, mask, y, vidx)

    def line(name, nll, t1):
        print(f"  {name:22s} NLL={nll.mean():.4f}  top1={100*t1.mean():.2f}%")
    print("\n=== HELD-OUT (by player) ===")
    line("Maia-3 raw", m_nll, m_t1)
    line("state-only (beta=0)", s_nll, s_t1)
    line("search-only (alpha=0)", r_nll, r_t1)
    line("fusion", f_nll, f_t1)
    gap = s_nll.mean() - f_nll.mean()
    print(f"\nfusion vs state-only:  Delta NLL = {gap:+.4f} nats  "
          f"(top1 {100*(f_t1.mean()-s_t1.mean()):+.2f} pts)   [positive = fusion better]")

    # --- pre-registered differential: gap by time_class x swing_class ---
    tc = np.array(meta["time_class"])[vidx]
    sw = np.array(meta["swing"])[vidx]
    d_gap = s_nll - f_nll   # per-decision improvement from fusion
    print("\n=== differential gain (Delta NLL, fusion - state) by stratum ===")
    print(f"  {'stratum':28s} {'n':>8s} {'dNLL':>9s}")
    for t in ["classical", "rapid", "blitz", "bullet"]:
        for s in ["down", "up"]:
            sel = (tc == t) & (sw == s)
            if sel.sum():
                print(f"  {t+'/'+s+' swing':28s} {sel.sum():>8d} {d_gap[sel].mean():>+9.4f}")
    # headline interaction contrast: (classical,down) vs (blitz,up)
    a = d_gap[(tc == "classical") & (sw == "down")]
    b = d_gap[(tc == "blitz") & (sw == "up")]
    if len(a) and len(b):
        print(f"\ninteraction: gain(classical,swing-down)={a.mean():+.4f}  "
              f"vs gain(blitz,swing-up)={b.mean():+.4f}  "
              f"contrast={a.mean()-b.mean():+.4f} nats")
        print("(pre-registered: fusion should help most in classical/high-swing, ~0 in blitz/low-swing)")


@torch.no_grad()
def dhat_over(model, dev, delta, logq, mask, dmask, ctx, y, idx):
    """Per-decision posterior depth E[d] on the given rows."""
    ds = TensorDataset(delta[idx], logq[idx], mask[idx], dmask[idx], ctx[idx], y[idx])
    out = []
    for d, q, m, dm, c, t in DataLoader(ds, batch_size=8192):
        dh, _ = model.depth_of_satisficing(d.to(dev), q.to(dev), m.to(dev), c.to(dev),
                                           t.to(dev), dm.to(dev))
        out.append(dh.cpu())
    return torch.cat(out).numpy()


def band_of(elo):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= elo < hi:
            return i
    return -1


def e1(cfg):
    """Fig1: played-move regret across depth, by rating band and swing class.
    Straight from train.pt (delta[n, y[n], :] is the played move's depth-resolved regret)."""
    os.makedirs(FIGDIR, exist_ok=True)
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]
    grid = cfg["model"]["depth_grid"]
    elo = np.array(meta["elo"]); sw = np.array(meta["swing"])
    pr = delta[np.arange(len(y)), y].numpy()        # [N, D] played-move regret on the grid
    bands = np.array([band_of(e) for e in elo])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(BANDS)))
    for title, mask_sw, axi in [("all decisions", np.ones(len(y), bool), ax[0]),
                                ("swing-down (informative)", sw == "down", ax[1])]:
        for bi in range(len(BANDS)):
            sel = (bands == bi) & mask_sw
            if sel.sum() > 50:
                axi.plot(grid, pr[sel].mean(0), color=cmap[bi], label=f"{BAND_MID[bi]}")
        axi.set_title(title); axi.set_xlabel("engine search depth")
        axi.grid(alpha=.3)
    ax[0].set_ylabel("mean regret of played move (win-prob)")
    ax[1].legend(title="rating", fontsize=7, ncol=2)
    fig.tight_layout(); p = f"{FIGDIR}/fig1_error_vs_depth.png"; fig.savefig(p, dpi=150)
    # quantify: how much more rating-separated is swing-down at deep vs shallow?
    deep = -1
    print(f"E1 -> {p}")
    for label, msk in [("all", np.ones(len(y), bool)), ("swing-down", sw == "down")]:
        bm = np.array([pr[(bands == bi) & msk, deep].mean() for bi in range(len(BANDS))])
        print(f"  {label:11s} played-regret @depth{grid[deep]} by band: "
              + " ".join(f"{v:.3f}" for v in bm)
              + f"   (spread hi-lo={bm.max()-bm.min():.3f})")


def e2(cfg):
    """Fig2a: depth of satisficing vs rating, bootstrap CIs over players.
    Fig2b: within-player change under time pressure (E[d] vs think-time quartile)."""
    os.makedirs(FIGDIR, exist_ok=True)
    mc = cfg["model"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]; players = np.array(meta["player"]); elo = np.array(meta["elo"])
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    vidx = np.where(va)[0]
    print("E2: fitting fusion...")
    model = fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr)
    dh = dhat_over(model, dev, delta, logq, mask, dmask, ctx, y, vidx)
    vb = np.array([band_of(e) for e in elo[vidx]]); vpl = players[vidx]

    # Fig2a: per-band mean E[d] with player-cluster bootstrap 95% CI (bootstrap per-player means)
    rng = np.random.default_rng(0); means, los, his = [], [], []
    for bi in range(len(BANDS)):
        sel = vb == bi
        if sel.sum() < 50:
            means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
        ps = np.unique(vpl[sel])
        pmean = np.array([dh[sel & (vpl == p)].mean() for p in ps])   # per-player mean E[d]
        means.append(pmean.mean())
        bs = pmean[rng.integers(0, len(pmean), size=(1000, len(pmean)))].mean(1)
        los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
    means = np.array(means)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(BAND_MID, means, yerr=[means-np.array(los), np.array(his)-means],
                   marker="o", capsize=3)
    ax[0].set_xlabel("rating"); ax[0].set_ylabel("depth of satisficing  E[d]")
    ax[0].set_title("(a) depth rises with skill"); ax[0].grid(alpha=.3)
    rho = spearmanr(elo[vidx], dh).correlation
    print(f"E2a per-band E[d]: " + " ".join(f"{m:.2f}" for m in means))
    print(f"   Spearman(E[d], rating) held-out = {rho:+.3f}")

    # Fig2b: within-player time-pressure effect. Need think-time -> join selected by pos_id.
    sel_df = pd.read_parquet(cfg["data"]["selected"], columns=["pos_id", "time_spent"]).set_index("pos_id")
    pid = np.array(meta["pos_id"])[vidx]
    tspent = sel_df.reindex(pid)["time_spent"].to_numpy()
    ok = np.isfinite(tspent) & (tspent >= 0)
    # within-player: per player, correlate E[d] with log time; report mean slope sign + de-meaned bins
    dq = np.full(len(dh), np.nan)
    for p in np.unique(vpl[ok]):
        m = ok & (vpl == p)
        if m.sum() >= 20:
            dq[m] = dh[m] - dh[m].mean()                  # de-mean within player
    q = np.full(len(dh), -1)
    tt = np.where(ok, tspent, np.nan)
    valid = np.isfinite(dq) & np.isfinite(tt)
    quart = np.quantile(tt[valid], [.25, .5, .75])
    q[valid] = np.digitize(tt[valid], quart)
    binmeans = [dq[valid & (q == k)].mean() for k in range(4)]
    ax[1].plot(["Q1 (fast)", "Q2", "Q3", "Q4 (slow)"], binmeans, marker="o")
    ax[1].axhline(0, color="grey", lw=.7); ax[1].set_title("(b) within-player vs think-time")
    ax[1].set_ylabel("E[d] - player mean"); ax[1].grid(alpha=.3)
    fig.tight_layout(); p = f"{FIGDIR}/fig2_depth_rating_time.png"; fig.savefig(p, dpi=150)
    print(f"E2 -> {p}")
    print(f"E2b within-player de-meaned E[d] by think-time quartile: "
          + " ".join(f"{v:+.3f}" for v in binmeans) + "  (expect rising fast->slow)")


def e3(cfg):
    """Fig3: convergent validity. Refit fusion with CLOCK ablated (not an input), then
    correlate held-out E[d] with observed think-time -- a non-circular validation."""
    os.makedirs(FIGDIR, exist_ok=True)
    mc = cfg["model"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]; players = np.array(meta["player"])
    ctx = ctx.clone(); ctx[:, 1] = 0.0                    # zero clock feature (context[1]=clock)
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    vidx = np.where(va)[0]
    print("E3: fitting CLOCK-FREE fusion (clock feature zeroed)...")
    model = fit(mc, dev, delta, logq, mask, dmask, ctx, y, tr)
    dh = dhat_over(model, dev, delta, logq, mask, dmask, ctx, y, vidx)

    sel_df = pd.read_parquet(cfg["data"]["selected"], columns=["pos_id", "time_spent"]).set_index("pos_id")
    pid = np.array(meta["pos_id"])[vidx]; ply = np.array(meta["ply"])[vidx]
    tspent = sel_df.reindex(pid)["time_spent"].to_numpy()
    ok = np.isfinite(tspent) & (tspent > 0)
    phases = [("opening", ply <= 24), ("middlegame", (ply > 24) & (ply <= 60)),
              ("endgame", ply > 60), ("all", np.ones(len(dh), bool))]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    print("E3 Spearman(E[d] [clock-free], think-time), held out:")
    for name, pmask in phases:
        m = ok & pmask
        if m.sum() > 100:
            rho = spearmanr(dh[m], tspent[m]).correlation
            print(f"   {name:11s} rho={rho:+.3f}  (n={m.sum()})")
            if name != "all":
                ax.bar(name, rho)
    ax.axhline(0, color="grey", lw=.7); ax.set_ylabel("Spearman rho (E[d], think-time)")
    ax.set_title("E3: inferred depth vs real think-time\n(model fit without clock)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = f"{FIGDIR}/fig3_depth_vs_time.png"; fig.savefig(p, dpi=150)
    print(f"E3 -> {p}")


def main(cfg, result):
    runners = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}
    todo = list(runners) if result == "all" else [result]
    for r in todo:
        if r not in runners:
            raise SystemExit(f"result '{r}' not implemented (have: {', '.join(runners)} | all)")
        print(f"\n########## {r.upper()} ##########")
        runners[r](cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--result", default="all", help="which result: e1 | e2 | e3 | e4 | all")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.result)
