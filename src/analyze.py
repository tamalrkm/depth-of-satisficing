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
import figstyle  # noqa: F401  -- sets publication rcParams + palette on import
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

    # --- pre-registered differential #1 (manuscript E4): swing MAGNITUDE x time control ---
    tc = np.array(meta["time_class"])[vidx]
    d_gap = s_nll - f_nll   # per-decision improvement from fusion
    tsw = ctx[vidx, 3].numpy()                       # total swing of position (magnitude)
    hi = tsw >= np.median(tsw)
    print("\n=== registered E4: gain by swing-MAGNITUDE x time control ===")
    print(f"  {'stratum':28s} {'n':>8s} {'dNLL':>9s}")
    for t in ["classical", "rapid", "blitz", "bullet"]:
        for lab, m in [("high-swing", hi), ("low-swing", ~hi)]:
            sel = (tc == t) & m
            if sel.sum():
                print(f"  {t+'/'+lab:28s} {sel.sum():>8d} {d_gap[sel].mean():>+9.4f}")
    a0 = d_gap[(tc == "classical") & hi]; b0 = d_gap[(tc == "blitz") & ~hi]
    if len(a0) and len(b0):
        print(f"  interaction(classical/high - blitz/low) = {a0.mean()-b0.mean():+.4f} nats")

    # --- differential #2 (dissociation bonus): gap by time_class x swing DIRECTION ---
    sw = np.array(meta["swing"])[vidx]
    print("\n=== bonus: gain by swing-DIRECTION x time control ===")
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

    # --- significance via player-clustered bootstrap (robust alt. to the registered mixed model) ---
    # The registered E4 contrast is the SWING-MAGNITUDE x time-control interaction:
    # gain(classical, high-swing) - gain(blitz, low-swing).
    vpl = np.array(players)[vidx]
    pl = np.unique(vpl)
    rng = np.random.default_rng(0)
    by_p = {p: np.where(vpl == p)[0] for p in pl}
    cl_hi = (tc == "classical") & hi          # hi = high swing magnitude (from the E4 block above)
    bl_lo = (tc == "blitz") & ~hi
    inter = d_gap[cl_hi].mean() - d_gap[bl_lo].mean()
    boots_overall, boots_inter = [], []
    for _ in range(1000):
        samp = rng.choice(pl, len(pl), replace=True)
        rows = np.concatenate([by_p[p] for p in samp])
        boots_overall.append(d_gap[rows].mean())
        ai = d_gap[rows][cl_hi[rows]]; bi = d_gap[rows][bl_lo[rows]]
        if len(ai) and len(bi):
            boots_inter.append(ai.mean() - bi.mean())
    bo = np.percentile(boots_overall, [2.5, 97.5])
    bc = np.percentile(boots_inter, [2.5, 97.5])
    print(f"\nplayer-clustered bootstrap (1000x):")
    print(f"  overall fusion-state gain:        {gap:+.4f}  95% CI [{bo[0]:+.4f}, {bo[1]:+.4f}]  "
          f"{'(excludes 0)' if bo[0] > 0 else '(includes 0)'}")
    print(f"  swing-magnitude x TC interaction: {inter:+.4f}  95% CI [{bc[0]:+.4f}, {bc[1]:+.4f}]  "
          f"{'(excludes 0)' if bc[0] > 0 else '(includes 0)'}")

    # --- registered test: linear mixed-effects with player random intercept ---
    # gain ~ classical * high_swing, (1 | player). Interaction = the pre-registered effect.
    import statsmodels.formula.api as smf
    dfm = pd.DataFrame({"gain": d_gap, "classical": (tc == "classical").astype(int),
                        "highswing": hi.astype(int), "player": vpl})
    try:
        mfit = smf.mixedlm("gain ~ classical * highswing", dfm, groups=dfm["player"]).fit()
        co = mfit.params.get("classical:highswing"); pv = mfit.pvalues.get("classical:highswing")
        print(f"\nmixed-effects (gain ~ classical*highswing + 1|player):")
        print(f"  interaction coef = {co:+.4f}  p = {pv:.2e}")
    except Exception as e:
        print(f"\nmixed-effects fit failed: {e}")

    # --- Fig4: gain by time control x swing magnitude ---
    fig, ax = plt.subplots(figsize=(6, 4.2))
    tcs = ["classical", "rapid", "blitz", "bullet"]
    himeans = [d_gap[(tc == t) & hi].mean() for t in tcs]
    lomeans = [d_gap[(tc == t) & ~hi].mean() for t in tcs]
    xx = np.arange(len(tcs))
    ax.bar(xx - 0.2, himeans, 0.38, label="high-swing", color="navy")
    ax.bar(xx + 0.2, lomeans, 0.38, label="low-swing", color="silver")
    ax.axhline(0, color="grey", lw=.7); ax.set_xticks(xx); ax.set_xticklabels(tcs)
    ax.set_ylabel("held-out NLL gain, fusion - Maia (nats)")
    ax.set_title("E4: depth helps only in slow, high-swing play")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = f"{FIGDIR}/fig4_prediction_gain.png"; figstyle.save(fig, p)
    print(f"E4 -> {p}")


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
                                ("swing-up (deep-discovery, informative)", sw == "up", ax[1])]:
        for bi in range(len(BANDS)):
            sel = (bands == bi) & mask_sw
            if sel.sum() > 50:
                axi.plot(grid, pr[sel].mean(0), color=cmap[bi], label=f"{BAND_MID[bi]}")
        axi.set_title(title); axi.set_xlabel("engine search depth")
        axi.grid(alpha=.3)
    ax[0].set_ylabel("mean regret of played move (win-prob)")
    ax[1].legend(title="rating", fontsize=7, ncol=2)
    figstyle.panel_label(ax[0], "a"); figstyle.panel_label(ax[1], "b")
    fig.tight_layout(); p = f"{FIGDIR}/fig1_error_vs_depth.png"; figstyle.save(fig, p)
    # quantify: how much more rating-separated is swing-down at deep vs shallow?
    deep = -1
    print(f"E1 -> {p}")
    for label, msk in [("all", np.ones(len(y), bool)), ("swing-up", sw == "up")]:
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
    vtc = np.array(meta["time_class"])[vidx]

    # Fig2a: depth vs rating WITHIN each time control (pooling confounds skill with clock,
    # since elite/high-rated play is mostly classical). Bullet is the natural negative control.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rng = np.random.default_rng(0)
    print(f"E2a depth vs rating, WITHIN time control (pooled rho={spearmanr(elo[vidx], dh).correlation:+.3f}, confounded):")
    colors = {"classical": "navy", "rapid": "teal", "blitz": "darkorange", "bullet": "grey"}
    for t in ["classical", "rapid", "blitz", "bullet"]:
        tsel = vtc == t
        means, los, his = [], [], []
        for bi in range(len(BANDS)):
            sel = tsel & (vb == bi)
            if sel.sum() < 50:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            ps = np.unique(vpl[sel])
            pmean = np.array([dh[sel & (vpl == p)].mean() for p in ps])
            means.append(pmean.mean())
            bs = pmean[rng.integers(0, len(pmean), size=(1000, len(pmean)))].mean(1)
            los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
        means = np.array(means)
        rho = spearmanr(elo[vidx][tsel], dh[tsel]).correlation
        lab = f"{t} (rho={rho:+.2f})" + ("  [control]" if t == "bullet" else "")
        ok = ~np.isnan(means)
        ax[0].errorbar(np.array(BAND_MID)[ok], means[ok],
                       yerr=[(means-np.array(los))[ok], (np.array(his)-means)[ok]],
                       marker="o", capsize=2, color=colors[t], label=lab)
        print(f"   {t:10s} n={tsel.sum():>6d}  Spearman={rho:+.3f}  "
              f"per-band E[d]: " + " ".join(f"{m:.2f}" for m in means if not np.isnan(m)))
    ax[0].set_xlabel("rating"); ax[0].set_ylabel("depth of satisficing  E[d]")
    ax[0].set_title("(a) depth rises with skill, within time control"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7)

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
    figstyle.panel_label(ax[0], "a"); figstyle.panel_label(ax[1], "b")
    fig.tight_layout(); p = f"{FIGDIR}/fig2_depth_rating_time.png"; figstyle.save(fig, p)
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
    fig.tight_layout(); p = f"{FIGDIR}/fig3_depth_vs_time.png"; figstyle.save(fig, p)
    print(f"E3 -> {p}")


def e6(cfg, syn_cfg_path="config_syn.yaml"):
    """Fig6: recover planted depth from synthetic agents (identifiability).

    Recovery is a per-agent depth estimate from the agents' MOVES, not from context: the
    synthetic agents are pure-search (no Maia patterns) and were generated at one decision
    sharpness, so the estimator uses the search likelihood (alpha=0), fits beta to the
    synthetic population, and reads each agent's depth from a flat-prior posterior. (Applying
    the human-fit context prior instead collapses E[d] to the prior mean -- the per-decision
    move evidence is too weak to move a fixed prior; see the manuscript discussion.)"""
    os.makedirs(FIGDIR, exist_ok=True)
    syn = yaml.safe_load(open(syn_cfg_path))
    blob, delta, logq, mask, dmask, ctx, y = load(syn)
    meta = blob["meta"]
    grid = np.array(cfg["model"]["depth_grid"]); D = len(grid)
    sel = pd.read_parquet(syn["data"]["selected"], columns=["pos_id", "planted_depth"]).set_index("pos_id")
    planted = sel.reindex(np.array(meta["pos_id"]))["planted_depth"].to_numpy()
    agent = np.array(meta["player"])
    mb = mask.bool()

    def logp_yd(beta, alpha=0.0):                         # [N, D] log P(y | depth d)
        logits = (-beta * delta + alpha * logq.unsqueeze(-1)).masked_fill(~mb.unsqueeze(-1), -1e9)
        lp = torch.log_softmax(logits, 1)
        return lp.gather(1, y.view(-1, 1, 1).expand(-1, 1, D)).squeeze(1).numpy()

    from scipy.special import logsumexp
    betas = np.linspace(1, 14, 27)
    beta_hat = betas[int(np.argmax([logsumexp(logp_yd(b) - np.log(D), 1).sum() for b in betas]))]
    L = logp_yd(beta_hat)
    print(f"E6: fitted beta on synthetic = {beta_hat:.1f} (agents generated at beta_gen={cfg['synthetic']['softmax_beta_gen']})")

    agents = np.unique(agent); rec, rec_pl = [], []
    for ag in agents:
        m = agent == ag
        s = L[m].sum(0); r = np.exp(s - s.max()); r /= r.sum()
        rec.append(float((grid * r).sum())); rec_pl.append(int(planted[m][0]))
    rec = np.array(rec); rec_pl = np.array(rec_pl)
    depths = sorted(set(rec_pl))
    gmean = np.array([rec[rec_pl == d].mean() for d in depths])
    gsd = np.array([rec[rec_pl == d].std() for d in depths])
    mae = np.mean(np.abs(rec - rec_pl))
    rho = spearmanr(rec_pl, rec).correlation
    rho_g = spearmanr(depths, gmean).correlation

    fig, ax = plt.subplots(figsize=(5.4, 5))
    lim = [min(depths) - 2, max(depths) + 2]
    ax.plot(lim, lim, "--", color="grey", label="identity")
    ax.scatter(rec_pl + np.random.default_rng(0).normal(0, .15, len(rec_pl)), rec,
               s=6, alpha=.15, color="steelblue")
    ax.errorbar(depths, gmean, yerr=gsd, marker="o", color="navy", capsize=3, label="recovered E[d] (mean±sd)")
    ax.set_xlabel("planted depth $d_{plant}$"); ax.set_ylabel("recovered depth E[d]")
    ax.set_title(f"E6 recovery: per-agent MAE={mae:.1f}, group $\\rho$={rho_g:.2f}")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); p = f"{FIGDIR}/fig6_synthetic_recovery.png"; figstyle.save(fig, p)
    print("E6 recovery (planted -> recovered E[d], per-agent mean±sd):")
    for d, m, s in zip(depths, gmean, gsd):
        print(f"   d_plant={d:2d} -> E[d]={m:5.2f} ± {s:.2f}")
    print(f"   per-agent MAE={mae:.2f} plies | Spearman per-agent={rho:+.3f} | "
          f"group-mean monotonic Spearman={rho_g:+.3f}")
    print(f"E6 -> {p}")


def _ridge_cv(X, t, dims, seed=0, lams=(0.1, 1, 3, 10, 30, 100)):
    """Nested-CV ridge: outer 5-fold R^2, inner 5-fold to pick lambda. X standardised inside.
    `dims` selects feature columns. Closed-form ridge, numpy only."""
    rng = np.random.default_rng(seed)
    n = len(t); idx = rng.permutation(n); folds = np.array_split(idx, 5)
    Xd = X[:, dims]
    preds = np.full(n, np.nan)
    for k in range(5):
        te = folds[k]; trn = np.concatenate([folds[j] for j in range(5) if j != k])
        # inner CV to pick lambda
        best_l, best_e = lams[0], 1e9
        inf = np.array_split(rng.permutation(trn), 5)
        for lam in lams:
            err = 0
            for j in range(5):
                ite = inf[j]; itr = np.concatenate([inf[m] for m in range(5) if m != j])
                mu = Xd[itr].mean(0); sd = Xd[itr].std(0) + 1e-9
                A = (Xd[itr] - mu) / sd; b = t[itr] - t[itr].mean()
                w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
                p = ((Xd[ite] - mu) / sd) @ w + t[itr].mean()
                err += ((p - t[ite]) ** 2).sum()
            if err < best_e:
                best_e, best_l = err, lam
        mu = Xd[trn].mean(0); sd = Xd[trn].std(0) + 1e-9
        A = (Xd[trn] - mu) / sd; b = t[trn] - t[trn].mean()
        w = np.linalg.solve(A.T @ A + best_l * np.eye(A.shape[1]), A.T @ b)
        preds[te] = ((Xd[te] - mu) / sd) @ w + t[trn].mean()
    return 1 - ((preds - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def e5(cfg, min_dec=100):
    """Fig5: per-player 4-axis profile -> rating recovery (1-D depth vs full profile, nested CV)
    + elite case study. Depth axis uses an ELO-FREE model (elo zeroed) so recovering rating is
    not circular; trap/discovery axes are observable behaviour."""
    os.makedirs(FIGDIR, exist_ok=True)
    mc = cfg["model"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]
    raw_players = np.array(meta["player"]); elo = np.array(meta["elo"]); src = np.array(meta["source"])
    swl = np.array(meta["swing"])                     # 'up'/'down'
    tc = np.array(meta["time_class"])
    dpool = np.where(src == "broadcast", "otb", tc)   # rating pool (Lichess Glicko per TC; OTB=FIDE)

    # Broadcast PGN names are fragmented ("Carlsen, Magnus" / "Magnus Carlsen" / "Carlsen Magnus
    # (NOR)" / "LevonAronian"...). Canonicalise elite identities to one key: drop federation tags
    # and punctuation, then use the sorted set of name tokens of length>=2 (order-invariant, so
    # "Last, First" == "First Last"). Online players keep their unique Lichess username.
    import re
    def _canon(name, source):
        if source != "broadcast":
            return name
        s = re.sub(r"\([^)]*\)", " ", name.lower())   # strip (NOR), (USA), ...
        s = re.sub(r"[^a-z\s]", " ", s)               # drop underscores, commas, digits
        toks = sorted({t for t in s.split() if len(t) >= 2})
        return "elite::" + " ".join(toks) if toks else name
    players = np.array([_canon(p, s) for p, s in zip(raw_players, src)])
    n_merged = len(set(raw_players[src == "broadcast"])) - len(set(players[src == "broadcast"]))
    print(f"E5: canonicalised broadcast names "
          f"({len(set(raw_players[src=='broadcast']))} variants -> {len(set(players[src=='broadcast']))} identities, {n_merged} merged)")
    ctx2 = ctx.clone(); ctx2[:, 0] = 0.0              # zero ELO (keep clock -> time-elasticity works)
    tr, _ = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    print("E5: fitting elo-free depth model...")
    model = fit(mc, dev, delta, logq, mask, dmask, ctx2, y, tr)
    dh = dhat_over(model, dev, delta, logq, mask, dmask, ctx2, y, np.arange(len(y)))

    pr = delta[np.arange(len(y)), y].numpy()
    shallow_attr = 1.0 - pr[:, 0]                     # how good the played move looked shallow
    is_down = swl == "down"; is_up = swl == "up"
    tt = pd.read_parquet(cfg["data"]["selected"], columns=["pos_id", "time_spent"]).set_index("pos_id")
    tspent = tt.reindex(np.array(meta["pos_id"]))["time_spent"].to_numpy()
    ltime = np.log(np.clip(tspent, 1, None))

    rows = []
    for p in np.unique(players):
        m = players == p
        if m.sum() < min_dec:
            continue
        depth = dh[m].mean()
        trap = (is_down[m] * shallow_attr[m]).mean()          # falls for attractive traps
        disc = is_up[m].mean()                                # finds deep-value moves
        xm, ym = ltime[m], dh[m]
        ok = np.isfinite(xm) & np.isfinite(ym)
        telas = 0.0
        if ok.sum() > 20 and np.std(xm[ok]) > 1e-6:        # guard constant/degenerate think-time
            xc = xm[ok] - xm[ok].mean(); yc = ym[ok] - ym[ok].mean()
            telas = float((xc @ yc) / (xc @ xc))           # OLS slope, no SVD
        modal = pd.Series(dpool[m]).mode().iloc[0]
        rating = elo[m & (dpool == modal)].mean()     # single-pool (modal-control) rating
        rows.append((p, rating, modal, src[m][0] == "broadcast", depth, trap, disc, telas, m.sum()))
    P = pd.DataFrame(rows, columns=["player", "rating", "pool", "elite", "depth", "trap", "disc", "telas", "n"])
    # Lichess ratings are pool-specific: recover rating WITHIN pool (z-scored per pool).
    P["zrating"] = P.groupby("pool")["rating"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    P["name"] = [p.replace("elite::", "").title() if p.startswith("elite::") else p for p in P["player"]]
    P.to_csv("data/player_profiles.csv", index=False)
    print(f"E5: {len(P)} players with >={min_dec} decisions ({P.elite.sum()} elite) -> data/player_profiles.csv")
    z0 = {f: (P[f].mean(), P[f].std() + 1e-9) for f in ["depth", "trap", "disc", "telas"]}
    print("   selected young elites (z vs population):")
    for key in ["gukesh", "sindarov", "erigaisi", "praggnanandhaa", "firouzja", "keymer"]:
        hit = P[P.player.str.contains(key, case=False)]
        if len(hit):
            r = hit.sort_values("n", ascending=False).iloc[0]
            zz = "  ".join(f"{f}={(r[f]-z0[f][0])/z0[f][1]:+.2f}" for f in ["depth", "trap", "disc", "telas"])
            print(f"     {r['name'][:24].ljust(25)} rating={int(r['rating'])} n={int(r['n'])}  {zz}")

    feats = ["depth", "trap", "disc", "telas"]
    X = P[feats].to_numpy(); t = P["zrating"].to_numpy()     # within-pool z-scored rating target
    r2_1d = _ridge_cv(X, t, [0])                             # depth only
    r2_full = _ridge_cv(X, t, [0, 1, 2, 3])                  # full profile
    print(f"   within-pool rating recovery (nested-CV R^2): 1-D depth={r2_1d:.3f}  |  4-axis profile={r2_full:.3f}")

    # elite case study: top players by #decisions
    el = P[P.elite].sort_values("n", ascending=False).head(6).copy()
    z = {f: (P[f].mean(), P[f].std() + 1e-9) for f in feats}
    print("\n   elite profiles (z-scored vs population):")
    print("   " + "player".ljust(24) + "rating  " + "  ".join(f"{f:>6s}" for f in feats))
    for _, r in el.iterrows():
        zz = "  ".join(f"{(r[f]-z[f][0])/z[f][1]:>6.2f}" for f in feats)
        print(f"   {r['name'][:23].ljust(24)}{int(r['rating']):>5d}   {zz}")
    # 1-D (depth) ranking vs true-rating ranking among the elite (misranking demo)
    el_d = el.sort_values("depth", ascending=False)["player"].tolist()
    el_r = el.sort_values("rating", ascending=False)["player"].tolist()
    disc_pairs = sum(1 for i in range(len(el_r)) for j in range(i+1, len(el_r))
                     if el_d.index(el_r[i]) > el_d.index(el_r[j]))
    print(f"   1-D depth-score vs true-rating order: {disc_pairs} discordant pairs / {len(el_r)*(len(el_r)-1)//2}")

    # Fig5: z-scored radar-ish bar profiles for the elite
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.13
    for i, (_, r) in enumerate(el.iterrows()):
        ax.bar(np.arange(len(feats)) + i*w, [(r[f]-z[f][0])/z[f][1] for f in feats],
               w, label=f"{r['name'][:16]} ({int(r['rating'])})")
    ax.set_xticks(np.arange(len(feats)) + 2.5*w); ax.set_xticklabels(feats)
    ax.axhline(0, color="grey", lw=.7); ax.set_ylabel("z vs population")
    ax.set_title(f"E5 elite profiles (within-pool rating R$^2$: 1-D={r2_1d:.2f}, profile={r2_full:.2f})")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    p = f"{FIGDIR}/fig5_player_profiles.png"; figstyle.save(fig, p)
    print(f"E5 -> {p}")


def main(cfg, result):
    runners = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}
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
