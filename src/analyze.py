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
    torch.manual_seed(0)          # reproducible init + batch order: figures match reported text
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
    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    tcs = ["classical", "rapid", "blitz", "bullet"]
    himeans = [d_gap[(tc == t) & hi].mean() for t in tcs]
    lomeans = [d_gap[(tc == t) & ~hi].mean() for t in tcs]
    xx = np.arange(len(tcs))
    bh = ax.bar(xx - 0.19, himeans, 0.34, label="high-swing positions",
                color=figstyle.ACCENT, edgecolor="white", linewidth=1.2, zorder=3)
    bl = ax.bar(xx + 0.19, lomeans, 0.34, label="low-swing positions",
                color=figstyle.ACCENT_LIGHT, edgecolor="white", linewidth=1.2, zorder=3)
    figstyle.zero_line(ax)
    for bars in (bh, bl):                      # value at every tip: only 8 bars, all read
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=7, color=figstyle.INK2)
    ax.set_xticks(xx); ax.set_xticklabels(tcs)
    ax.set_ylabel("held-out gain over state-only model\n(Δ NLL, nats)")
    ax.margins(y=0.18)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", handlelength=1.2, handleheight=1.0)
    p = f"{FIGDIR}/fig4_prediction_gain.png"; figstyle.save(fig, p)
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


# Worked example for Fig1a. Chosen deterministically from the primary month: among classical
# decisions (Elo 1800-2200, plies 20-60, 4-6 candidates), the position maximising
# trap_drop(played) + discovery_rise(best alt) + shallow_seduction gap (all >= thresholds
# 0.10/0.10/0.03); see the selection sweep in the repo history. Winner: a 1910-rated player
# recaptures a bishop with fxe6 -- excellent at shallow depth, refuted (Qxe6+) deeper --
# while the quiet Kb8 reveals its worth only at depth.
EXAMPLE_POS = "dvqfQ20I:39"
EXAMPLE_LABELS = {"f7e6": ("played:  fxe6  (trap)", "#d03b3b"),
                  "c8b8": ("best:  Kb8  (deep discovery)", "#184f95")}


def _worked_example_panel(cfg, axi):
    """Fig1a: one real decision's candidate win-prob trajectories across engine depth."""
    dt = pd.read_parquet(cfg["data"]["depth_traj"])
    ex = dt[dt.pos_id == EXAMPLE_POS]
    for mv, grp in ex.groupby("move"):
        grp = grp.sort_values("depth")
        if mv in EXAMPLE_LABELS:
            lab, col = EXAMPLE_LABELS[mv]
            axi.plot(grp.depth, grp.winprob, color=col, lw=2.3, zorder=4)
            end = grp.iloc[-1]
            figstyle.direct_label(axi, end.depth, end.winprob, lab, col, dx=5,
                                  fontsize=7.5, weight="bold")
        else:
            axi.plot(grp.depth, grp.winprob, color=figstyle.MUTED, lw=1.0, alpha=0.45, zorder=2)
    axi.set_xlabel("engine search depth (plies)")
    axi.set_ylabel("win probability, player to move")
    axi.set_title("One decision at increasing depth")
    axi.set_xticks([1, 6, 11, 16, 21])
    axi.set_xlim(right=27.5)                 # room for the direct labels
    return ex.pos_id.nunique()


def e1(cfg):
    """Fig1: (a) worked example; (b-d) played-move regret across depth, by rating band and
    swing class. b-d straight from train.pt (delta[n, y[n], :] is the played move's regret)."""
    os.makedirs(FIGDIR, exist_ok=True)
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]
    grid = cfg["model"]["depth_grid"]
    elo = np.array(meta["elo"]); sw = np.array(meta["swing"])
    tc = np.array(meta["time_class"])
    slow = np.isin(tc, ["classical", "rapid"])      # depth signal lives in slow play (within-TC rigor)
    pr = delta[np.arange(len(y)), y].numpy()        # [N, D] played-move regret on the grid
    bands = np.array([band_of(e) for e in elo])
    band_names = [f"{lo}–{hi}" for lo, hi in BANDS[:-1]] + ["2600+"]
    fig, ax = plt.subplots(1, 4, figsize=(13.6, 3.6), constrained_layout=True)
    _worked_example_panel(cfg, ax[0])
    figstyle.panel_label(ax[0], "a", dx=-0.20)
    panels = [("All decisions", np.ones(len(y), bool)),
              ("Swing-up (deep discovery)", sw == "up"),
              ("Swing-down (traps)", sw == "down")]
    print(f"E1 (slow controls, classical+rapid; n={int(slow.sum()):,}):")
    for pi, ((title, msw), axi, lab) in enumerate(zip(panels, ax[1:], "bcd")):
        m = msw & slow
        for bi in range(len(BANDS)):
            sel = (bands == bi) & m
            if sel.sum() > 50:
                extreme = bi in (0, len(BANDS) - 1)
                curve = pr[sel].mean(0)
                axi.plot(grid, curve, color=figstyle.BAND_RAMP[bi],
                         lw=2.3 if extreme else 1.7, label=band_names[bi], zorder=3)
                if pi == 1 and extreme:      # direct-label the ramp extremes once, panel c
                    figstyle.direct_label(axi, grid[-1], curve[-1], band_names[bi],
                                          figstyle.BAND_RAMP[max(bi, 3)], dx=5,
                                          weight="bold")
        axi.set_title(title)
        axi.set_xlabel("engine search depth (plies)")
        axi.set_xticks([2, 6, 10, 14, 18, 22])
        figstyle.panel_label(axi, lab, dx=-0.14 if pi == 0 else -0.06)
        bm = np.array([pr[(bands == bi) & m, -1].mean() if ((bands == bi) & m).sum() > 50 else np.nan
                       for bi in range(len(BANDS))])
        print(f"  {title:28s} deep-regret by band: " + " ".join(f"{v:.3f}" for v in bm if not np.isnan(v))
              + f"   (spread={np.nanmax(bm)-np.nanmin(bm):.3f})")
    # regret panels share one scale (worked-example panel a has its own, win-prob)
    ylo = min(a.get_ylim()[0] for a in ax[1:]); yhi = max(a.get_ylim()[1] for a in ax[1:])
    for a in ax[1:]:
        a.set_ylim(ylo, yhi)
    for a in ax[2:]:
        a.tick_params(labelleft=False)
    ax[2].set_xlim(right=grid[-1] + 3.6)     # room for the direct labels
    ax[1].set_ylabel("mean regret of played move\n(win probability)")
    leg = ax[3].legend(title="rating band", ncol=2, loc="upper left",
                       handlelength=1.4, columnspacing=1.0, labelspacing=0.3)
    leg.get_title().set_fontsize(8)
    p = f"{FIGDIR}/fig1_error_vs_depth.png"; figstyle.save(fig, p)
    print(f"E1 -> {p}")


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
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.35, 1]})
    rng = np.random.default_rng(0)
    print(f"E2a depth vs rating, WITHIN time control (pooled rho={spearmanr(elo[vidx], dh).correlation:+.3f}, confounded):")
    ends = []                                     # (end_x, end_y, label, colour) for direct labels
    for t in figstyle.TC_ORDER:
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
        means, los, his = np.array(means), np.array(los), np.array(his)
        rho = spearmanr(elo[vidx][tsel], dh[tsel]).correlation
        ok = ~np.isnan(means)
        col = figstyle.TC_COLORS[t]
        xs = np.array(BAND_MID)[ok]
        ax[0].fill_between(xs, los[ok], his[ok], color=col, alpha=0.18, lw=0, zorder=2)
        ax[0].plot(xs, means[ok], marker="o", color=col, zorder=3,
                   **figstyle.ring())
        ends.append((xs[-1], means[ok][-1],
                     t + (" (control)" if t == "bullet" else ""), col))
        print(f"   {t:10s} n={tsel.sum():>6d}  Spearman={rho:+.3f}  "
              f"per-band E[d]: " + " ".join(f"{m:.2f}" for m in means if not np.isnan(m)))
    # direct labels at line ends, staggered so converging series stay attached & legible
    ends.sort(key=lambda e: e[1])
    last_y = -1e9
    for ex, ey, lab, col in ends:
        ly = max(ey, last_y + 0.32)               # >= 0.32 plies apart on the E[d] axis
        figstyle.direct_label(ax[0], ex, ly, lab,
                              col if lab.startswith(("classical", "rapid")) else figstyle.INK2,
                              dx=6, weight="bold" if lab.startswith(("classical", "rapid")) else "regular")
        last_y = ly
    ax[0].set_xlim(right=BAND_MID[-1] + 60)
    ax[0].set_xlabel("rating (per-control bands)")
    ax[0].set_ylabel("depth of satisficing, E[d] (plies)")
    ax[0].set_title("Depth of satisficing by rating band")

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
    binmeans = np.array([dq[valid & (q == k)].mean() for k in range(4)])
    # player-clustered bootstrap CI around the same pooled quartile means
    vv = np.where(valid)[0]
    pl_v = vpl[vv]; qq_v = q[vv]; dq_v = dq[vv]
    plv = np.unique(pl_v)
    rows_of = {p: np.where(pl_v == p)[0] for p in plv}
    boots = np.full((1000, 4), np.nan)
    for b in range(1000):
        take = np.concatenate([rows_of[p] for p in rng.choice(plv, len(plv), replace=True)])
        for k in range(4):
            mk = qq_v[take] == k
            if mk.sum():
                boots[b, k] = dq_v[take][mk].mean()
    lo, hi_ = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    xx = np.arange(4)
    ax[1].errorbar(xx, binmeans, yerr=[binmeans - lo, hi_ - binmeans],
                   fmt="o", color=figstyle.ACCENT, ecolor=figstyle.ACCENT,
                   elinewidth=1.6, capsize=0, markersize=7, zorder=3,
                   **figstyle.ring())
    ax[1].plot(xx, binmeans, color=figstyle.ACCENT, lw=1.4, alpha=0.55, zorder=2)
    figstyle.zero_line(ax[1])
    ax[1].set_xticks(xx); ax[1].set_xticklabels(["Q1\n(fastest)", "Q2", "Q3", "Q4\n(slowest)"])
    ax[1].set_xlabel("think-time quartile")
    ax[1].set_ylabel("Δ E[d] within player (plies)")
    ax[1].set_title("Within-player effect of think time")
    figstyle.panel_label(ax[0], "a"); figstyle.panel_label(ax[1], "b")
    p = f"{FIGDIR}/fig2_depth_rating_time.png"; figstyle.save(fig, p)
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
    print("E3 Spearman(E[d] [clock-free], think-time), held out:")
    rhos = {}
    for name, pmask in phases:
        m = ok & pmask
        if m.sum() > 100:
            rhos[name] = spearmanr(dh[m], tspent[m]).correlation
            print(f"   {name:11s} rho={rhos[name]:+.3f}  (n={m.sum()})")
    fig, ax = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True)
    names = [n for n in ("opening", "middlegame", "endgame") if n in rhos]
    vals = [rhos[n] for n in names]
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, vals, height=0.46, color=figstyle.ACCENT,
            edgecolor="white", linewidth=1.0, zorder=3)
    for yv, v in zip(yy, vals):                       # value inside each bar tip (white on blue)
        ax.annotate(f"+{v:.2f}", (v, yv), xytext=(-5, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color="white", zorder=4)
    if "all" in rhos:                                  # all-phases reference
        ax.axvline(rhos["all"], color=figstyle.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"all phases +{rhos['all']:.2f}", (rhos["all"], len(names) - 0.42),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=7.5, color=figstyle.MUTED, annotation_clip=False)
    ax.set_yticks(yy); ax.set_yticklabels(names)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("Spearman ρ  (clock-free E[d]  vs  observed think time)")
    ax.grid(axis="y", visible=False)
    p = f"{FIGDIR}/fig3_depth_vs_time.png"; figstyle.save(fig, p)
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

    fig, ax = plt.subplots(figsize=(4.4, 4.1), constrained_layout=True)
    lim = [min(depths) - 2, max(depths) + 2]
    ax.plot(lim, lim, ls=(0, (4, 3)), lw=1.2, color=figstyle.MUTED, label="identity", zorder=2)
    ax.scatter(rec_pl + np.random.default_rng(0).normal(0, .15, len(rec_pl)), rec,
               s=11, alpha=.30, color=figstyle.ACCENT_LIGHT, lw=0, zorder=3,
               label="individual agents")
    ax.errorbar(depths, gmean, yerr=gsd, marker="o", color=figstyle.ACCENT_DARK,
                elinewidth=1.6, capsize=0, markersize=7, lw=2.0, zorder=4,
                label="mean ± s.d. per planted depth", **figstyle.ring())
    ax.set_xlabel("planted depth, $d_\\mathrm{plant}$ (plies)")
    ax.set_ylabel("recovered depth, E[d] (plies)")
    ax.set_xticks(depths)
    ax.set_yticks(range(4, 22, 4))
    ax.legend(loc="upper left", handlelength=1.4)
    p = f"{FIGDIR}/fig6_synthetic_recovery.png"; figstyle.save(fig, p)
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

    # Fig5: dot-plot small multiples -- one panel per profile axis, elite players on a shared
    # rating-ordered y axis. Identity comes from the row label, so no colour legend is needed.
    def surname(name):
        toks = name.split()
        return max(toks, key=len).title() if toks else name
    el = el.sort_values("rating", ascending=True).reset_index(drop=True)   # top player on top row
    ylabels = [f"{surname(r['name'])}  ({int(r['rating'])})" for _, r in el.iterrows()]
    axis_titles = [("depth", "depth of\nsatisficing"), ("trap", "trap\nsusceptibility"),
                   ("disc", "deep-discovery\nrate"), ("telas", "time\nelasticity")]
    fig, axes = plt.subplots(1, 4, figsize=(9.8, 2.9), sharey=True, constrained_layout=True)
    yy = np.arange(len(el))
    for axi, (f, ttl) in zip(axes, axis_titles):
        zs = np.array([(r[f] - z[f][0]) / z[f][1] for _, r in el.iterrows()])
        figstyle.zero_line(axi, axis="x")
        axi.hlines(yy, 0, zs, color=figstyle.ACCENT_LIGHT, lw=1.4, zorder=2)
        axi.plot(zs, yy, "o", color=figstyle.ACCENT, markersize=7.5, zorder=3,
                 **figstyle.ring())
        axi.set_title(ttl, fontsize=8.5)
        axi.set_xlabel("z vs population", fontsize=7.5)
        lim = max(1.0, np.abs(zs).max() * 1.25)
        axi.set_xlim(-lim, lim)
        axi.grid(axis="y", visible=False)
    axes[0].set_yticks(yy); axes[0].set_yticklabels(ylabels, fontsize=8)
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
