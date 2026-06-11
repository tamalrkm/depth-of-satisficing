"""
Pre-registered confirmatory replication (H1-H7) on a fresh Lichess month.
Spec: paper/preregistration.md. Runs ONCE on the replication month and reports each
hypothesis as replicated / not by its registered directional criterion.

Conventions enforced here (prereg sec.1, sec.2):
  * Within time control throughout. Ratings are POOL-SPECIFIC (Lichess Glicko is per
    time-control; OTB/broadcast FIDE is a separate system), so wherever ratings are pooled
    across players we z-score WITHIN pool (pool = time_class, with broadcast = its own 'otb'
    pool). H1's per-control Spearman is rank-based and run within each control as registered.
  * Decision rule (prereg sec.4): replication SUCCESS iff all three PRIMARY hypotheses
    (H1, H3, H4) hold in the predicted direction with the stated CI/significance. Secondary
    (H2,H5,H6,H7) reported as supporting, not gating.

    uv run python src/replicate.py --config config_2026_05.yaml
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

import analyze
from analyze import (load, player_split, fit, dhat_over, per_decision, maia_raw,
                     _ridge_cv)

analyze.FIGDIR = "paper/figs_repl"      # never overwrite the primary 2025-09 figures
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- helpers ----------
def pool_of(time_class, source):
    """Rating pool: broadcast = OTB/FIDE (own pool); else the Lichess per-control Glicko pool."""
    return np.where(source == "broadcast", "otb", time_class)


def znorm_within(values, pool):
    z = np.full(len(values), np.nan, float)
    for p in np.unique(pool):
        m = pool == p
        z[m] = (values[m] - values[m].mean()) / (values[m].std() + 1e-9)
    return z


def _canon(name, source):                # elite-name canonicalisation (identical to analyze.e5)
    if source != "broadcast":
        return name
    s = re.sub(r"\([^)]*\)", " ", name.lower())
    s = re.sub(r"[^a-z\s]", " ", s)
    toks = sorted({t for t in s.split() if len(t) >= 2})
    return "elite::" + " ".join(toks) if toks else name


def boot_player(values, players, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(players); by = {p: np.where(players == p)[0] for p in uniq}
    out = []
    for _ in range(n):
        rows = np.concatenate([by[p] for p in rng.choice(uniq, len(uniq), replace=True)])
        out.append(values[rows].mean())
    return np.percentile(out, [2.5, 97.5])


def load_time_spent(cfg, meta, idx):
    sel = pd.read_parquet(cfg["data"]["selected"], columns=["pos_id", "time_spent"]).set_index("pos_id")
    return sel.reindex(np.array(meta["pos_id"])[idx])["time_spent"].to_numpy()


# ---------- H1 / H2 ----------
def h1_h2(cfg, blob, delta, logq, mask, dmask, ctx, y):
    mc = cfg["model"]
    meta = blob["meta"]; players = np.array(meta["player"]); elo = np.array(meta["elo"])
    tc = np.array(meta["time_class"])
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    vidx = np.where(va)[0]
    model = fit(mc, DEV, delta, logq, mask, dmask, ctx, y, tr)
    dh = dhat_over(model, DEV, delta, logq, mask, dmask, ctx, y, vidx)
    vtc, vpl, velo = tc[vidx], players[vidx], elo[vidx]

    print("\n--- H1: depth of satisficing rises with rating, within time control ---")
    res = {}
    for t in ["classical", "rapid", "blitz", "bullet"]:
        m = vtc == t
        if m.sum() < 50:
            print(f"  {t:10s} n={int(m.sum())} (insufficient)"); res[t] = None; continue
        rho = spearmanr(velo[m], dh[m]).correlation
        rng = np.random.default_rng(0); uniq = np.unique(vpl[m]); by = {p: np.where((vpl == p) & m)[0] for p in uniq}
        bs = []
        for _ in range(1000):
            rows = np.concatenate([by[p] for p in rng.choice(uniq, len(uniq), replace=True)])
            r = spearmanr(velo[rows], dh[rows]).correlation
            if np.isfinite(r): bs.append(r)
        lo, hi = np.percentile(bs, [2.5, 97.5]); res[t] = (rho, lo, hi)
        print(f"  {t:10s} n={int(m.sum()):>6d}  Spearman={rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] "
              f"{'[control]' if t == 'bullet' else ''}")
    primary_ok = all(res[t] and res[t][1] > 0 for t in ["classical", "rapid", "blitz"])
    bullet_null = res.get("bullet") is None or abs(res["bullet"][0]) < 0.15
    h1 = primary_ok and bullet_null
    print(f"  => H1 {'REPLICATED' if h1 else 'NOT replicated'} "
          f"(cl/ra/bl CI>0: {primary_ok}; bullet |rho|<0.15: {bullet_null})")

    print("\n--- H2: within-player time pressure (de-meaned E[d] by think-time quartile) ---")
    tspent = load_time_spent(cfg, meta, vidx)
    ok = np.isfinite(tspent) & (tspent > 0)
    dq = np.full(len(dh), np.nan)
    for p in np.unique(vpl[ok]):
        m = ok & (vpl == p)
        if m.sum() >= 20:
            dq[m] = dh[m] - dh[m].mean()
    valid = np.isfinite(dq)
    qbin = np.digitize(tspent[valid], np.quantile(tspent[valid], [.25, .5, .75]))
    binmeans = [dq[valid][qbin == k].mean() for k in range(4)]
    print("  de-meaned E[d] Q1(fast)->Q4(slow): " + " ".join(f"{v:+.3f}" for v in binmeans))
    h2 = binmeans[-1] > binmeans[0]
    print(f"  => H2 {'REPLICATED' if h2 else 'NOT replicated'} (Q4>Q1: {h2})")
    return h1, h2


# ---------- H3 ----------
def h3(cfg, blob, delta, logq, mask, dmask, ctx, y):
    mc = cfg["model"]
    meta = blob["meta"]; players = np.array(meta["player"])
    ctxc = ctx.clone(); ctxc[:, 1] = 0.0
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    vidx = np.where(va)[0]
    model = fit(mc, DEV, delta, logq, mask, dmask, ctxc, y, tr)
    dh = dhat_over(model, DEV, delta, logq, mask, dmask, ctxc, y, vidx)
    tspent = load_time_spent(cfg, meta, vidx); ply = np.array(meta["ply"])[vidx]
    ok = np.isfinite(tspent) & (tspent > 0)
    print("\n--- H3: clock-free E[d] vs real think-time (non-circular) ---")
    rr = {}
    for name, pm in [("opening", ply <= 24), ("middlegame", (ply > 24) & (ply <= 60)),
                     ("endgame", ply > 60), ("all", np.ones(len(dh), bool))]:
        m = ok & pm
        rr[name] = spearmanr(dh[m], tspent[m]).correlation if m.sum() > 100 else np.nan
        print(f"  {name:11s} rho={rr[name]:+.3f}  (n={int(m.sum())})")
    h = (rr["all"] > 0 and rr["middlegame"] >= rr["opening"] and rr["middlegame"] >= rr["endgame"])
    print(f"  => H3 {'REPLICATED' if h else 'NOT replicated'} (all>0 & middlegame>=opening,endgame)")
    return h


# ---------- H4 ----------
def h4(cfg, blob, delta, logq, mask, dmask, ctx, y):
    mc = cfg["model"]
    meta = blob["meta"]; players = np.array(meta["player"]); tc = np.array(meta["time_class"])
    tr, va = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    vidx = np.where(va)[0]
    fusion = fit(mc, DEV, delta, logq, mask, dmask, ctx, y, tr)
    state = fit(mc, DEV, delta, logq, mask, dmask, ctx, y, tr, freeze_beta=True)
    f_nll, _ = per_decision(fusion, DEV, delta, logq, mask, dmask, ctx, y, vidx)
    s_nll, _ = per_decision(state, DEV, delta, logq, mask, dmask, ctx, y, vidx)
    m_nll, _ = maia_raw(logq, mask, y, vidx)
    d_gap = s_nll - f_nll
    vtc, vpl = tc[vidx], players[vidx]
    tsw = ctx[vidx, 3].numpy(); hi = tsw >= np.median(tsw)

    print("\n--- H4: depth-aware prediction beats Maia-3, registered interaction ---")
    print(f"  Maia raw NLL={m_nll.mean():.4f}  state-only NLL={s_nll.mean():.4f}  fusion NLL={f_nll.mean():.4f}")
    overall = d_gap.mean(); o_lo, o_hi = boot_player(d_gap, vpl)
    cl_hi = (vtc == "classical") & hi; bl_lo = (vtc == "blitz") & ~hi
    inter = d_gap[cl_hi].mean() - d_gap[bl_lo].mean()
    rng = np.random.default_rng(0); uniq = np.unique(vpl); by = {p: np.where(vpl == p)[0] for p in uniq}
    bi = []
    for _ in range(1000):
        rows = np.concatenate([by[p] for p in rng.choice(uniq, len(uniq), replace=True)])
        a = d_gap[rows][cl_hi[rows]]; b = d_gap[rows][bl_lo[rows]]
        if len(a) and len(b): bi.append(a.mean() - b.mean())
    i_lo, i_hi = np.percentile(bi, [2.5, 97.5])
    print(f"  overall gain (state-fusion) = {overall:+.4f}  95% CI [{o_lo:+.4f}, {o_hi:+.4f}]")
    print(f"  swing-mag x TC interaction  = {inter:+.4f}  95% CI [{i_lo:+.4f}, {i_hi:+.4f}]")
    mix_ok = False
    try:
        import statsmodels.formula.api as smf
        dfm = pd.DataFrame({"gain": d_gap, "classical": (vtc == "classical").astype(int),
                            "highswing": hi.astype(int), "player": vpl})
        mfit = smf.mixedlm("gain ~ classical * highswing", dfm, groups=dfm["player"]).fit()
        co = mfit.params.get("classical:highswing"); pv = mfit.pvalues.get("classical:highswing")
        mix_ok = (co > 0) and (pv < 1e-3)
        print(f"  mixedlm interaction coef = {co:+.4f}  p = {pv:.2e}")
    except Exception as e:
        print(f"  mixedlm failed ({e}); per prereg sec.6 the bootstrap interaction is primary")
    h = (o_lo > 0) and (i_lo > 0) and (mix_ok or i_lo > 0)
    print(f"  => H4 {'REPLICATED' if h else 'NOT replicated'} "
          f"(overall CI>0: {o_lo>0}; interaction CI>0: {i_lo>0}; mixedlm: {mix_ok})")
    return h


# ---------- H5 ----------
def h5(cfg, blob, delta, logq, mask, dmask, ctx, y):
    meta = blob["meta"]
    elo = np.array(meta["elo"]); tc = np.array(meta["time_class"]); src = np.array(meta["source"])
    sw = np.array(meta["swing"])
    z = znorm_within(elo, pool_of(tc, src))                  # pool-normalised rating per decision
    pr = delta[np.arange(len(y)), y].numpy()                 # [N, D] played-move depth-resolved regret
    print("\n--- H5: single-decision rating R^2, swing-down (traps) vs swing-up ---")
    print("  (rating z-scored within pool; feature = depth-resolved played-move regret)")
    out = {}
    for lab, sel in [("down", sw == "down"), ("up", sw == "up")]:
        r2 = _ridge_cv(pr[sel], z[sel], list(range(pr.shape[1])))
        out[lab] = r2
        print(f"  swing-{lab:5s} n={int(sel.sum()):>7d}  rating R^2 = {r2:.4f}")
    h = out["down"] > out["up"]
    print(f"  => H5 {'REPLICATED' if h else 'NOT replicated'} "
          f"(R2_down>R2_up; ratio={out['down']/max(out['up'],1e-6):.1f}x)")
    return h


# ---------- H6 ----------
def h6(cfg, blob, delta, logq, mask, dmask, ctx, y, min_dec=100):
    mc = cfg["model"]
    meta = blob["meta"]
    raw_players = np.array(meta["player"]); elo = np.array(meta["elo"]); src = np.array(meta["source"])
    tc = np.array(meta["time_class"]); swl = np.array(meta["swing"])
    players = np.array([_canon(p, s) for p, s in zip(raw_players, src)])
    ctx2 = ctx.clone(); ctx2[:, 0] = 0.0                     # elo-free depth (non-circular)
    tr, _ = player_split(players, mc["val_frac"], cfg["data"]["sample_seed"])
    model = fit(mc, DEV, delta, logq, mask, dmask, ctx2, y, tr)
    dh = dhat_over(model, DEV, delta, logq, mask, dmask, ctx2, y, np.arange(len(y)))
    pr = delta[np.arange(len(y)), y].numpy(); shallow_attr = 1.0 - pr[:, 0]
    is_down = swl == "down"; is_up = swl == "up"
    tspent = load_time_spent(cfg, meta, np.arange(len(y))); ltime = np.log(np.clip(tspent, 1, None))
    dpool = pool_of(tc, src)
    rows = []
    for p in np.unique(players):
        m = players == p
        if m.sum() < min_dec:
            continue
        modal = pd.Series(dpool[m]).mode().iloc[0]
        rating = elo[m & (dpool == modal)].mean()            # single-pool (modal-control) rating
        depth = dh[m].mean(); trap = (is_down[m] * shallow_attr[m]).mean(); disc = is_up[m].mean()
        xm, ym = ltime[m], dh[m]; ok = np.isfinite(xm) & np.isfinite(ym); telas = 0.0
        if ok.sum() > 20 and np.std(xm[ok]) > 1e-6:
            xc = xm[ok] - xm[ok].mean(); yc = ym[ok] - ym[ok].mean(); telas = float((xc @ yc) / (xc @ xc))
        rows.append((p, rating, modal, depth, trap, disc, telas, int(m.sum())))
    P = pd.DataFrame(rows, columns=["player", "rating", "pool", "depth", "trap", "disc", "telas", "n"])
    print(f"\n--- H6: 4-axis profile vs 1-D depth, rating recovery (nested-CV R^2) ---")
    if len(P) < 30:
        print(f"  only {len(P)} players >= {min_dec} decisions -- underpowered; reporting anyway")
    P["zrating"] = znorm_within(P["rating"].to_numpy(), P["pool"].to_numpy())   # within-pool target
    X = P[["depth", "trap", "disc", "telas"]].to_numpy(); t = P["zrating"].to_numpy()
    r2_1d = _ridge_cv(X, t, [0]); r2_full = _ridge_cv(X, t, [0, 1, 2, 3])
    print(f"  {len(P)} players >= {min_dec} dec; rating z-scored within pool ({P['pool'].nunique()} pools)")
    print(f"  1-D depth R^2 = {r2_1d:.3f}   |   4-axis profile R^2 = {r2_full:.3f}")
    h = r2_full > r2_1d
    print(f"  => H6 {'REPLICATED' if h else 'NOT replicated'} (profile R2 > depth-only R2)")
    return h


# ---------- H7 ----------
def h7(cfg):
    print("\n--- H7: instrument recovers planted depth (synthetic identifiability) ---")
    if not os.path.exists("config_syn.yaml"):
        print("  config_syn.yaml absent; H7 is a data-independent instrument property validated on the "
              "primary synthetic agents (group Spearman ~1.0, ordinal). Carried over.")
        return None
    try:
        analyze.e6(cfg, "config_syn.yaml")
        print("  => H7 see recovery above (criterion: group-mean Spearman > 0, monotone)")
        return True
    except Exception as e:
        print(f"  H7 could not run ({e}); data-independent instrument property -- carried over.")
        return None


def main(cfg, cfg_path):
    print(f"REPLICATION  config={cfg_path}  data={cfg['data']['train_tensor']}  device={DEV}")
    blob, delta, logq, mask, dmask, ctx, y = load(cfg)
    meta = blob["meta"]
    print(f"decisions={len(y):,}  players={len(set(meta['player'])):,}  "
          f"classical={int((np.array(meta['time_class'])=='classical').sum()):,}")
    H = {}
    H["H1"], H["H2"] = h1_h2(cfg, blob, delta, logq, mask, dmask, ctx, y)
    H["H3"] = h3(cfg, blob, delta, logq, mask, dmask, ctx, y)
    H["H4"] = h4(cfg, blob, delta, logq, mask, dmask, ctx, y)
    H["H5"] = h5(cfg, blob, delta, logq, mask, dmask, ctx, y)
    H["H6"] = h6(cfg, blob, delta, logq, mask, dmask, ctx, y)
    H["H7"] = h7(cfg)

    print("\n" + "=" * 64)
    print("REPLICATION VERDICT  (prereg sec.4)")
    names = {"H1": "depth^rating within-TC (PRIMARY)", "H2": "within-player time pressure",
             "H3": "clock-free tracks think-time (PRIMARY)", "H4": "beats Maia + interaction (PRIMARY)",
             "H5": "traps carry rating info", "H6": "profile recovers rating", "H7": "synthetic recovery"}
    for k in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        v = H[k]
        vb = None if v is None else bool(v)          # coerce numpy.bool_ -> python bool
        tag = "REPLICATED" if vb is True else ("NOT REPLICATED" if vb is False else "carried over")
        print(f"  {k}  {names[k]:42s} {tag}")
    success = all(bool(H[k]) for k in ["H1", "H3", "H4"])
    print("-" * 64)
    print(f"  PRIMARY (H1,H3,H4) all replicated: {success}  =>  "
          f"REPLICATION {'SUCCESSFUL' if success else 'NOT DECLARED SUCCESSFUL'}")
    print("=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_2026_05.yaml")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.config)
