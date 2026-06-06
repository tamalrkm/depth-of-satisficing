"""
Stage 5: join stages 1-4 into padded model tensors -> data/train.pt.

For each selected decision we assemble, over the candidate move set (union of the engine's
top-K across all depths; the played move is always present):
    delta     [M, D]  regret = best_winprob_d - winprob_{m,d} on cfg.model.depth_grid, >=0
    logq      [M]     log Maia-3 policy (floored for moves Maia gives ~0)
    move_mask [M]     1 real move / 0 pad
    context   [C]     [elo_norm, clock_norm, ply_norm, total_swing, time_class one-hot(4)]
    y                 index of the played move within the M candidates
plus per-decision meta (player, elo, time_class, swing-class up/down, ply) for splitting
and the Results analyses. Per-move swing is sw(m) = sum_d (delta_{m,d} - delta_{m,D})
(the 2015 definition); position total-swing = sum_m |sw(m)|.

Run:
    python src/build_dataset.py --config config.yaml
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch
import yaml

TIME_CLASSES = ["bullet", "blitz", "rapid", "classical"]
Q_FLOOR = 1e-6


def val_at(ds, target):
    """Value of trajectory `ds` ({depth: val}) at `target`: the nearest shallower reported
    depth, back-filling the earliest if target precedes all reported depths."""
    depths = sorted(ds)
    le = [d for d in depths if d <= target]
    return ds[le[-1]] if le else ds[depths[0]]


def fill_on_grid(ds, grid):
    """ds: {engine_depth: winprob}. Return winprob at each grid depth, forward-filling from
    the nearest shallower reported depth (back-filling the very early grid points)."""
    if not ds:
        return [0.0] * len(grid)
    return [val_at(ds, g) for g in grid]


def main(cfg):
    d, mc = cfg["data"], cfg["model"]
    grid = mc["depth_grid"]
    D = len(grid)
    target = cfg["engine"]["depth"]   # positions reaching < this were node-capped (see run_engine)

    sel = pd.read_parquet(d["selected"]).set_index("pos_id")
    traj = pd.read_parquet(d["depth_traj"])
    maia = pd.read_parquet(d["maia_q"])

    # per-position move trajectories and Maia policy
    traj_by_pos = {}
    for pos_id, g in traj.groupby("pos_id"):
        moves = {}
        for mv, gm in g.groupby("move"):
            moves[mv] = dict(zip(gm["depth"].tolist(), gm["winprob"].tolist()))
        traj_by_pos[pos_id] = moves
    q_by_pos = {pos_id: dict(zip(g["move"], g["q"]))
                for pos_id, g in maia.groupby("pos_id")}

    # best win-prob per raw depth for each analysed position (side-to-move POV). Used to
    # recover a played move's value from the position it leads to, avoiding any re-query.
    best_raw_by_pos = {}
    for pid, mvs in traj_by_pos.items():
        bd = {}
        for ds in mvs.values():
            for dd, ww in ds.items():
                if ww > bd.get(dd, -1.0):
                    bd[dd] = ww
        best_raw_by_pos[pid] = bd

    samples = []  # (delta[M,D], logq[M], played_idx, context[C], meta)
    max_M = 0
    skipped = 0
    for pos_id, moves in traj_by_pos.items():
        if pos_id not in sel.index or pos_id not in q_by_pos:
            skipped += 1
            continue
        meta = sel.loc[pos_id]
        played = meta["played_uci"]
        real_cand = sorted(moves.keys())

        # reached depth of this position's engine search (same for all its moves -- one multipv
        # search). If the node cap stopped it short of `target`, mask grid depths beyond what
        # was actually searched so the model marginalises over observed depths only.
        reached = max(best_raw_by_pos.get(pos_id, {}), default=0)
        capped = reached < target
        dmask = np.array([0.0 if (capped and g > reached) else 1.0 for g in grid],
                         dtype=np.float32)

        # winprob on grid per candidate -> regret vs per-depth best
        wp = np.array([fill_on_grid(moves[m], grid) for m in real_cand])   # [R, D]
        if played in real_cand:
            cand = real_cand
        else:
            # Played move fell outside the engine's top-K at every depth (we don't re-query).
            # Recover its value from the position it leads to: value(m, depth d) =
            # 1 - best_winprob(next position, depth d-1) -- POV flips, and one ply shallower
            # so m sits on the same footing as its siblings (negamax). The successor was
            # analysed too (consecutive plies of a sampled game), so this is free.
            nb = best_raw_by_pos.get(f"{str(meta['game_id'])}:{int(meta['ply']) + 1}")
            if nb:
                m_wp = np.array([1.0 - val_at(nb, max(g - 1, 1)) for g in grid])
            else:
                # no analysed successor (game end / max_ply / --limit edge): "as bad as worst kept"
                m_wp = wp.min(axis=0)
            wp = np.vstack([wp, m_wp])
            cand = real_cand + [played]
        M = len(cand)
        best = wp.max(axis=0, keepdims=True)                          # [1, D]
        delta = np.clip(best - wp, 0.0, None)                         # [M, D]

        # per-move swing sw(m) = sum_d (delta_{m,d} - delta_{m,D}); total swing of position.
        # ICMLA-2015 convention: a move SWINGS UP if its delta DECREASES with depth (potential
        # shows only deep) => sw>0; SWINGS DOWN if delta rises with depth (a trap) => sw<0.
        sw = (delta - delta[:, -1:]).sum(axis=1)                      # [M]
        total_swing = float(np.abs(sw).sum())
        played_idx = cand.index(played)
        swing_class = "up" if sw[played_idx] > 0 else "down"

        qd = q_by_pos[pos_id]
        logq = np.array([math.log(max(qd.get(m, Q_FLOOR), Q_FLOOR)) for m in cand])

        elo = float(meta["side_to_move_elo"])
        clock = meta["clock_before"]
        clock = 0.0 if pd.isna(clock) else float(clock)
        ply = float(meta["ply"])
        tc = meta["time_class"]
        tc_onehot = [1.0 if tc == c else 0.0 for c in TIME_CLASSES]
        context = np.array([
            (elo - 1500.0) / 700.0,
            min(clock, 1200.0) / 600.0,
            ply / float(cfg["data"]["max_ply"]),
            total_swing,
            *tc_onehot,
        ], dtype=np.float32)

        samples.append((delta.astype(np.float32), logq.astype(np.float32),
                        played_idx, context, dmask,
                        dict(player=str(meta["player"]), elo=float(elo), time_class=str(tc),
                             swing=swing_class, ply=int(ply), pos_id=str(pos_id),
                             source=str(meta.get("source", "online")),
                             reached_depth=int(reached), capped=bool(capped))))
        max_M = max(max_M, M)

    if not samples:
        raise SystemExit("no joined samples -- check stage outputs (depth_traj / maia_q / selected)")

    N, C = len(samples), samples[0][3].shape[0]
    delta_t = torch.zeros(N, max_M, D)
    logq_t = torch.zeros(N, max_M)
    mask_t = torch.zeros(N, max_M)
    dmask_t = torch.ones(N, D)            # depth-validity mask (1 = observed, 0 = past node cap)
    ctx_t = torch.zeros(N, C)
    y_t = torch.zeros(N, dtype=torch.long)
    meta = {k: [] for k in ("player", "elo", "time_class", "swing", "ply", "pos_id", "source",
                            "reached_depth", "capped")}

    for n, (delta, logq, yi, ctx, dmask, m) in enumerate(samples):
        M = delta.shape[0]
        delta_t[n, :M] = torch.from_numpy(delta)
        logq_t[n, :M] = torch.from_numpy(logq)
        mask_t[n, :M] = 1.0
        dmask_t[n] = torch.from_numpy(dmask)
        ctx_t[n] = torch.from_numpy(ctx)
        y_t[n] = yi
        for k in meta:
            meta[k].append(m[k])

    blob = dict(delta=delta_t, logq=logq_t, move_mask=mask_t, depth_mask=dmask_t,
                context=ctx_t, y=y_t, meta=meta)
    torch.save(blob, d["train_tensor"])
    n_capped = sum(meta["capped"])
    print(f"wrote {N} decisions (M<= {max_M}, D={D}, C={C}), skipped {skipped} "
          f"-> {d['train_tensor']}")
    print(f"  players={len(set(meta['player']))}  "
          f"swing up/down = {meta['swing'].count('up')}/{meta['swing'].count('down')}  "
          f"node-capped positions = {n_capped} ({100*n_capped/N:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
