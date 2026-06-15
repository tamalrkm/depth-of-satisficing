# Data manifest — derived datasets for OSF deposit

Raw sources are public (Lichess open DB / broadcasts; chess.com Published-Data API). The files
below are the **derived** datasets produced by the pipeline (see `README.md`). They are
git-ignored; deposit them on OSF and put the DOI in `paper/manuscript.tex` (Data availability) and
`README.md`.

## Datasets (one folder per pool/month)

| dataset | folder | decisions | source mix |
|---|---|--:|---|
| **Primary** (Lichess 2025-09) | `data/` | 922,412 | online + elite broadcast/OTB |
| **Replication-1** (Lichess 2026-05, OSF KB4ZQ) | `data/repl/` | 179,360 | online + broadcast |
| **Replication-2 / D2** (Lichess 2026-04, OSF A6NYK) | `data/repl04/` | 168,654 | online + broadcast |
| **Independent pool** (chess.com) | `data/chesscom/` | 99,358 | titled-player games (bullet/blitz/rapid) |
| **Synthetic** (identifiability) | per `config_syn.yaml` | — | fixed-depth agents |

## Files per dataset & sizes

| file | ~size (primary / repl / repl04 / chesscom) | what |
|---|---|---|
| `positions.parquet` | 48 / 9 / 9 / 3 MB | one row per decision: FEN, played move, clocks, ratings, time class |
| `selected.parquet`  | 48 / 9 / 9 / 3 MB | decisions kept for analysis (pass-through; = positions) |
| `depth_traj.parquet` | 320 / 62 / 58 / 35 MB | **engine output** (the expensive stage): per (position, candidate move, depth) win-probability |
| `maia_q.parquet`    | 216 / 42 / 39 / 23 MB | Maia-3 policy per (position, legal move) |
| `train.pt`          | 1851 / 342 / 330 / 190 MB | model-ready tensors (see schema) — lets others reproduce **all** results without the engine |
| `model.pt`          | 0.1 MB (primary) | trained latent-depth model state |

(Transient/optional: `data/positions_broadcast.parquet` is the hardcoded broadcast staging file —
last written by the most recent run; regenerable. `data/chesscom/positions_scope.parquet` is the
12-GM scoping pull — omit.)

## Schemas

**positions / selected** (parquet): `pos_id (=game_id:ply), game_id, ply, fen, side_to_move_elo,
oppo_elo, time_class, base_time, increment, clock_before, clock_after, time_spent,
lichess_eval (NaN for chess.com/broadcast), played_uci, player, source`.

**depth_traj** (parquet): `pos_id, move (uci), depth, winprob` — win-prob = (W+0.5·D)/1000 from
Stockfish `UCI_ShowWDL`, side-to-move POV; depth grid the engine reached (≤21; node cap 2×10⁸).

**maia_q** (parquet): `pos_id, move (uci), q` — Maia-3 (`maia3-79m`) policy, side-to-move-Elo-conditioned.

**train.pt** (torch dict): `delta [N,M,D]` regret in win-prob units on grid {2,…,22}; `logq [N,M]`
log-Maia; `move_mask [N,M]`; `depth_mask [N,D]` (node-cap validity); `context [N,8]`
(elo, clock, ply, total-swing, time-class one-hot); `y [N]` played-move index; `meta` dict
(player, elo, time_class, swing up/down, ply, pos_id, source, reached_depth, capped).

## Recommended deposit (OSF)
Minimum for full reproducibility **without re-running the 11 h engine**: `depth_traj.parquet`
(the expensive engine output) + `selected.parquet` + `maia_q.parquet` for each dataset, plus
`train.pt` (most convenient — analyses run directly off it). Suggested OSF structure: one
component/folder per dataset (`primary-2025-09/`, `replication-2026-05/`, `replication-2026-04/`,
`chesscom/`, `synthetic/`). Largest single file (`train.pt`, 1.85 GB) is under OSF's 5 GB/file
limit; if the project total is inconvenient, host the four `train.pt` on Zenodo and link from OSF.
License: **CC-BY** (matching the pre-registration).
