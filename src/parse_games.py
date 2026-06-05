"""
Stage 1: stream PGN -> sampled positions with clocks + lichess eval.

Reads a PGN source (plain .pgn, .pgn.zst stream-decompressed with `zstandard`, or stdin),
balanced-reservoir-samples games to hit cfg targets per (elo_bin x time_class), and emits
one row per candidate decision (the side-to-move's move at that ply).

Output schema (data/positions.parquet):
    pos_id, game_id, ply, fen, side_to_move_elo, oppo_elo, time_class,
    base_time, increment, clock_before, clock_after, time_spent,
    lichess_eval (cp, +ve = side-to-move better; NaN if absent), played_uci, player,
    hist_uci   (space-joined UCI from startpos to this ply, for faithful Maia-3 history)

Parsing notes:
  - per-move clock from the '%clk' comment; time_spent = clk_before - clk_after + increment
  - lichess '%eval' comment gives a shallow eval used by the SELECTION stage pre-filter;
    stored side-to-move POV in centipawns (mate -> +/-10000)
  - time_class from the TimeControl header (base + 40*inc): bullet<180, blitz<480,
    rapid<1500, else classical
  - skip plies outside [min_ply, max_ply] (opening book churn / very long games)

Run:
    python src/parse_games.py --config config.yaml --pgn data/raw/smoke.pgn
    zstdcat dump.pgn.zst | python src/parse_games.py --config config.yaml --stdin
"""
import argparse
import glob
import io
import math
import os
import re
import sys
from itertools import groupby
from multiprocessing import Pool

import chess
import chess.pgn
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
EVAL_RE = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")


def time_class(base, inc):
    """Lichess time-class buckets from estimated game duration (base + 40*inc seconds)."""
    est = base + 40 * inc
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"


def parse_time_control(tc):
    if not tc or tc in ("-", "?"):
        return 0.0, 0.0
    base, _, inc = tc.partition("+")
    try:
        return float(base), float(inc or 0)
    except ValueError:
        return 0.0, 0.0


def elo_bin(elo, bins):
    """Map a rating to the lower edge of its 200-point band; None if below the lowest bin."""
    b = None
    for edge in bins:
        if elo >= edge:
            b = edge
    return b


def parse_clk(comment):
    m = CLK_RE.search(comment)
    if not m:
        return None
    h, mm, ss = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(ss)


def parse_eval(comment):
    """Return centipawns (White POV). Mate -> +/-10000. None if no %eval."""
    m = EVAL_RE.search(comment)
    if not m:
        return None
    tok = m.group(1)
    if tok.startswith("#"):
        sign = -1 if tok[1:].startswith("-") else 1
        return sign * 10000.0
    return float(tok) * 100.0


class _LimitedRaw:
    """Wrap a binary file so it yields at most `limit` bytes -- lets us decompress only the
    front of a still-downloading .zst (parse a prefix before the full file lands)."""

    def __init__(self, fh, limit):
        self.fh = fh
        self.rem = limit

    def read(self, n=-1):
        if self.rem <= 0:
            return b""
        if n is None or n < 0:
            n = self.rem
        data = self.fh.read(min(n, self.rem))
        self.rem -= len(data)
        return data

    def close(self):
        self.fh.close()


def open_pgn(args, cfg):
    """Return a text stream of PGN from --stdin, --pgn (optionally .zst), or the cfg month file.
    With --max-bytes, only the first N bytes of a .zst are decompressed (a parseable prefix)."""
    if args.stdin:
        return sys.stdin
    path = args.pgn
    if path is None:
        month = cfg["data"]["lichess_month"]
        path = f"{cfg['data']['raw_dir']}/lichess_db_standard_rated_{month}.pgn.zst"
    if path.endswith(".zst"):
        import zstandard
        fh = open(path, "rb")
        raw = _LimitedRaw(fh, args.max_bytes) if getattr(args, "max_bytes", 0) else fh
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def iter_games(stream):
    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception as exc:
            # a truncated prefix ends mid-frame: stop cleanly after the last complete game
            print(f"(stopped at stream end/truncation: {type(exc).__name__})", file=sys.stderr)
            return
        if game is None:
            return
        yield game


_PARQUET_TAGS = ("Event", "Site", "White", "Black", "Result", "WhiteTitle", "BlackTitle",
                 "WhiteElo", "BlackElo", "TimeControl", "UTCDate", "UTCTime")


def iter_games_parquet(paths):
    """Yield chess.pgn games from the official Lichess HF parquet (header columns + a
    `movetext` column whose inline {[%clk]/[%eval]} comments we reattach via read_game)."""
    import pyarrow.parquet as pq

    for path in paths:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        tags = [t for t in _PARQUET_TAGS if t in names]
        for batch in pf.iter_batches(batch_size=4000):
            mt = batch.column("movetext")
            cols = {t: batch.column(t) for t in tags}
            for i in range(len(mt)):
                movetext = mt[i].as_py()
                if not movetext:
                    continue
                game = chess.pgn.read_game(io.StringIO(movetext))
                if game is None:
                    continue
                for t in tags:
                    v = cols[t][i].as_py()
                    if v is not None and v != "":
                        game.headers[t] = str(v)
                yield game


def game_rows(game, cfg):
    """Yield position dicts for one game, or nothing if it fails header filters."""
    h = game.headers
    try:
        welo, belo = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
    except ValueError:
        return
    if welo <= 0 or belo <= 0:
        return
    base, inc = parse_time_control(h.get("TimeControl", ""))
    tc = time_class(base, inc)
    game_id = h.get("Site", h.get("GameId", "game")).rstrip("/").split("/")[-1]
    white, black = h.get("White", "?"), h.get("Black", "?")

    min_ply, max_ply = cfg["data"]["min_ply"], cfg["data"]["max_ply"]
    board = game.board()
    # running clocks (seconds remaining) per side; seed with base time
    clk = {chess.WHITE: float(base), chess.BLACK: float(base)}
    hist_uci = []
    rows = []
    for ply, node in enumerate(game.mainline()):
        mover = board.turn
        comment = node.comment or ""
        clk_after = parse_clk(comment)
        clk_before = clk[mover]
        time_spent = np.nan
        if clk_after is not None:
            time_spent = clk_before - clk_after + inc
            clk[mover] = clk_after
        ev_white = parse_eval(comment)              # White POV cp, evaluated AFTER the move
        fen = board.fen()
        played = node.move
        if min_ply <= ply <= max_ply and played in board.legal_moves:
            stm_elo = welo if mover == chess.WHITE else belo
            opp_elo = belo if mover == chess.WHITE else welo
            # store eval in side-to-move POV (eval comment reflects post-move; sign by mover)
            lich = np.nan if ev_white is None else (ev_white if mover == chess.WHITE else -ev_white)
            rows.append(dict(
                game_id=game_id, ply=ply, fen=fen,
                side_to_move_elo=stm_elo, oppo_elo=opp_elo,
                time_class=tc, base_time=base, increment=inc,
                clock_before=clk_before,
                clock_after=(np.nan if clk_after is None else clk_after),
                time_spent=time_spent, lichess_eval=lich,
                played_uci=played.uci(),
                player=(white if mover == chess.WHITE else black),
                hist_uci=" ".join(hist_uci),
            ))
        board.push(played)
        hist_uci.append(played.uci())
    return rows, tc, welo, belo


def reservoir_sample(games, cfg, per_cell, rng, max_games=0):
    """Balanced reservoir of GAMES per (elo_bin x time_class), capped at per_cell.
    Returns ({cell: [game_rows, ...]}, n_scanned); each game's value is its list of row dicts."""
    bins = cfg["data"]["elo_bins"]
    classes = set(cfg["data"]["time_classes"])
    reservoir, seen, n = {}, {}, 0
    for game in games:
        if max_games and n >= max_games:
            break
        n += 1
        out = game_rows(game, cfg)
        if not out:
            continue
        rows, tc, welo, belo = out
        if not rows or tc not in classes:
            continue
        b = elo_bin(max(welo, belo), bins)
        if b is None:
            continue
        cell = (b, tc)
        seen[cell] = seen.get(cell, 0) + 1
        res = reservoir.setdefault(cell, [])
        if len(res) < per_cell:
            res.append(rows)
        else:
            j = rng.integers(0, seen[cell])      # uniform reservoir replacement
            if j < per_cell:
                res[j] = rows
    return reservoir, n


def rowgroup_games(shard, rg_indices):
    """Yield chess.pgn games from specific row-groups of one HF parquet shard."""
    pf = pq.ParquetFile(shard)
    tags = [t for t in _PARQUET_TAGS if t in set(pf.schema_arrow.names)]
    for rg in rg_indices:
        tbl = pf.read_row_group(rg)
        mt = tbl.column("movetext")
        cols = {t: tbl.column(t) for t in tags}
        for i in range(len(mt)):
            movetext = mt[i].as_py()
            if not movetext:
                continue
            g = chess.pgn.read_game(io.StringIO(movetext))
            if g is None:
                continue
            for t in tags:
                v = cols[t][i].as_py()
                if v is not None and v != "":
                    g.headers[t] = str(v)
            yield g


def _parse_worker(task):
    wid, units, cfg, per_cell, parts_dir, seed = task
    path = os.path.join(parts_dir, f"part_{wid:04d}.parquet")
    if os.path.exists(path):                     # resume: this group is already done
        return path, -1, 0
    rng = np.random.default_rng(seed + wid)

    def games():
        for shard, grp in groupby(units, key=lambda u: u[0]):
            yield from rowgroup_games(shard, [rg for _, rg in grp])

    reservoir, n = reservoir_sample(games(), cfg, per_cell, rng)
    rows = [r for cell_games in reservoir.values() for game in cell_games for r in game]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path, len(rows), n


def parallel_parse_parquet(cfg, paths, workers, per_cell, fresh):
    """Fan parsing across cores by row-group; checkpoint each worker to a part file (resumable
    across a crash/restart); then merge and cap each cell to per_cell games."""
    d = cfg["data"]
    bins = d["elo_bins"]
    parts_dir = os.path.join(d["raw_dir"], "parse_parts")
    os.makedirs(parts_dir, exist_ok=True)
    if fresh:
        for f in glob.glob(os.path.join(parts_dir, "*.parquet")):
            os.remove(f)

    units = []
    for shard in paths:
        units += [(shard, rg) for rg in range(pq.ParquetFile(shard).num_row_groups)]
    per = math.ceil(len(units) / workers)
    groups = [units[w * per:(w + 1) * per] for w in range(workers)]
    tasks = [(w, g, cfg, per_cell, parts_dir, d["sample_seed"]) for w, g in enumerate(groups) if g]
    print(f"parallel parse: {len(units)} row-groups over {len(tasks)} workers "
          f"(checkpointed to {parts_dir}; re-run resumes)")
    with Pool(processes=len(tasks)) as pool:
        results = list(tqdm(pool.imap_unordered(_parse_worker, tasks), total=len(tasks),
                            desc="parse workers"))
    n_scanned = sum(n for _, _, n in results)
    n_resumed = sum(1 for _, r, _ in results if r == -1)
    if n_resumed:
        print(f"  ({n_resumed}/{len(tasks)} worker groups skipped via resume)")

    frames = [f for f in (pd.read_parquet(p) for p in glob.glob(os.path.join(parts_dir, "*.parquet")))
              if len(f)]
    if not frames:
        raise SystemExit("no positions parsed -- check the parquet source / filters")
    df = pd.concat(frames, ignore_index=True)

    # cap each (elo-bin x time-class) cell to per_cell games (seeded, across the pooled workers)
    edges = np.array(sorted(bins))
    bi = np.searchsorted(edges, np.maximum(df["side_to_move_elo"], df["oppo_elo"]), side="right") - 1
    df["_bin"] = np.where(bi >= 0, edges[bi.clip(0)], -1)
    rng = np.random.default_rng(d["sample_seed"])
    keep = set()
    for _, grp in df.groupby(["_bin", "time_class"]):
        gids = grp["game_id"].unique()
        if len(gids) > per_cell:
            gids = rng.choice(gids, per_cell, replace=False)
        keep.update(gids.tolist())
    kept = df[df["game_id"].isin(keep)]
    by_cell = kept.groupby(["_bin", "time_class"]).game_id.nunique()
    out = kept.drop(columns="_bin").copy()
    out.insert(0, "pos_id", out["game_id"].astype(str) + ":" + out["ply"].astype(str))
    out.to_parquet(d["positions"], index=False)

    print(f"scanned ~{n_scanned} games -> {len(out)} positions over "
          f"{out['game_id'].nunique()} sampled games -> {d['positions']}")
    print("games per (bin x time_class):")
    print(by_cell.to_string())


def main(cfg, args):
    d = cfg["data"]
    bins = d["elo_bins"]
    per_cell = args.games_per_cell or d["games_per_cell"]

    if args.parquet:
        paths = (sorted(glob.glob(os.path.join(args.parquet, "**", "*.parquet"), recursive=True))
                 if os.path.isdir(args.parquet) else [args.parquet])
        if not paths:
            raise SystemExit(f"no parquet files under {args.parquet}")
        if args.workers and args.workers > 1:
            parallel_parse_parquet(cfg, paths, args.workers, per_cell, args.fresh)
            return
        print(f"reading {len(paths)} parquet shard(s) from {args.parquet}")
        games = iter_games_parquet(paths)
    else:
        games = iter_games(open_pgn(args, cfg))

    rng = np.random.default_rng(d["sample_seed"])
    reservoir, n_games = reservoir_sample(tqdm(games, desc="scan games"), cfg, per_cell, rng,
                                          args.max_games)
    all_rows = [r for cell_games in reservoir.values() for game in cell_games for r in game]
    if not all_rows:
        raise SystemExit("no positions parsed -- check the source / filters")

    df = pd.DataFrame(all_rows)
    df.insert(0, "pos_id", df["game_id"].astype(str) + ":" + df["ply"].astype(str))
    df.to_parquet(d["positions"], index=False)

    print(f"scanned {n_games} games -> {len(df)} positions "
          f"over {df.game_id.nunique()} sampled games -> {d['positions']}")
    by_cell = (df.assign(bin=df.apply(lambda r: elo_bin(max(r.side_to_move_elo, r.oppo_elo), bins), axis=1))
                 .groupby(["bin", "time_class"]).game_id.nunique())
    print("games per (bin x time_class):")
    print(by_cell.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pgn", default=None, help="path to a .pgn or .pgn.zst file")
    ap.add_argument("--parquet", default=None, help="path to a HF Lichess parquet file or dir of shards")
    ap.add_argument("--stdin", action="store_true", help="read PGN from stdin")
    ap.add_argument("--max-games", type=int, default=0, help="stop after scanning N games (0 = all)")
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="decompress only the first N bytes of a .zst (parse a still-downloading prefix)")
    ap.add_argument("--games-per-cell", type=int, default=0, help="override cfg games_per_cell")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel parse workers for --parquet (1 = single-thread); fans over row-groups")
    ap.add_argument("--fresh", action="store_true", help="wipe parse_parts/ before a parallel parse")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a)
