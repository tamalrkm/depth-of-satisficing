"""
Stage 6: ingest local Lichess broadcast dumps into the elite-OTB stratum.

The monthly broadcast dumps (lichess_db_broadcast_YYYY-MM.pgn.zst) are relayed OTB tournament
games -- FIDE-rated, overwhelmingly classical. We keep the strong ones as a SEPARATE stratum
(`source="broadcast"`) for the elite/champion analyses and as the deepest-deliberation test of
the model comparison (state-only Maia should fail hardest here; the depth component should
recover the gap).

Broadcast-specific handling:
  - TimeControl is free-text and unit-inconsistent ("90+30" can mean 90 minutes), so we do NOT
    trust it: every kept game is tagged time_class="classical".
  - Elos are FIDE (a different scale from Lichess-online), kept as-is and tagged
    source="broadcast" so downstream never merges them into the online (bin x class) bands.
  - clocks are present in ~half the games (relay-dependent).
  - games get unique ids `bc{N}` so pos_id stays game-unique (and the next-position played-move
    trick in build_dataset still works across consecutive plies).

Filter: keep games with min(WhiteElo, BlackElo) >= --min-elo; uniformly cap to --max-games.

Run:
    python src/fetch_broadcasts.py --config config.yaml --dir gdrive/broadcast \
        --min-elo 2500 --max-games 5000
    # then fold into the engine pipeline:
    python src/fetch_broadcasts.py --config config.yaml --dir gdrive/broadcast --merge
"""
import argparse
import glob
import io
import os
import re

import chess.pgn
import numpy as np
import pandas as pd
import yaml
import zstandard

from parse_games import game_rows

WELO = re.compile(r'\[WhiteElo "(\d+)"')
BELO = re.compile(r'\[BlackElo "(\d+)"')
VARIANT = re.compile(r'\[Variant "([^"]+)"')
BROADCAST_POS = "data/positions_broadcast.parquet"


def iter_elite_blocks(path, min_elo, max_elo):
    """Yield raw PGN game blocks for HUMAN, STANDARD games with both players in [min_elo, max_elo]
    (header regex only -- we full-parse only games that pass). The Elo ceiling drops engine
    events (TCEC etc., rated 3000+); the Variant check drops Chess960/Fischer-Random."""
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore").read()
    for block in re.split(r"\n\n(?=\[Event )", text):
        v = VARIANT.search(block)
        if v and v.group(1).strip().lower() != "standard":
            continue                                   # skip Chess960 / Fischer-Random / other
        we, be = WELO.search(block), BELO.search(block)
        if not we or not be:
            continue
        lo, hi = sorted((int(we.group(1)), int(be.group(1))))
        if lo >= min_elo and hi <= max_elo:            # exclude engines (Elo above human ceiling)
            yield block


def sample_blocks(files, min_elo, max_elo, max_games, seed):
    rng = np.random.default_rng(seed)
    reservoir, n = [], 0
    for f in files:
        for block in iter_elite_blocks(f, min_elo, max_elo):
            n += 1
            if len(reservoir) < max_games:
                reservoir.append(block)
            else:
                j = rng.integers(0, n)
                if j < max_games:
                    reservoir[j] = block
        print(f"  {os.path.basename(f)}: elite-so-far={n}, kept={len(reservoir)}", flush=True)
    return reservoir, n


def build_positions(cfg, blocks):
    rows = []
    for idx, block in enumerate(blocks):
        try:
            g = chess.pgn.read_game(io.StringIO(block))
            if g is None or g.errors:        # skip games python-chess flagged (bad/variant SAN)
                continue
            out = game_rows(g, cfg)
        except Exception:
            continue
        if not out:
            continue
        grows, _tc, _welo, _belo = out
        bc_id = f"bc{idx}"
        for r in grows:
            r["game_id"] = bc_id              # unique id; keeps pos_id game-unique
            r["time_class"] = "classical"     # don't trust free-text broadcast TC
            r["source"] = "broadcast"
        rows.extend(grows)
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(0, "pos_id", df["game_id"].astype(str) + ":" + df["ply"].astype(str))
    return df


def merge_into_main(cfg):
    """Concatenate online positions (+ source="online") with the broadcast positions, writing
    positions.parquet and selected.parquet so the engine resume picks up the new FENs."""
    d = cfg["data"]
    bc = pd.read_parquet(BROADCAST_POS)
    online = pd.read_parquet(d["positions"])
    if "source" not in online.columns:
        online["source"] = "online"
    merged = pd.concat([online, bc], ignore_index=True)
    merged.to_parquet(d["positions"], index=False)
    merged.to_parquet(d["selected"], index=False)   # pass-through select
    print(f"merged: {len(online)} online + {len(bc)} broadcast = {len(merged)} positions "
          f"-> {d['positions']} & {d['selected']}")
    print("  re-run run_engine (no --fresh) to analyse the new broadcast FENs via resume")


def main(cfg, d_dir, min_elo, max_elo, max_games, seed, merge):
    if merge:
        merge_into_main(cfg)
        return
    files = sorted(glob.glob(os.path.join(d_dir, "*.pgn.zst")))
    if not files:
        raise SystemExit(f"no broadcast .pgn.zst under {d_dir}")
    print(f"scanning {len(files)} broadcast months "
          f"(human standard, FIDE in [{min_elo},{max_elo}], cap {max_games})")
    blocks, n_elite = sample_blocks(files, min_elo, max_elo, max_games, seed)
    df = build_positions(cfg, blocks)
    if not len(df):
        raise SystemExit("no broadcast positions kept (check --min-elo)")
    df.to_parquet(BROADCAST_POS, index=False)
    elos = pd.concat([df.groupby("game_id").side_to_move_elo.first()])
    print(f"elite broadcast: {n_elite} games >= FIDE{min_elo}; kept {df.game_id.nunique()} games "
          f"-> {len(df)} positions -> {BROADCAST_POS}")
    print(f"  kept-game Elo: median={int(elos.median())} p10={int(elos.quantile(.1))} "
          f"max={int(elos.max())};  clocks present in "
          f"{100*df.time_spent.notna().mean():.0f}% of positions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dir", default="gdrive/broadcast", help="dir of lichess_db_broadcast_*.pgn.zst")
    ap.add_argument("--min-elo", type=int, default=2500, help="keep games with min(both Elos) >= this")
    ap.add_argument("--max-elo", type=int, default=2900,
                    help="human Elo ceiling; excludes engine events (TCEC etc.) rated above this")
    ap.add_argument("--max-games", type=int, default=5000, help="uniform cap on kept elite games")
    ap.add_argument("--merge", action="store_true", help="merge broadcast positions into the main pipeline")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    main(cfg, a.dir, a.min_elo, a.max_elo, a.max_games, cfg["data"]["sample_seed"], a.merge)
