"""
Harvest chess.com games (D3 of prereg v2, OSF A6NYK) for the cross-pool IRT invariance test.

chess.com has no bulk dump; we use the public Published-Data API (no key), serially:
  titled players  /pub/titled/{GM,IM,...}   (player-dense; play many TCs)
  archive list    /pub/player/{u}/games/archives
  monthly games   /pub/player/{u}/games/{YYYY}/{MM}   (PGN with %clk, per-game pool rating)

Emits a positions parquet matching the Lichess schema so run_engine -> maia -> build_dataset
reuse unchanged:
  pos_id, game_id, ply, fen, side_to_move_elo, oppo_elo, time_class, base_time, increment,
  clock_before, clock_after, time_spent, lichess_eval(NaN), played_uci, player, source='chesscom'

    uv run python src/fetch_chesscom.py --titles GM --max-players 12 --months 3 --max-games-per-tc 40 \
        --out data/chesscom/positions_scope.parquet
"""
import argparse
import io
import json
import time
import urllib.request

import chess
import chess.pgn
import numpy as np
import pandas as pd

UA = "depth-of-satisficing-research/1.0 (mailto:tamal@gm.rkmvu.ac.in)"
BASE = "https://api.chess.com/pub"
TC_KEEP = {"bullet", "blitz", "rapid"}          # real-time only (skip 'daily' correspondence)
MIN_PLY, MAX_PLY = 9, 120


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** k)                # backoff
                continue
            raise
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def tc_from_header(tcstr):
    """chess.com TimeControl header: 'base+inc' or 'base' (seconds); '1/86400' = daily."""
    if not tcstr or "/" in tcstr:
        return 0, 0
    try:
        if "+" in tcstr:
            b, i = tcstr.split("+"); return int(float(b)), int(float(i))
        return int(float(tcstr)), 0
    except Exception:
        return 0, 0


def parse_game(g, max_ply=MAX_PLY):
    """Yield position rows from one chess.com game JSON dict."""
    if g.get("rules") != "chess" or not g.get("rated", False):
        return
    tc = g.get("time_class")
    if tc not in TC_KEEP:
        return
    pgn = g.get("pgn")
    if not pgn:
        return
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return
    h = game.headers
    base, inc = tc_from_header(h.get("TimeControl", ""))
    welo = g["white"].get("rating"); belo = g["black"].get("rating")
    white = g["white"].get("username", "?"); black = g["black"].get("username", "?")
    gid = "cc" + str(g.get("url", "")).rstrip("/").split("/")[-1]
    board = game.board()
    prev_clk = {chess.WHITE: float(base), chess.BLACK: float(base)}
    ply = 0
    for node in game.mainline():
        mv = node.move
        if mv is None:
            break
        ply += 1
        mover = board.turn
        fen = board.fen()
        clk_after = node.clock()                  # seconds left after the move (from %clk)
        clk_before = prev_clk[mover]
        if clk_after is not None:
            time_spent = clk_before - clk_after + inc
            prev_clk[mover] = clk_after
        else:
            time_spent = np.nan
        if MIN_PLY <= ply <= max_ply:
            stm_elo = welo if mover == chess.WHITE else belo
            opp_elo = belo if mover == chess.WHITE else welo
            yield dict(game_id=gid, ply=ply, fen=fen,
                       side_to_move_elo=stm_elo, oppo_elo=opp_elo,
                       time_class=tc, base_time=base, increment=inc,
                       clock_before=clk_before, clock_after=clk_after,
                       time_spent=time_spent, lichess_eval=np.nan,
                       played_uci=mv.uci(),
                       player=(white if mover == chess.WHITE else black),
                       source="chesscom")
        board.push(mv)


def main(a):
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    titles = a.titles.split(",")
    usernames = []
    for t in titles:
        d = get(f"{BASE}/titled/{t}")
        pl = d.get("players", []) if d else []
        usernames += pl
        print(f"  /titled/{t}: {len(pl)} players")
        time.sleep(0.3)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(usernames)
    usernames = usernames[: a.max_players]
    print(f"harvesting {len(usernames)} players, last {a.months} months, "
          f"cap {a.max_games_per_tc} games/player/TC")

    rows = []
    per_tc_games = {}
    for n, u in enumerate(usernames, 1):
        arch = get(f"{BASE}/player/{u}/games/archives")
        time.sleep(0.25)
        months = (arch.get("archives", []) if arch else [])[-a.months:]
        cap = {"bullet": 0, "blitz": 0, "rapid": 0}
        for murl in reversed(months):              # newest first
            data = get(murl); time.sleep(0.25)
            if not data:
                continue
            for g in data.get("games", []):
                tc = g.get("time_class")
                if tc in TC_KEEP and cap.get(tc, 9e9) < a.max_games_per_tc:
                    try:
                        got = list(parse_game(g))
                    except Exception:
                        got = []                  # skip any malformed game, keep harvesting
                    if got:
                        rows += got; cap[tc] += 1
                        per_tc_games[tc] = per_tc_games.get(tc, 0) + 1
        if n % 10 == 0 or n == len(usernames):
            print(f"  [{n}/{len(usernames)}] {u}: cumulative {len(rows):,} positions")

    if not rows:
        raise SystemExit("no positions harvested")
    df = pd.DataFrame(rows)
    df.insert(0, "pos_id", df["game_id"].astype(str) + ":" + df["ply"].astype(str))
    df = df.drop_duplicates("pos_id")
    df.to_parquet(a.out, index=False)
    print(f"\nwrote {len(df):,} positions ({df.game_id.nunique():,} games, "
          f"{df.player.nunique():,} players) -> {a.out}")
    print("per time-class positions:", df.time_class.value_counts().to_dict())
    print("games per time-class:", per_tc_games)
    # per-player x TC density (for H-INV reliability planning)
    dens = df.groupby(["player", "time_class"]).size()
    print(f"decisions per (player x TC): median {int(dens.median())}, "
          f"players with >=100 in >=2 TCs: "
          f"{(dens[dens>=100].reset_index().groupby('player').size()>=2).sum()}")
    print(f"side-to-move rating range: [{df.side_to_move_elo.min()}, {df.side_to_move_elo.max()}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", default="GM")
    ap.add_argument("--max-players", type=int, default=12)
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--max-games-per-tc", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="data/chesscom/positions_scope.parquet")
    main(ap.parse_args())
