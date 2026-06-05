"""
Smoke-only helper: synthesise a small PGN so `just smoke` runs end-to-end with no 30GB
Lichess download and no network. NOT part of the science pipeline -- real runs stream a
Lichess `.pgn.zst` dump through parse_games.py.

Games are lightweight weighted-random legal play (slight capture/centre bias) with realistic
headers (Elo sampled per cfg bin, a TimeControl, Result) and Lichess-style `%clk`/`%eval`
move comments, so every downstream parser path is exercised.

Run:
    python src/gen_smoke_games.py --config config.yaml --out data/raw/smoke.pgn --games 500
"""
import argparse
import os
import random

import chess
import chess.pgn
import yaml

TC_BY_CLASS = {  # base+inc strings that land in each time_class bucket
    "bullet": "60+0", "blitz": "300+0", "rapid": "600+5", "classical": "1800+30",
}
CENTER = {chess.E4, chess.D4, chess.E5, chess.D5, chess.C4, chess.F4, chess.C5, chess.F5}
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material_cp(board):
    """Crude White-POV material eval in centipawns (for a plausible %eval comment only)."""
    s = 0
    for sq, pc in board.piece_map().items():
        v = PIECE_VAL[pc.piece_type] * 100
        s += v if pc.color == chess.WHITE else -v
    return s


def move_weight(board, move):
    w = 1.0
    if board.is_capture(move):
        w += 3.0
    if move.to_square in CENTER:
        w += 1.5
    if board.gives_check(move):
        w += 1.0
    return w


def pick_move(board, rng):
    moves = list(board.legal_moves)
    weights = [move_weight(board, m) for m in moves]
    return rng.choices(moves, weights=weights, k=1)[0]


def clk_str(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def make_game(gid, rng, bins, classes, max_ply):
    tc = rng.choice(classes)
    base, inc = (int(x) for x in TC_BY_CLASS[tc].replace("+", " ").split())
    edge = rng.choice(bins)
    welo = edge + rng.randint(0, 199)
    belo = edge + rng.randint(0, 199)

    game = chess.pgn.Game()
    game.headers.update(
        Event="smoke", Site=f"https://smoke.local/{gid}",
        White=f"w_{welo}", Black=f"b_{belo}",
        WhiteElo=str(welo), BlackElo=str(belo),
        TimeControl=f"{base}+{inc}", Result="*",
    )
    board = game.board()
    clk = {chess.WHITE: float(base), chess.BLACK: float(base)}
    node = game
    n = rng.randint(40, max_ply)
    for _ in range(n):
        if board.is_game_over():
            break
        mover = board.turn
        mv = pick_move(board, rng)
        board.push(mv)
        spent = min(clk[mover], rng.uniform(1, max(2, base / 30)))
        clk[mover] = clk[mover] - spent + inc
        node = node.add_variation(mv)
        comment = f"[%clk {clk_str(clk[mover])}]"
        if rng.random() < 0.5:
            comment += f" [%eval {material_cp(board) / 100:.2f}]"
        node.comment = comment
    game.headers["Result"] = board.result(claim_draw=True) if board.is_game_over() else "1/2-1/2"
    return game


def main(cfg, out, games, seed):
    d = cfg["data"]
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for gid in range(games):
            g = make_game(gid, rng, d["elo_bins"], d["time_classes"], d["max_ply"])
            print(g, file=f, end="\n\n")
    print(f"wrote {games} games -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/raw/smoke.pgn")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.out, a.games, a.seed)
