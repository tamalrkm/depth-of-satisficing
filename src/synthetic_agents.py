"""
Stage 8 (implemented): synthetic agents that satisfice at a KNOWN depth.

Each agent, at every move, runs Stockfish to a fixed depth d_plant and plays a soft-max
over the depth-d_plant evaluations (inverse temperature beta_gen). We then run the FULL
pipeline on these games and check that the model's estimated depth-of-satisficing recovers
d_plant. This is the ground-truth check that the instrument measures depth, not difficulty.

Run:
    python src/synthetic_agents.py --config config.yaml
Produces PGNs under data/synthetic/ ; feed them through parse->select->engine->maia->train,
then compare analyze.py's per-agent depth estimate to the planted depth.
"""
import argparse, os, yaml, random
import chess, chess.engine, chess.pgn


def winprob(info):
    wdl = info.get("wdl")
    if wdl is not None:
        rel = wdl.relative
        return (rel.wins + 0.5 * rel.draws) / 1000.0
    return info["score"].relative.wdl(model="sf").expectation()


def agent_move(engine, board, d_plant, beta_gen, multipv):
    """Soft-max over depth-d_plant win-probs (side-to-move POV)."""
    vals = {}
    with engine.analysis(board, chess.engine.Limit(depth=d_plant), multipv=multipv) as an:
        for info in an:
            if info.get("depth") == d_plant and "pv" in info:
                vals[info["pv"][0]] = winprob(info)
    if not vals:
        return random.choice(list(board.legal_moves))
    moves, ws = zip(*vals.items())
    import math
    z = [math.exp(beta_gen * w) for w in ws]
    s = sum(z)
    r, acc = random.random() * s, 0.0
    for mv, zz in zip(moves, z):
        acc += zz
        if acc >= r:
            return mv
    return moves[-1]


def play_game(engine, d_plant, beta_gen, multipv, max_ply=120):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"synthetic_depth_{d_plant}"
    game.headers["WhiteElo"] = game.headers["BlackElo"] = str(1000 + 100 * d_plant)
    node = game
    while not board.is_game_over() and board.ply() < max_ply:
        mv = agent_move(engine, board, d_plant, beta_gen, multipv)
        board.push(mv)
        node = node.add_variation(mv)
    return game


def main(cfg):
    s = cfg["synthetic"]
    eng = chess.engine.SimpleEngine.popen_uci(cfg["engine"]["path"])
    eng.configure({"Threads": cfg["engine"]["threads"], "Hash": cfg["engine"]["hash_mb"]})
    try:
        eng.configure({"UCI_ShowWDL": True})
    except chess.engine.EngineError:
        pass
    os.makedirs("data/synthetic", exist_ok=True)
    for d in s["planted_depths"]:
        path = f"data/synthetic/depth_{d}.pgn"
        with open(path, "w") as f:
            for _ in range(s["games_per_depth"]):
                g = play_game(eng, d, s["softmax_beta_gen"], cfg["engine"]["multipv"])
                print(g, file=f, end="\n\n")
        print(f"wrote {path}")
    eng.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
