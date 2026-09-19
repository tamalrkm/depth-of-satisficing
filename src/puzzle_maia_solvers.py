"""Person side for the puzzle validation: simulated solvers of graded skill, from Maia-3.

Lichess publishes a puzzle's human difficulty but not who attempted it, so the person side of
the item-response picture is missing. Maia-3 supplies it: the policy head is conditioned on a
rating, so running the same puzzle position through Maia at a ladder of ratings yields a
synthetic solver of that strength. For each puzzle we record, at each rating, the probability
the model assigns to the puzzle's solution and whether the solution is its top move.

This buys two things the engine analysis alone cannot:
  * a person axis -- solve probability as a function of solver strength, per item;
  * a separation of PATTERN from DEPTH. Maia does not search. A puzzle Maia solves at every
    rating is pattern-findable; one it fails at every rating needs search. The decisive test is
    whether our engine depth features still predict the human puzzle rating after controlling
    for how findable the solution is by pattern alone.

Run (after src.puzzle_validation sample):
    uv run python -m src.puzzle_maia_solvers --out DIR [--ratings 1100,1300,...]
"""
import argparse
from collections import deque

import chess
import numpy as np
import pandas as pd
import torch
import yaml
from torch.amp import autocast
from tqdm import tqdm

from maia3.dataset import get_legal_moves_mask, tokenize_board
from maia3.utils import get_all_possible_moves, mirror_move

from .maia_features import build_maia_cfg, tokens_for
from maia3.uci import load_model

ALL_MOVES = get_all_possible_moves()
MOVE2IDX = {m: i for i, m in enumerate(ALL_MOVES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ratings", default="1100,1300,1500,1700,1900,2100,2300,2500")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ratings = [int(r) for r in a.ratings.split(",")]

    sel = pd.read_parquet(f"{a.out}/selected.parquet")      # pos_id, fen, played_uci (= solution)
    device = cfg["maia"].get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    mcfg, spec = build_maia_cfg(cfg["maia"]["model"], device)
    model = load_model(mcfg)
    print(f"{len(sel)} puzzles x {len(ratings)} simulated solver ratings [{spec.display_name}]")

    # tokenise once; the position is all Maia needs (puzzle FENs carry no move history)
    toks, boards, keep = [], [], []
    for r in sel.itertuples():
        t, b = tokens_for(r.fen, "", mcfg)
        if any(b.legal_moves):
            toks.append(t); boards.append(b); keep.append((r.pos_id, r.played_uci))

    rows = []
    for elo in ratings:
        for i in tqdm(range(0, len(toks), a.batch_size), desc=f"Maia@{elo}", leave=False):
            tb = torch.stack(toks[i:i + a.batch_size]).to(device)
            e = torch.full((len(tb),), elo, dtype=torch.long, device=device)
            with torch.no_grad(), autocast("cuda", enabled=mcfg.use_amp):
                logits, _, _ = model(tb, e, e)
            logits = logits.float()
            for k, board in enumerate(boards[i:i + a.batch_size]):
                pid, sol = keep[i + k]
                mask = get_legal_moves_mask(board, MOVE2IDX).to(device)
                probs = torch.softmax(logits[k].masked_fill(~mask, float("-inf")), dim=-1)
                white = board.turn == chess.WHITE
                key = sol if white else mirror_move(sol)
                idx = MOVE2IDX.get(key)
                if idx is None:
                    continue
                top = int(probs.argmax())
                rows.append((pid, elo, float(probs[idx]), int(top == idx)))
    df = pd.DataFrame(rows, columns=["puzzle_id", "solver_rating", "p_solution", "top1"])
    df.to_parquet(f"{a.out}/maia_solvers.parquet", index=False)
    g = df.groupby("solver_rating").agg(p=("p_solution", "mean"), top1=("top1", "mean"))
    print("\nsimulated solver ladder (mean over puzzles):")
    for r_, row in g.iterrows():
        print(f"  Maia@{r_}: P(solution)={row.p:.3f}  top-1 solve rate={row.top1:.3f}")
    print(f"\nwrote {a.out}/maia_solvers.parquet ({len(df):,} rows)")


if __name__ == "__main__":
    main()
