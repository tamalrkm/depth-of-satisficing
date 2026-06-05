"""
Stage 4 (implemented): Maia-3 policy as the state-tower prior `q`.

We read Maia-3's FULL policy distribution over the legal moves of each position,
conditioned on the side-to-move's Elo (and the opponent's Elo), straight from the
model's policy head -- not a single best move.

Maia-3 API (verified against the installed `maia3` package, commit-pinned in pyproject):
  - `maia3.models.MAIA3Model(cfg).forward(tokens, self_elos, oppo_elos)`
        -> logits_move [B, 4352], logits_value [B, 3], logits_ponder [B]
  - move vocabulary is `maia3.utils.get_all_possible_moves()` (4096 from-to + 256
    promotions); the board is MIRRORED when Black is to move, so a predicted index is
    decoded with `mirror_move(...)` for Black, and legality is taken from
    `maia3.dataset.get_legal_moves_mask(board, move2idx)`.
  - inputs are built by `tokenize_board` + `get_historical_tokens` (history of 8 boards;
    time dims are ignored by the move head).
The full canonical reference is `maia3/uci.py::Maia3UCIEngine.score_moves`.

Output schema (data/maia_q.parquet), one row per (position, legal move):
    pos_id  str
    move    str   UCI (in the real board frame, directly comparable to played_uci)
    q       float Maia-3 policy probability, normalised over the legal moves

Run:
    python src/maia_features.py --config config.yaml [--model maia3-5m] [--limit N]
"""
import argparse
from collections import deque

import chess
import pandas as pd
import torch
import yaml
from torch.amp import autocast
from tqdm import tqdm

from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import (
    apply_model_config,
    resolve_checkpoint_path,
    resolve_model_spec,
)
from maia3.uci import load_model
from maia3.utils import get_all_possible_moves, mirror_move

ALL_MOVES = get_all_possible_moves()
MOVE2IDX = {m: i for i, m in enumerate(ALL_MOVES)}


def build_maia_cfg(model_name, device):
    """Construct the architecture/inference cfg the same way `maia3-uci` does."""
    spec = resolve_model_spec(model_name)
    cfg = argparse.Namespace(use_relative_bias=False, use_absolute_pe=False)
    apply_model_config(cfg, spec)            # overlays BASE_SIZE_CONFIG + size preset
    cfg.device = device
    cfg.trust_checkpoint = False
    cfg.use_amp = device.startswith("cuda")
    cfg.checkpoint_path = resolve_checkpoint_path(spec)
    return cfg, spec


def tokens_for(fen, hist_uci, mcfg):
    """Return (token_tensor [64,F], board).

    If `hist_uci` (space-joined UCI moves from the startpos to this position) is given,
    replay it to build a faithful deque of the last `history` boards; otherwise replicate
    the current position (the supported `use_uci_history=False` inference mode).

    If the replay fails (e.g., a broadcast game with a non-standard starting position whose
    Variant header was missing -- a Chess960 or analysis-from-position game), we silently
    fall back to position-only history."""
    H = mcfg.history
    if hist_uci:
        try:
            board = chess.Board()
            hist = deque([tokenize_board(board)], maxlen=H)
            for u in hist_uci.split():
                board.push(chess.Move.from_uci(u))
                hist.append(tokenize_board(board))
            if board.board_fen() != chess.Board(fen).board_fen():
                raise ValueError("history replay drifted from stored FEN")
        except (AssertionError, ValueError):
            board = chess.Board(fen)
            hist = deque([tokenize_board(board)], maxlen=H)
    else:
        board = chess.Board(fen)
        hist = deque([tokenize_board(board)], maxlen=H)
    return get_historical_tokens(hist, mcfg, 0.0, 0.0, 0.0, 0.0), board


def main(cfg, model_name=None, device=None, batch_size=256, limit=0):
    model_name = model_name or cfg["maia"]["model"]
    device = device or cfg["maia"].get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    mcfg, spec = build_maia_cfg(model_name, device)
    model = load_model(mcfg)

    sel = pd.read_parquet(cfg["data"]["selected"])
    if limit:
        sel = sel.iloc[:limit]
    use_hist = bool(cfg["maia"].get("use_uci_history", True)) and "hist_uci" in sel.columns

    rows = []
    buf_tok, buf_self, buf_oppo, buf_board, buf_pos = [], [], [], [], []

    def flush():
        if not buf_tok:
            return
        toks = torch.stack(buf_tok).to(device)
        se = torch.tensor(buf_self, dtype=torch.long, device=device)
        oe = torch.tensor(buf_oppo, dtype=torch.long, device=device)
        with torch.no_grad(), autocast("cuda", enabled=mcfg.use_amp):
            logits_move, _, _ = model(toks, se, oe)
        logits_move = logits_move.float()
        for k, board in enumerate(buf_board):
            mask = get_legal_moves_mask(board, MOVE2IDX).to(device)
            probs = torch.softmax(logits_move[k].masked_fill(~mask, float("-inf")), dim=-1)
            white = board.turn == chess.WHITE
            for mv in board.legal_moves:
                key = mv.uci() if white else mirror_move(mv.uci())
                idx = MOVE2IDX.get(key)
                if idx is not None:
                    rows.append((buf_pos[k], mv.uci(), float(probs[idx])))
        buf_tok.clear(); buf_self.clear(); buf_oppo.clear()
        buf_board.clear(); buf_pos.clear()

    for r in tqdm(sel.itertuples(), total=len(sel), desc=f"Maia-3 [{spec.display_name}]"):
        hist_uci = getattr(r, "hist_uci", "") if use_hist else ""
        tokens, board = tokens_for(r.fen, hist_uci or "", mcfg)
        if not any(board.legal_moves):
            continue
        buf_tok.append(tokens)
        buf_self.append(int(r.side_to_move_elo))
        buf_oppo.append(int(getattr(r, "oppo_elo", r.side_to_move_elo)))
        buf_board.append(board)
        buf_pos.append(r.pos_id)
        if len(buf_tok) >= batch_size:
            flush()
    flush()

    df = pd.DataFrame(rows, columns=["pos_id", "move", "q"])
    df.to_parquet(cfg["data"]["maia_q"], index=False)
    print(f"wrote {len(df)} (pos,move) rows over {df.pos_id.nunique()} positions "
          f"-> {cfg['data']['maia_q']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="override cfg.maia.model (e.g. maia3-5m)")
    ap.add_argument("--device", default=None, help="override cfg.maia.device")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="cap #positions (0 = all)")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.model, a.device, a.batch_size, a.limit)
