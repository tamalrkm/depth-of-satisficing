"""
Stage 2: choose WHICH decisions get deep analysis.

The engine stage is the only real cost and it is cheap relative to the science,
so the DEFAULT is PASS-THROUGH: keep every non-book decision. The two-stage cost saver is
opt-in behind `--filter` for when the engine budget is tight.

`--filter` (uses the cheap lichess '%eval' already parsed in stage 1):
  - keep a decision if >= min_candidates plausible moves sit within `winprob_margin`
    win-prob of the best (i.e. it is a genuine choice, not a forced reply); since we only
    have a single scalar eval per position here, we approximate "is a choice" by requiring
    the position not be a near-decided/forced one (|eval| small enough to matter), and
  - always keep swing-down candidates when `keep_all_swing_down` is set.
Without a real shallow multi-PV pass the filter is necessarily coarse; the honest, default
behaviour is pass-through. A genuine shallow Stockfish gate belongs in run_engine, not here.

Output: data/selected.parquet (subset of positions.parquet, same schema)

Run:
    python src/select_positions.py --config config.yaml          # pass-through
    python src/select_positions.py --config config.yaml --filter  # cost-saver gate
"""
import argparse

import numpy as np
import pandas as pd
import yaml


def cp_to_winprob(cp):
    """Logistic map cp -> win prob, only for the cheap pre-filter (NOT the model's scale)."""
    return 1.0 / (1.0 + np.exp(-cp / 400.0))


def apply_filter(df, cfg):
    sel = cfg["select"]
    margin = sel["winprob_margin"]
    keep_swing = sel.get("keep_all_swing_down", True)

    have_eval = df["lichess_eval"].notna()
    wp = cp_to_winprob(df["lichess_eval"].fillna(0.0).to_numpy())
    # "genuine choice": position not already near-decided from the mover's POV.
    # near-decided => wp very high or very low (a forced/only-good-move situation).
    is_choice = (wp > margin) & (wp < 1.0 - margin)
    # swing-down proxy: mover stands clearly worse-than-equal after playing (eval <= 0),
    # i.e. the played move may look fine shallow but be a mistake -- always keep these.
    swing_down = df["lichess_eval"].fillna(0.0).to_numpy() <= 0.0

    keep = (~have_eval) | is_choice          # no eval -> keep (can't cheaply rule out)
    if keep_swing:
        keep = keep | (have_eval & swing_down)
    return df[keep].copy()


def main(cfg, do_filter):
    pos = pd.read_parquet(cfg["data"]["positions"])
    if do_filter:
        out = apply_filter(pos, cfg)
        mode = "filtered"
    else:
        out = pos.copy()
        mode = "pass-through"
    out.to_parquet(cfg["data"]["selected"], index=False)
    print(f"[{mode}] selected {len(out)}/{len(pos)} positions -> {cfg['data']['selected']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--filter", action="store_true", help="enable the swing-candidate cost-saver")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.filter)
