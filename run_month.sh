#!/usr/bin/env bash
# Full confirmatory-replication pipeline for ONE Lichess month, driven entirely by a
# month config (config_2026_06.yaml etc). Mirrors the stage sequence used for 2026-04/05.
#
#   ./run_month.sh config_2026_06.yaml            # full run (source = database.lichess.org)
#   ./run_month.sh config_2026_06.yaml --dry-run  # readiness check only: no engine, no train
#   ./run_month.sh config_2026_06.yaml --hf       # use the HF parquet mirror instead
#
# SOURCE: default is the raw site database.lichess.org (--http path). The HF mirror lags
# ~9 months (only reaches 2025-09 as of 2026-06), so it is NOT usable for recent months;
# --hf is kept only for back-months that the mirror has caught up on.
#
# Stages: download -> parse -> select -> broadcasts(merge) -> engine -> maia -> build
#         -> train -> replicate. The engine stage is the long one (~10h on this box).
# Everything is config-driven; the only per-month edits live in the config (month + paths).
set -euo pipefail

CFG="${1:?usage: run_month.sh <config.yaml> [--dry-run] [--http]}"
shift || true
DRY=0; SRC="http"
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --hf)      SRC="hf" ;;
    --http)    SRC="http" ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

MONTH=$(grep -oP 'lichess_month:\s*\K[0-9-]+' "$CFG")
REPL=$(dirname "$(grep -oP 'train_tensor:\s*\K\S+' "$CFG")")   # e.g. data/repl06
HF_DIR="data/raw/hf/${MONTH}"
LOG="${REPL}/pipeline.log"

banner(){ echo "=== $* $(date -u) ==="; }
run(){ echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

echo "config=$CFG  month=$MONTH  repl_dir=$REPL  source=$SRC  dry_run=$DRY"
mkdir -p "$REPL" "${REPL}/shards"
[ "$DRY" = 1 ] || exec > >(tee -a "$LOG") 2>&1

# Prefer a local dump (already copied to gdrive/ or data/raw/) over re-downloading 28 GB.
DUMP=""
for cand in "gdrive/lichess_db_standard_rated_${MONTH}.pgn.zst" \
            "data/raw/lichess_db_standard_rated_${MONTH}.pgn.zst"; do
  [ -f "$cand" ] && { DUMP="$cand"; break; }
done

banner "DOWNLOAD ($MONTH, src=$SRC)"
if [ -n "$DUMP" ]; then
  echo "  local dump found -> $DUMP (skipping download)"
  PARSE_SRC=(--pgn "$DUMP")
elif [ "$SRC" = hf ]; then
  run uv run python src/download_data.py --config "$CFG"
  PARSE_SRC=(--parquet "$HF_DIR")
else
  run uv run python src/download_data.py --config "$CFG" --source http
  PARSE_SRC=(--pgn "data/raw/lichess_db_standard_rated_${MONTH}.pgn.zst")
fi

banner "PARSE"
# --max-games caps the reservoir SCAN at 2.5M games (prereg protocol; 2026-04/05 used the
# same cap). Without it the scan runs the whole ~90M-game month AND, because sampling is a
# uniform reservoir, yields a different balanced sample -> not comparable across months.
# NOTE: parse_games only parallelizes the PARQUET path; for a .pgn.zst the scan is
# single-threaded, so --workers is intentionally omitted here.
MAX_SCAN=2500000
run uv run python src/parse_games.py --config "$CFG" "${PARSE_SRC[@]}" --max-games "$MAX_SCAN" --fresh

banner "SELECT (pass-through)"
run uv run python src/select_positions.py --config "$CFG"

# Elite-OTB stratum: month-matched broadcast only (as in 2026-04/05), NOT the full archive.
BC_DIR="gdrive/broadcast_${MONTH//-/_}"
BC_MASTER="gdrive/broadcast/lichess_db_broadcast_${MONTH}.pgn.zst"
if [ ! -d "$BC_DIR" ] && [ -f "$BC_MASTER" ]; then
  echo "  building month-matched broadcast dir $BC_DIR (hardlink from archive)"
  [ "$DRY" = 1 ] || { mkdir -p "$BC_DIR"; ln -f "$BC_MASTER" "$BC_DIR/"; }
fi
if [ -d "$BC_DIR" ] || [ -f "$BC_MASTER" ]; then
  banner "BROADCAST PARSE + MERGE (elite-OTB stratum: $BC_DIR)"
  run uv run python src/fetch_broadcasts.py --config "$CFG" --dir "$BC_DIR" --min-elo 2500 --max-elo 2900 --max-games 5000
  run uv run python src/fetch_broadcasts.py --config "$CFG" --merge
else
  echo "  (no month-matched broadcast for $MONTH; skipping elite-OTB stratum)"
fi

banner "ENGINE (node-capped, workers from config -- the ~10h stage)"
run uv run python src/run_engine.py --config "$CFG"

banner "MAIA"
run uv run python src/maia_features.py --config "$CFG"

banner "BUILD"
run uv run python src/build_dataset.py --config "$CFG"

banner "TRAIN"
run uv run python src/train.py --config "$CFG"

banner "REPLICATE (confirmatory H1-H7, prereg A6NYK)"
run uv run python src/replicate.py --config "$CFG"

banner "PIPELINE DONE"
echo "results: ${REPL}/  | replication verdict above | figs in paper/figs_repl/"
