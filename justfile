# Run with `just <target>`. Everything goes through `uv run` (no venv activation).
set shell := ["bash", "-cu"]

setup:
    uv sync
    uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# fetch data for the configured month. Default source = official Lichess parquet on HuggingFace
# (fast, no throttle); falls back to database.lichess.org .zst with `--source http`.
download:
    uv run python src/download_data.py --config config.yaml
    @echo "then: uv run python src/parse_games.py --config config.yaml --parquet data/raw/hf/$(grep -oP 'lichess_month:\s*\"\K[^\"]+' config.yaml)"

parse:
    uv run python src/parse_games.py --config config.yaml

select:
    uv run python src/select_positions.py --config config.yaml

engine:
    uv run python src/run_engine.py --config config.yaml

maia:
    uv run python src/maia_features.py --config config.yaml

dataset:
    uv run python src/build_dataset.py --config config.yaml

train:
    uv run python src/train.py --config config.yaml

analyze:
    uv run python src/analyze.py --config config.yaml

synthetic:
    uv run python src/synthetic_agents.py --config config.yaml

# ingest the local Lichess broadcast dumps (gdrive/broadcast/*.pgn.zst) as the elite-OTB
# stratum (FIDE 2500-2900 standard; engines/960 excluded), then --merge into selected.
broadcasts:
    uv run python src/fetch_broadcasts.py --config config.yaml --dir gdrive/broadcast --min-elo 2500 --max-elo 2900 --max-games 5000
    uv run python src/fetch_broadcasts.py --config config.yaml --merge

# back up the hard-to-recompute artifacts (engine + Maia + sampled games + config) so a
# retrain never needs to re-run Stockfish. Run after the engine completes; add shards/ for
# mid-run disk-failure protection.
backup dest="gdrive/backup":
    mkdir -p {{dest}}
    cp -v config.yaml data/depth_traj.parquet data/maia_q.parquet data/positions.parquet data/selected.parquet {{dest}}/ 2>/dev/null || true
    @echo "backed up engine+maia+games+config -> {{dest}}"

# tiny end-to-end check
# ~500 games -> pass-through select -> ~300 positions through SF18 -> Maia-3 (5m) ->
# tensors -> 2-epoch train -> confirm depth_of_satisficing is sane. Self-contained (no
# 30GB Lichess download, no network beyond the one-time Maia3-5M checkpoint fetch).
smoke:
    uv run python src/gen_smoke_games.py --config config.yaml --out data/raw/smoke.pgn --games 500
    uv run python src/parse_games.py --config config.yaml --pgn data/raw/smoke.pgn
    uv run python src/select_positions.py --config config.yaml
    # smoke uses reduced depth/multipv for speed (still 7 real grid depths); production is D=21,K=9
    uv run python src/run_engine.py --config config.yaml --workers 32 --limit 200 --depth 14 --multipv 8
    uv run python src/maia_features.py --config config.yaml --model maia3-5m --limit 200
    uv run python src/build_dataset.py --config config.yaml
    uv run python src/train.py --config config.yaml --epochs 2
    uv run python src/smoke_check.py --config config.yaml
