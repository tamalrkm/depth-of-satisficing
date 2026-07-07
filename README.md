# Errors reveal the depth of human reasoning

Measuring the **depth of satisficing** — how far a person searches before committing — from the
structure of human *mistakes* in chess, validated on online and elite over-the-board play, an
independent platform (chess.com), and synthetic ground truth. Manuscript: `paper/manuscript.tex`
(Springer Nature / NHB format).

A player is modelled as a latent distribution over engine search depths; the move played is a
product-of-experts fusion of (a) **Maia-3**'s human-pattern policy and (b) a soft-max over
**Stockfish 18** regret at the depth searched. The depth of satisficing is the posterior over
effective depth given the played move.

## Key results
- **Skill lives in the errors, at depth.** Players of different skill differ little on the moves
  they get right; the rating signal concentrates in the small fraction of **swing-up
  (deep-discovery)** decisions — moves whose worth appears only with depth.
- **Depth rises with skill in slow play** (classical/rapid: Spearman ≈ +0.47/+0.54, robust across
  three months), **weak/variable in blitz** (+0.43 / ~0 / +0.19), **null in bullet** (negative
  control).
- **Non-circular validation:** a clock-free model recovers real thinking time (ρ ≈ +0.30–0.40,
  middlegame-peaked) — twice-replicated.
- **Selective prediction:** a depth-aware model beats Maia-3 exactly on slow, high-swing
  decisions (registered swing×time-control interaction; mixed-effects p < 10⁻³⁶).
- **Item-response view:** positions are items with engine-anchored *critical-depth difficulty*;
  **swing is item discrimination** (ability×swing < 0) — replicated across three pools
  (Lichess 2025-09, chess.com, Lichess 2026-04).
- **Identifiability:** recovers planted depth in synthetic agents (ordinal).

## Pre-registrations (OSF)
- First replication (Lichess 2026-05): `10.17605/OSF.IO/KB4ZQ`
- Corrected, prospective confirmation (Lichess 2026-04 robustness done; 2026-06 primary pending):
  `10.17605/OSF.IO/A6NYK`
- Pre-registration documents: `paper/preregistration.md`, `paper/preregistration_v2.md`

## Repository layout
```
paper/        manuscript.tex (+ refs.bib, sn-jnl.cls), pre-registrations, figs/, figs_irt/
src/          pipeline + analysis (see below)
config*.yaml  one config per dataset (2025-09 primary, 2026-05, 2026-04, chess.com, synthetic)
data/         derived datasets (git-ignored; deposited on OSF — see DATA_MANIFEST.md)
```

## Pipeline (reproduce)
Environment: `uv` (Python). External, user-supplied: **Stockfish 18** (`config.yaml: engine.path`,
net `nn-71d6d32cb962`) and **Maia-3** (`maia3-79m`). Engine stage is CPU-bound; everything else is light.

```bash
# 1. positions  (Lichess month dump, or chess.com via fetch_chesscom.py)
uv run python src/parse_games.py    --config config.yaml --pgn <month.pgn.zst> --max-games 2500000
uv run python src/fetch_broadcasts.py --config config.yaml --dir <broadcast_dir> --merge   # elite OTB
uv run python src/select_positions.py --config config.yaml                                  # pass-through
# 2. engine (Stockfish 18, depth 21, MultiPV 9, node cap 2e8; 128 workers, NUMA-bound)
uv run python src/run_engine.py     --config config.yaml
# 3. human-pattern prior + model tensors
uv run python src/maia_features.py  --config config.yaml
uv run python src/build_dataset.py  --config config.yaml          # -> data/train.pt
# 4. results
uv run python src/analyze.py        --config config.yaml --result all   # E1-E6 + figures
uv run python src/replicate.py      --config config_2026_05.yaml         # pre-registered H1-H7
uv run python src/irt_grm.py        config.yaml                          # item-response (IRT) layer
```
chess.com pipeline: `fetch_chesscom.py` → same `run_engine`/`maia_features`/`build_dataset` with
`config_chesscom.yaml`. Synthetic identifiability: `synthetic_agents.py` (+ `config_syn.yaml`).

| script | role |
|---|---|
| `parse_games.py` / `fetch_chesscom.py` / `fetch_broadcasts.py` | ingest Lichess / chess.com / elite-OTB |
| `select_positions.py` | pass-through (keep all non-book decisions) |
| `run_engine.py` | parallel Stockfish driver (depth-resolved regret, WDL win-prob) |
| `maia_features.py` | Maia-3 policy per legal move |
| `build_dataset.py` | join → padded model tensors (`delta`, `logq`, `context`, swing, `y`) |
| `model.py` / `train.py` | latent-depth product-of-experts model |
| `analyze.py` | E1–E6 + figures 1–6 |
| `replicate.py` | pre-registered confirmatory H1–H7 on a replication month |
| `irt_grm.py` / `irt_prototype.py` | explanatory graded-response (item-response) analysis |
| `synthetic_agents.py` | fixed-depth agents for identifiability (E6/H7) |
| `diag.py`, `maia_otb_match.py`, `maia_otb_traps.py` | diagnostics / internal probes |

## Data & code availability
Derived datasets are deposited on OSF ([osf.io/ruhy8](https://osf.io/ruhy8); see
`DATA_MANIFEST.md` for files, schemas, sizes, and provenance). The pre-registered replication
protocol is archived at DOI [10.17605/OSF.IO/A6NYK](https://doi.org/10.17605/OSF.IO/A6NYK)
(first registration: [10.17605/OSF.IO/KB4ZQ](https://doi.org/10.17605/OSF.IO/KB4ZQ)).
Raw sources are public: Lichess open database (CC0), Lichess broadcast exports, and
the chess.com Published-Data API.

## Citing
Maharaj, T. ([ORCID 0009-0001-5835-8967](https://orcid.org/0009-0001-5835-8967)) & Regan, K. W.
*Errors reveal the depth of human reasoning.* (in prep).
T. Maharaj formerly published as T. T. Biswas.

## License
Code: _TBD_ (suggest MIT/Apache-2.0). Data: CC-BY (matching the pre-registration).
