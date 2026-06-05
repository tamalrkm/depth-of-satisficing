# Project conventions
This project uses uv for Python environment management.
- Run Python: `uv run python <script>`
- Run pytest: `uv run pytest`
- Add a package: `uv add <package>`
- Add a dev-only package: `uv add --dev <package>`
- Never use `pip install`, `conda install`, or manual venv activation.
Commit pyproject.toml and uv.lock to git.

## Machine
- 2x AMD EPYC 9335 -> 64 physical cores, SMT on -> 128 logical CPUs.
  NUMA node0 = cores 0-31,64-95; node1 = cores 32-63,96-127.
- NVIDIA H200 (~143 GB VRAM); torch on the **cu130** wheel index (driver CUDA 13.1).
- The GPU is oversized for this workload (Maia-3 + a tiny model): it is **never** the
  bottleneck. **Stockfish is CPU-bound** and the engine stage sets all wall-clock time.
- Stockfish 18 binary is built/supplied by the user at `config.yaml: engine.path`. Do not
  download or build it.

---

# What this project is
We measure the **depth of human reasoning** ("depth of satisficing") from the structure of
human *mistakes* in chess, on current online + elite-OTB games. A player is modelled as a
latent distribution over engine search depths; the move played is a product-of-experts
fusion of (a) Maia-3's human-pattern policy and (b) a soft-max over engine regret at the
depth searched. Full spec, equations, and the results plan are in `paper/manuscript.tex` --
read it before changing the model or analysis.

# Critical-path task -- DO THIS FIRST
`src/maia_features.py` is written against an **assumed** Maia-3 API. Before anything else:
1. In `uv run python`, import `maia3` and find the real call returning a **full policy
   distribution** (dict `uci_move -> prob`) conditioned on the side-to-move Elo, on GPU.
2. Rewrite `maia_features.py` to use it, batched. This gates `build_dataset.py`, `train.py`,
   and all results. If no policy head is exposed, use logits->softmax over the move head;
   the UCI single-best-move path is for plumbing tests only.

# Pipeline status & tasks (in order)
Implemented (verify, then extend): `engine_analysis.py`, `model.py`, `train.py`,
`synthetic_agents.py`.
Scaffold (implement):
1. `parse_games.py` -- stream a Lichess `.pgn.zst` dump (`zstandard`); balanced reservoir
   sample per (Elo-bin x time-class) from `config.yaml`; extract FEN, played move, `%clk`
   (-> time spent), `%eval`, ratings, time control -> `data/positions.parquet`.
2. `select_positions.py` -- **pass-through by default** (compute is free: keep all non-book
   decisions); swing-candidate filter behind a `--filter` flag.
3. `run_engine.py` (NEW) -- parallel Stockfish driver for the 2x EPYC box:
   - **`Threads=1` per process, many processes** (Lazy SMP scales sublinearly; we want
     aggregate throughput). Default `workers=64` (physical cores); allow 96/128 but benchmark.
   - **NUMA-bind each worker** cpu+mem to one node (`numactl --cpunodebind=N --membind=N`
     or `os.sched_setaffinity`) per the node layout above.
   - **Persistent engine per worker**: `popen` once, reuse `analysis()` per FEN; never
     respawn per position.
   - **FEN-dedup** before queueing; shard output to `data/shards/`, then concat.
   - Use SF18 shared-memory NNUE; print a throughput estimate after the first 200 positions.
4. `build_dataset.py` -- join stages; padded tensors (`delta [M,D]` regret in win-prob units
   on the depth grid, `logq [M]`, `move_mask`, `context`, `y`); per-move swing
   `sw(m)=sum_d(delta_{m,d}-delta_{m,D})`; `torch.save` the dict `train.py` expects.
5. `analyze.py` -- E1-E6 + the six figures (manuscript Results & figure list).
6. `fetch_broadcasts.py` (NEW) -- list/filter Lichess broadcasts (`GET /api/broadcast`,
   `/top`), export tournament PGNs (`GET /api/broadcast/{id}.pgn`), detect `%clk`, tag rows
   as the elite-OTB stratum.

# Definitions to keep consistent (from the paper)
- Win probability per move per depth from Stockfish `UCI_ShowWDL`:
  `winprob = (W + 0.5*D)/1000`, side-to-move POV. **No logistic/centipawn scaling.**
- Regret `delta_{i,d} = max_j winprob_{j,d} - winprob_{i,d} >= 0`.
- Candidate set = union of top-K across **all** depths; played move always included
  (re-query alone if it leaves top-K).
- Depth of satisficing = posterior `r_d ~ pi_d * P(y|d)`, reported as `E[d]` with credible
  intervals; never the old curve-intersection.

# First session goal (smoke test)
`parse` ~500 games -> `select` (pass-through) -> `run_engine` on a few hundred positions ->
`maia_features` -> `build_dataset` -> `train` 2 epochs -> confirm `model.depth_of_satisficing`
returns sane values. Add a `just smoke` target that does exactly this.

# Style
Small, composable CLIs driven by `config.yaml`; Parquet between stages; log output paths at
the end of each stage. No heavy frameworks. Keep the model math identical to the paper.
