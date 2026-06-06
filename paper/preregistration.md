# Pre-registration: Depth of satisficing — out-of-sample replication

**Title:** Errors reveal the depth of human reasoning — pre-registered replication on a fresh month
**Authors:** T. T. Biswas, K. W. Regan
**Status:** Pre-registered replication. Hypotheses, sampling, pipeline, tests, and decision
criteria below are fixed *before* the replication dataset is parsed or analysed.
**Design:** The hypotheses and predicted directions/magnitudes are derived from a prior analysis
of the Lichess **2025-09** month (and the elite OTB/broadcast set). They are now to be confirmed
on an **untouched** month — Lichess **2026-05** (standard, rated) — processed by an identical,
fixed pipeline. This is a pre-registered *replication*, not a blind test of novel hypotheses; we
state this explicitly.

---

## 1. Data & sampling (fixed)
- **Source:** Lichess open database, standard rated games, month **2026-05**; plus the elite
  over-the-board (broadcast) stratum for the individual-profile hypothesis (H6).
- **Sampling:** balanced reservoir sample per (rating band × time class) as in `config.yaml`
  (200-pt bands 1200–2600+; bullet/blitz/rapid/classical), seed **17**; plies 9–120.
- **Target size:** ≥150,000 analysed unique FENs (sufficient power; full month optional).
- **Split:** all model fitting and held-out evaluation are **split by player** (no player in
  both train and test). Broadcast player names are canonicalised to one identity per person.

## 2. Pipeline & fixed parameters (identical to the primary analysis)
- **Engine:** Stockfish 18, `Threads=1`, `Hash=256MB`, **depth D=21, MultiPV K=9**, per-position
  **node cap 2×10⁸** (search stops at D or the cap; capped positions, ~0.1%, are depth-masked).
- **Win probability** from `UCI_ShowWDL`: `P=(W+0.5·Dr)/1000`; regret `δ_{i,d}=max_j P_{j,d}−P_{i,d}`.
- **Maia-3** `maia3-79m` policy as the human-pattern prior.
- **Model:** latent-depth product-of-experts (depth grid {2,…,22}), entropy-regularised, β tied
  globally; depth-of-satisficing posterior `r_d ∝ π_d P(y|d)`, `d̂=Σ d·r_d`.
- **Swing convention (Biswas & Regan 2015):** a move **swings up** if its regret δ *decreases*
  with depth (`sw=Σ_d(δ_{m,d}−δ_{m,D})>0`; potential shows only deep); it **swings down** (a
  trap) if regret *rises* with depth (`sw<0`).

## 3. Confirmatory hypotheses (directional; predicted magnitudes from 2025-09 in brackets)

**PRIMARY**

- **H1 — Depth rises with skill, within time control; absent in bullet (control).**
  Within classical, rapid, and blitz separately, Spearman(`d̂`, rating) on held-out players is
  **positive with 95% CI excluding 0** [≈ +0.43 / +0.53 / +0.43]. In **bullet** the relationship
  is null [|ρ|<0.15, no monotone rise across bands] — a built-in negative control.
  *Test:* per-time-control Spearman + player-clustered bootstrap CIs (1000×).

- **H3 — Inferred depth tracks real thinking time (non-circular).**
  A model fit with the **clock feature removed** yields `d̂` that correlates **positively** with
  observed time-on-move on held-out data in every phase [overall ρ≈+0.39], and the **middlegame
  correlation is ≥ the opening and endgame** [middlegame ≈+0.44].
  *Test:* held-out Spearman by phase (opening ply≤24, middlegame 25–60, endgame >60).

- **H4 — Depth-aware prediction beats Maia-3 selectively (registered interaction).**
  Held-out cross-entropy of the fusion is **lower than state-only Maia-3** overall [ΔNLL≈+0.012,
  CI excludes 0], and the gain is **differential**: the swing-magnitude × time-control
  interaction (gain in classical/high-swing minus blitz/low-swing) is **positive** [≈+0.06 nats]
  with 95% CI excluding 0 and a mixed-effects (player random intercept) interaction coefficient
  **>0, p<0.001**.
  *Test:* three models (Maia-only β=0; search-only α=0; fusion) under one player split;
  player-clustered bootstrap + linear mixed-effects `gain ~ classical*high_swing + (1|player)`.

**SECONDARY**

- **H2 — Within-game/within-player time pressure.** (a) Within players, de-meaned `d̂` rises
  monotonically across think-time quartiles (fast→slow). (b) In OTB games (move-40 control), the
  played-move error in live positions (|eval|<150cp) rises through moves 31–40 and drops after
  move 40 [reproducing Biswas & Regan 2015 in win-probability terms].

- **H5 — Skill information concentrates in traps (swing-down).** Single-decision rating
  predictability (R²) is **higher for swing-down than swing-up** decisions [≈6×: 0.081 vs 0.012].

- **H6 — Individual profiles recover rating; 1-D depth does not.** A 4-axis profile (depth,
  trap-susceptibility, deep-discovery, time-elasticity; depth from an **elo-free** model)
  recovers rating by nested CV with **R² substantially above the depth-only baseline**
  [profile R²≈0.57 vs depth-alone R²≈0].

- **H7 — The instrument recovers planted depth (identifiability).** On synthetic agents that
  satisfice at fixed depths (β_gen=6), recovered depth increases **monotonically** with planted
  depth [group-level Spearman ≈1.0]; recovery is ordinal (per-agent estimates compressed).

## 4. Decision criteria (what counts as a successful replication)
- **Replication is declared successful** if **all three PRIMARY hypotheses (H1, H3, H4)** hold in
  the predicted direction with the stated CI/significance criteria on the 2026-05 data.
- Each SECONDARY hypothesis is reported as **replicated / not replicated** by its directional
  criterion; they are supporting, not gating.
- Magnitudes are **not** required to match the 2025-09 values; only the **directions and
  significance** above are confirmatory. Any deviation is reported.

## 5. Exploratory (explicitly NOT confirmatory)
Won-vs-lost decomposition; named individual-player (e.g., world-champion) profiles and the
"two-effective-dimensions" structure; the complexity-vs-clock decomposition of error; any
cross-domain (Go) transfer. These are reported as exploratory and labelled as such.

## 6. Notes
- Broadcast `WhiteElo`/`BlackElo` are event-time tags averaged per player; used for within-data
  structure, not as live ratings.
- This file is committed to the project git history to timestamp it before the 2026-05 data is
  processed; an external timestamp (OSF / AsPredicted) should be posted in parallel.
