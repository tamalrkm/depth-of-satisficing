# Pre-registration (v2): corrected, prospective confirmation of the depth of satisficing

**Title:** Errors reveal the depth of human reasoning — corrected hypotheses, prospective replication
**Authors:** Tamal Maharaj, Kenneth W. Regan
**Status:** Confirmatory replication. Hypotheses, sampling, pipeline, tests, and decision criteria
below are fixed *before* any confirmation dataset is parsed, sampled, engine-analysed, or
inspected at any granularity.
**Date of registration:** _(stamp on OSF posting; precedes processing of every dataset named in §1)_

---

## 0. Relationship to the prior pre-registration (integrity statement)
This registration **does not alter or supersede** our earlier pre-registration (the Lichess
**2026-05** replication; OSF DOI `<existing>`), which stands unchanged as the honest record of our
first replication. On 2026-05 the non-circular results (H3, H4) held, but **(i)** the depth↑rating
relationship held only in the slow time controls and **failed in blitz**, and **(ii)** the
predicted *direction* of the single-decision rating-information effect (H5) was **mis-stated** — a
swing-label convention had been inverted and was corrected in code; re-analysis of both 2025-09 and
2026-05 under the corrected convention shows the effect runs the **opposite** way (swing-*up*, not
swing-down). The present document freezes the **corrected** hypotheses for an **independent,
prospective** test on data not yet processed (and, for the primary set, not yet in existence as a
released dataset). The corrected hypotheses are exploratory with respect to 2026-05 and 2025-09 and
**confirmatory** only with respect to the datasets in §1.

---

## 1. Data & sampling (fixed)
Confirmation datasets, processed by an **identical, frozen pipeline** (§2). Each is processed
**once** and reported regardless of outcome; the **primary confirmatory decision (§4) rests on the
prospective dataset D1**. D2/D3 are robustness/extension.

- **D1 — PRIMARY (prospective):** Lichess **June 2026** standard-rated open-database dump. At
  registration this dump **does not yet exist** (Lichess publishes a month ~early in the following
  month; June games are still being played), so it cannot have been observed.
- **D2 — robustness:** Lichess **April 2026** standard-rated dump. This file is **present on the
  authors' disk but has never been opened, parsed, sampled, engine-analysed, or inspected** (see
  §1a). Used as an independent within-source robustness check.
- **D3 — cross-pool extension:** a **chess.com** sample harvested via the public Published-Data API
  (titled players via `/pub/titled/{title}` and top leaderboards, then each player's monthly game
  archives). **Not yet collected** at registration. chess.com is a **separate rating pool** from
  Lichess and from OTB/FIDE; D3 exists to test cross-pool measurement (§3, H-INV), not the
  within-pool primaries.
- **Sampling:** balanced reservoir sample per (rating band × time class) as in `config.yaml`
  (200-pt bands; bullet/blitz/rapid/classical), seed **17**, plies 9–120; plus the elite
  over-the-board (broadcast) stratum for individual-profile analyses. For D3, additionally retain
  **all** decisions of harvested titled/leaderboard players (player-dense, for H-INV).
- **Target size:** ≥150,000 analysed unique FENs per Lichess dataset.
- **Split:** all model fitting and held-out evaluation are **split by player** (no player in both
  train and test). Elite/broadcast names are canonicalised to one identity per person.
- **Pool-specific ratings:** Lichess and chess.com ratings are separate per-time-control systems and
  are **not comparable across controls or sites**. Every rating analysis is run **within a single
  pool** or with rating **z-scored within pool**; ratings are never pooled across controls/sites.

### 1a. Prior knowledge of / access to the data (secondary-data disclosure)
All sources are existing public datasets not collected by the authors (public usernames/player
names only; no intervention). Access status at registration:
- **D1 (Lichess 2026-06):** not yet released as a dump; not downloaded; outcome cannot have been seen.
- **D2 (Lichess 2026-04):** the `.pgn.zst` resides on disk, downloaded but **unopened and
  unprocessed**; no sampling, engine analysis, or inspection has occurred.
- **D3 (chess.com):** not yet harvested.
- The hypotheses/magnitudes were derived from the **2025-09** primary analysis and the **2026-05**
  replication (both fully analysed). No D1/D2/D3 outcome has been observed. After registration each
  dataset is processed once by the frozen pipeline and each test run once.

---

## 2. Pipeline & fixed parameters (identical to the primary analysis)
- **Engine:** Stockfish 18 (net `nn-71d6d32cb962`), `Threads=1`, `Hash=256MB`, **depth D=21,
  MultiPV K=9**, per-position **node cap 2×10⁸**; capped positions depth-masked.
- **Win probability** from `UCI_ShowWDL`: `P=(W+0.5·Dr)/1000`; regret `δ_{i,d}=max_j P_{j,d}−P_{i,d}`.
- **Maia-3** `maia3-79m` as the human-pattern prior.
- **Model:** latent-depth product-of-experts (depth grid {2,…,22}), entropy-regularised, β tied
  globally; depth of satisficing `r_d ∝ π_d P(y|d)`, `d̂=Σ d·r_d`.
- **Swing convention (corrected, Biswas & Regan 2015):** a move **swings up** if its regret
  *decreases* with depth (`sw=Σ_d(δ_{m,d}−δ_{m,D})>0`; worth shows only deep); **down** (a trap) if
  regret *rises* with depth (`sw<0`). Applied to the played move.
- **Item-response layer:** item difficulty `b_j` = critical depth (deepest grid depth at which the
  apparent-best move differs from the full-depth best move, in plies); ordered response = played
  move's full-depth regret binned best (≤.02) / inaccuracy (≤.05) / mistake (≤.10) / blunder (>.10).

## 3. Confirmatory hypotheses (directional; magnitudes from 2025-09 + 2026-05 in brackets)

**PRIMARY** (decision set; tested on D1, reported on D2/D3 where applicable)

- **H1 — Depth rises with skill in SLOW play; null in fast play.** Within **classical** and
  **rapid** separately, Spearman(`d̂`, rating) on held-out players is **positive, 95% CI excluding 0**
  [≈ +0.47 / +0.54]. In **blitz and bullet** the relationship is **null** [|ρ|<0.15] — both fast
  controls are negative controls (2026-05 showed blitz null; depth needs time to deploy).
  *Test:* per-control Spearman + player-clustered bootstrap (1000×).

- **H3 — Inferred depth tracks real thinking time (non-circular).** A model fit with the **clock
  feature removed** yields `d̂` correlating **positively** with observed think-time on held-out
  data, **strongest in the middlegame** [overall ≈+0.30; middlegame ≥ opening, endgame].
  *Test:* held-out Spearman by phase (opening ≤24, middlegame 25–60, endgame >60).

- **H4 — Depth-aware prediction beats Maia-3, registered interaction.** Held-out cross-entropy of
  the fusion is **lower than state-only Maia-3** [ΔNLL>0, CI excludes 0], and the
  swing-magnitude × time-control interaction (classical/high-swing minus blitz/low-swing) is
  **positive** [≈+0.05 nats, CI excludes 0] with a mixed-effects interaction coefficient
  **>0, p<0.001**.
  *Test:* three models (Maia-only β=0; search-only α=0; fusion) under one player split;
  player-clustered bootstrap + mixed-effects `gain ~ classical*high_swing + (1|player)`.

**SECONDARY** (reported; not gating)

- **H5 — Skill information concentrates in DEEP-DISCOVERY (swing-up) decisions [sign corrected].**
  Single-decision rating predictability (R², pool-normalised rating, nested-CV) is **higher for
  swing-up than swing-down** decisions [≈2–4×]. *(This reverses the direction in the prior
  registration; see §0.)*

- **H-IRT — Swing is item discrimination [new].** In an explanatory proportional-odds graded-response
  model with predictors {ability (within-pool rating z), item difficulty `b_j`, swing,
  ability×swing}: error severity **decreases with ability** (β<0), **increases with difficulty**
  (β>0), and the **ability×swing interaction is negative** (β<0) — ability discriminates more in
  high-swing items. All with p<0.01 [2025-09: −0.13 / +0.06 / −0.08].

- **H2 — Within-player time pressure.** Within players, de-meaned `d̂` rises across think-time
  quartiles (fast→slow).

- **H6 — Individual profiles recover within-pool rating; 1-D depth does not.** A 4-axis profile
  (depth from an elo-free model, trap-susceptibility, deep-discovery rate, time-elasticity) recovers
  **within-pool** rating by nested CV with **R² above the depth-only baseline** [profile ≈0.10 vs
  depth-alone ≈0, within-pool].

- **H7 — The instrument recovers planted depth.** On synthetic agents at fixed depths, recovered
  depth increases monotonically with planted depth (group Spearman ≈1.0; ordinal).

- **H-INV — Pool-invariant ability [cross-pool; D3-contingent].** If a **player-dense, multi-pool**
  sample is obtained (chess.com titled/leaderboard players with ≥N decisions per pool, split-half
  reliability of person ability θ > 0.6), then θ estimated within one pool **predicts** move-quality
  out-of-pool, and within-pool θ rank-correlates across pools on the shared ply scale. **Confirmatory
  only if the reliability/density preconditions are met; otherwise reported as exploratory.**

## 4. Decision criteria
- **Replication declared successful** if **all three PRIMARY hypotheses (H1, H3, H4)** hold in the
  predicted direction with the stated CI/significance on the **primary prospective dataset D1**.
- Each SECONDARY hypothesis is reported as replicated / not by its directional criterion (supporting,
  not gating). D2 and D3 are reported as robustness/extension.
- Magnitudes need not match; only **directions and significance** above are confirmatory. Any
  deviation is reported.

## 5. Data exclusion rules (fixed)
- Plies 9–120; parse-error / no-legal-candidate positions dropped.
- Node-capped positions retained, unreached grid depths masked; capped fraction reported.
- Decisions lacking a Maia-3 policy or any context feature excluded.
- H3/H2: decisions with missing/non-positive observed think-time excluded; increment-contaminated
  clock deltas filtered by a sign/sanity check.
- H6: players with <100 decisions excluded; broadcast Elo = per-player event-time average.
- Time/site classes from base+increment; chess.com mapped to the same bullet/blitz/rapid/classical
  buckets.

## 6. Contingencies (if–then)
- **If** D1 (June 2026) is unavailable or materially malformed at processing time, **then** the
  primary confirmatory test is run on the next untouched prospective Lichess month, and the
  substitution is reported; D2 is **not** promoted to primary (it was on disk at registration).
- **If** a mixed-effects model (H4) fails to converge, **then** the player-clustered bootstrap is the
  primary inference (mixed-effects reported as secondary or omitted with reason).
- **If** a rating band within a control has <50 held-out decisions, **then** it is omitted from the
  per-band curve; the Spearman uses all available decisions; omission reported.
- **If** the node cap is hit by >2% of positions, **then** the capped fraction is reported and the
  primary tests re-run excluding capped positions.
- **If** a month lacks per-move clock (`%clk`), **then** clock-dependent parts (H2, H3) are "not
  testable" and clock-independent H1/H4/H-IRT proceed.
- **If** the chess.com harvest (D3) does not reach the density/reliability preconditions, **then**
  H-INV is reported as exploratory, not confirmatory.

## 7. Anticipated deviations
- Sample size and per-cell coverage differ by month/site; actual counts reported.
- The broadcast/elite stratum differs by month; coverage reported, not adjusted.
- Engine (SF18, net `nn-71d6d32cb962`) and Maia-3 (`maia3-79m`) versions pinned; any unavoidable
  change reported.
- Pipeline bug-fixes not touching hypotheses/variables/tests are permitted and logged; any change
  affecting a test is reported as a deviation.

## 8. Exploratory (explicitly NOT confirmatory)
Won-vs-lost decomposition; named individual-player profiles; complexity-vs-clock decomposition;
cross-domain (Go) transfer; full cross-pool/cross-site IRT person-invariance and a depth-anchored
intrinsic-performance-rating conversion beyond the registered H-INV preconditions; computerised
adaptive-testing on the depth scale. Reported as exploratory and labelled as such.

## 9. Notes
- This file is committed to the project git history to timestamp it; an external OSF timestamp is
  posted in parallel, **before any dataset in §1 is processed**.
- The prior 2026-05 registration remains posted and unchanged (§0).
