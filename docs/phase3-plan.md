# Phase 3 plan: graph features and a GNN against the frozen baseline

*Written 2026-10-29 and rewritten the same day after review against the code and the machine (see "Revision notes" at the end). Based on `docs/decisions.md` (locked), `reports/phase2-results.md`, `docs/phase2-plan.md` and `src/`. Planning only; nothing implemented yet.*

## Goal

Answer the project's headline question: **does information about the past transaction network beat `lgbm/AB` on the same test transactions?** It is answered twice, with hand-made graph features (`lgbm/ABC`) and with a GNN (`gnn/AB`), plus an `mlp/AB` control that separates "graph" from "different learner". Phase 3 is done when the new models are scored once on test with a **paired bootstrap against the frozen `lgbm/AB` scores**, iid and attempt-cluster, plus recall by pattern with the focus on UNLISTED.

## Where Phase 2 left us

- **Baseline:** `lgbm/AB`, test PR-AUC 0.292 [0.261, 0.328] (cluster [0.232, 0.351]), F1 0.351, P/R@500/day 0.34 / 0.35. Val threshold 0.1652. Scores frozen in `scores` (`lgbm`, `AB`), model in `data/models/lgbm_AB.txt`.
- **What it misses:** UNLISTED laundering, 9 of 371 on test (2.4%). 99.9% of what it flags is ACH, and 38.5% of UNLISTED is not ACH. Listed schemes are already at 0.30–0.77 recall.
- **What drives it:** the sender's recent ACH share (43% of gain) and the "no history" effect (2.3% of rows hold 53% of laundering).
- **Closed in Phase 2:** pair history belongs to set C, not B. The extra bootstrap is attempt-cluster only (day-block ruled out with 2 test days).

## Fix first (done in T0–T1)

1. **Phase 2 Checkpoint C is open.** Its rebuild (`train.py all`) retrains and overwrites `lgbm/AB`. T0 backs it up first and compares.
2. **Phase 2 report files get overwritten by default.** `train.py` always writes `reports/phase2-tuning.csv`, and `evaluate.py --split val` rewrites `val-metrics.csv` and `val-recall-by-pattern.csv` with every model in `scores`. T1 adds `--phase 3`, which routes every output to `reports/phase3-*`.
3. **`evaluate.py` can't score new models on test.** It refuses while `reports/test-metrics.csv` exists, and `PAIRS` is hard-coded. T1 adds a per-model "scored once" guard.
4. **The bootstrap is too slow.** The Phase 2 test run took 67 min (17:05–18:12), and Phase 3 adds 3–4 models and more pairs. T1 implements the speed-up named in the `ponytail:` note in `metrics.py`.
5. **No deep-learning stack.** torch and PyTorch Geometric aren't installed. Python 3.13 on Windows needs torch ≥ 2.6 and PyG ≥ 2.6; the pure-Python PyG install is enough for the layers below. T4.

## Decisions

### Resolved with triet (2026-10-29, "keep the default")

- **Set C is kept** and comes before the GNN.
- **Hardware:** CPU first. The machine has an **NVIDIA GeForce MX550 with 2 GB VRAM** (checked with `nvidia-smi`) next to Intel Iris Xe. The T4 timing slice times both and the faster one is used. Colab (the `CLAUDE.md` fallback) only if both miss the T4 budget.
- **`lgbm/ABG` is built only if `gnn/AB` beats `lgbm/AB` on val**, with the stacking fix below. It gets expanded into full tasks only if that condition is met.

### Set C: hand-made graph features (first comparison)

Why first:
- **Cheap:** DuckDB windows like set B, minutes to build, LightGBM in about 5 min.
- **Answers the headline question** with the Phase 2 machinery.
- **A fair yardstick:** if `gnn/AB` beats `lgbm/AB` but not `lgbm/ABC`, the gain is "graph information", not "the GNN".
- **A fallback** if the GNN fails, and **explainable** for Phase 4 (SHAP on LightGBM).

Columns (all strictly past, windows ≤ 3d, the `RANGE … INTERVAL 1 MINUTE PRECEDING` rule):
- **Fan-in / fan-out:** distinct counterparties of the sender (past sends) and of the receiver (past receipts), 1h / 24h / 3d. `COUNT(DISTINCT …) OVER (… RANGE …)` works in DuckDB 2.0 (toy check: 0, 1, 1, 2), so this is the set-B pattern.
- **Pass-through:** for the sender, its past *receipts* in 1h / 24h (count, USD) and the ratio of its past 24h USD out to USD in; for the receiver, its past *sends* likewise. This is the other side of each account, which B never sees.
- **Pair history:** past u → v count and minutes since the last u → v (capped at 3d); reverse pair v → u in 3d (2-cycle).
- **Short cycles:** does a **time-ordered** path v → w → u exist with `ts − 3d ≤ ts(v→w) < ts(w→u) < ts`? Stored as 0/1 plus count. **The join starts from the sender u's past receipts w → u** (in-degree at most about 1,084 over the whole period) and looks up an earlier v → w. Starting from the receiver's past sends would expand hubs (about 60k sends in 3 days) into billions of paths. If it still takes over 10 min, the window drops to 24h.
- **Not in C:** global centrality, PageRank, anything over the whole graph. Those grow with absolute time and are slow.

### GNN: hourly past-only snapshots, edge classification

- **Task:** edge (transaction) classification, as in the IBM paper (Egressy et al., *Provably Powerful GNNs for Directed Multigraphs*, 2024), so scores land per `tx_id` and the paired bootstrap works unchanged.
- **Structural leakage rule:** a transaction in hour `h` is scored from a snapshot holding only edges with `ts` in **[h − W, h)**, so every edge in it is strictly earlier than every target. Edges earlier in the same hour aren't seen; within-hour freshness comes from the target's own set-B features in the head.
- **Why snapshots:** with edges sorted by `ts`, a snapshot is one contiguous slice (`searchsorted`), free to build and checkable with one assert. Per-edge temporal sampling needs a custom sampler and 5M subgraphs. About 240 hourly snapshots; tail separately.
- **Window:** **W = 24h** by default (about 500k edges per snapshot); **3d** tried once on val if memory allows. Both obey "≤ 3d, nothing grows with absolute time".
- **Model:**
  - **Architecture:** GINe-style (GIN with edge features in the message), with **reverse message passing** (separate weights for in- and out-edges); 2 layers, hidden 64.
  - **Node input:** a constant plus `log1p` in/out degree within the snapshot. No account IDs. Hubs have tens of thousands of edges per window, hence the log.
  - **Edge input:** set A, encoded once.
- **Head:** for a target u → v, an MLP on `[h_u, h_v, A + B of the target]`, so `gnn/AB` sees what `lgbm/AB` sees plus learned graph context.
- **Training:** train hours only; one snapshot is one step; the loss is on that hour's targets (about 22k rows, about 17 positives on average) with positive weighting; Adam; early stopping on **val PR-AUC**; 3 seeds, picked on val. Thresholds from val.
- **Hard bans** (Phase 2's plus GNN-specific):
  - no learnable per-account embeddings (they memorize dirty accounts across the cut);
  - no labels or pattern data as inputs (no label propagation);
  - no `tx_id`, `split` or raw IDs.
- **Upgrades named, not planned:** port numbering and ego IDs from the IBM paper, PNA instead of GIN, per-edge temporal sampling. Add only if `gnn/AB` is close to `lgbm/ABC` and there's time.
- **Expectation, not a target:** the IBM paper's HI-Small minority F1 for GINe with its extensions is high (about 0.57, written from memory). **Check the exact figure and its time rule in the paper before quoting it.** Our strict past-only setup should come in lower.

### `mlp/AB`: the control

The GNN's head trained alone: same MLP, same A + B input, same loop, with the graph embeddings left out (one flag in `gnn.py`). `gnn/AB − mlp/AB` is the effect of the learned graph context alone. `mlp/AB − lgbm/AB` is the cost or gain of an MLP instead of trees.

### `lgbm/ABG` (conditional; expanded only if opened)

LightGBM on A + B + G, where G = the GNN's `[h_u, h_v]` for each transaction. It asks whether the GNN's graph signal survives inside the tree model.
- **Stacking trap:** a GNN trained on train labels produces embeddings for train rows that are fitted to those labels. A LightGBM trained on them would overtrust them, and the score would be inflated.
- **Fix:** time-split stacking. Train a separate GNN on Sep 1–4 only and use it for the embeddings of every row from Sep 5 on. Fit `lgbm/ABG` on Sep 5–6 (rows its GNN never saw labels for), early-stop on val, and score val, test and tail with the same GNN. It is weaker than full training (2 training days) and that is stated.
- **Explainability is limited:** SHAP can rank embedding dimensions, but an investigator can't read them. A and B stay readable.

### Models compared on test (new names; `lgbm/AB` untouched)

| Name | What | Question it answers |
|---|---|---|
| `lgbm/AB` | frozen baseline | — |
| `lgbm/ABC` | LightGBM, Phase 2 grid, A + B + C | Do hand-made graph features help? |
| `mlp/AB` | GNN head alone, A + B | Control for learner (MLP vs trees) |
| `gnn/AB` | GNN on snapshots, head on A + B | Does learned graph context help? |
| `lgbm/ABG` *(conditional)* | LightGBM on A + B + GNN embeddings, time-split stacked | Does the GNN's signal survive inside the explainable model? |

Paired test comparisons: `lgbm/ABC − lgbm/AB`, `gnn/AB − lgbm/AB`, `gnn/AB − lgbm/ABC`, `gnn/AB − mlp/AB` (plus `lgbm/ABG − lgbm/AB` if built). Significant = paired 95% CI excludes 0, under iid **and** attempt-cluster resampling. `lgbm/AB`'s side of each pair uses its **existing** test scores; it is never rescored.

### Outputs

- `features` gets the C columns (`SET_C` in `features.py`); `FEATURE_SETS` gets `"ABC"`.
- `src/gnn.py`: export, snapshots, model, training, and scoring into `scores` as `gnn/AB` and `mlp/AB`; checkpoints in `data/models/gnn_AB.pt` and `mlp_AB.pt`.
- Reports: `reports/phase3-*.csv` (val, test, tail, `phase3-tuning.csv`) and `reports/phase3-results.md`. **Phase 2 report files are never touched:** after each task, `git diff reports/` shows only `phase3-*` files.

## Dependency graph

```
T0 back up frozen baseline (done), close Phase 2 Checkpoint C
 └ T1 eval harness: --phase 3 routing, fast bootstrap, per-model test guard
    ├ T2 set C + leakage check ─ T3 lgbm/ABC on val ─────────────────────────────┐
    └ T4 GNN export, snapshots, timing ─ T5 gnn/AB + mlp/AB ─ T6 GNN grid + val ─┴ Checkpoint B (triet) ─ T7 test once
                                                               └ (lgbm/ABG only if gnn/AB > lgbm/AB on val)
```

The laptop runs one heavy job at a time, so C goes first (cheap, de-risks the comparison) and then the GNN.

---

## Task 0: Back up the frozen baseline and close Phase 2 Checkpoint C (XS) — DONE 2026-10-29

**Description:** Copy the `lgbm/AB` rows of `scores` to a table `scores_frozen_lgbm_AB`, and `lgbm_AB.txt` to `lgbm_AB.frozen.txt`. **No rebuild or retrain** (triet, 2026-10-29): every model was trained with the current code, so a rebuild would only reproduce what exists, at ~10 min and with a risk to the baseline. The full rebuild moves to this phase's Checkpoint C.

**Acceptance criteria:**
- [x] Frozen copies exist: 5,078,345 rows, one per `tx_id`, max abs diff 0 against the live scores; model file byte-identical
- [x] Phase 2 Checkpoint C's rebuild box ticked as waived in `phase2-plan.md`
- [x] triet has reviewed the Phase 2 results (`reports/phase2-results.md`, approved 2026-10-29); the last Phase 2 Checkpoint C box is ticked

**Verification:** one SQL compare of `scores` against `scores_frozen_lgbm_AB` over all 5M `tx_id`s; `cmp` of the model files.
**Dependencies:** None
**Files:** none (DuckDB table, model file copy; ticks in `docs/phase2-plan.md`)

**Outcome (2026-10-29):** backup done and verified (rows by split: train 3,248,921, val 965,524, test 862,792, tail 1,108). triet approved the Phase 2 results; Phase 2 is closed.

## Task 1: Evaluation harness for Phase 3 (S) — DONE 2026-10-29

**Description:**
- **Fast bootstrap:** sort scores once; per resample, AP from weighted cumulative sums over the bootstrap counts (`bincount(indices)`), which is mathematically the same as AP on the resampled rows. It must work for iid and cluster resampling.
- **Phase routing:** `evaluate.py` and `train.py` both get `--phase 3`, which routes every output (val metrics, recall by pattern, tuning log) to `reports/phase3-*`.
- **Test guard:** `--final` takes `--models`, refuses any model already in `phase3-test-metrics.csv`, and pairs models as listed under Decisions.

**Acceptance criteria:**
- [x] `metrics.py` self-check passes; the fast bootstrap gives the same CIs as the old one on the Phase 2 val scores (same seed, within 1e-6) for iid and cluster, and is ≥ 10× faster *(measured 8×, see Outcome)*
- [x] `evaluate.py --split test --final --phase 3 --models lgbm/ABC` refuses `lgbm/AB` as a new model (it is only a pairing partner) and refuses a second run of the same model
- [x] Phase 2 report files byte-identical before and after (`git diff reports/` shows only new `phase3-*` files)

**Verification:** `python src/metrics.py`; a dry run on val with `--phase 3`.
**Dependencies:** T0
**Files:** `src/metrics.py`, `src/evaluate.py`, `src/train.py`
**Outcome:**- **Fast bootstrap:** for PR-AUC, `_bootstrap` sorts each score once and turns every resample into per-row draw counts (for clusters `bincount(pick)[codes]`, the same random draws as before). AP is computed only at score groups holding a positive. It is exact: max difference 1e-16 against sklearn on the resampled rows (iid and cluster), and the 1,000× val CI of `lgbm/AB` reproduces Phase 2 to the digit (0.3926 [0.3596, 0.4255]). Speed: 23 ms (iid) / 28 ms (cluster) per resample at 965k rows, against 187 / 226 ms, so **8×, short of the 10× target**. The rest is drawing and counting 965k indices plus one gather and cumulative sum, accepted. The self-check now asserts fast = slow, ties included.- **`evaluate.py`:** `--phase 3` writes `reports/phase3-*` and always adds cluster CIs, pairs and recall CIs. `--models` limits which models are evaluated. A phase 3 `--final` run needs `--models`, refuses any model already in Phase 2's test file (`lgbm/AB` is a pairing partner only), and refuses a second run while `phase3-{split}-metrics.csv` exists. Pairs are `PAIRS[3]`, used when both models have scores.- **`train.py`:** `--phase N` routes the tuning log to `reports/phaseN-tuning.csv`.- **Checks:** all guards exercised. A val dry run (`lgbm/AB`, `lgbm/A`, 200 resamples, about 29 s) wrote only `phase3-val-*`; md5 of all 14 Phase 2 report files unchanged; dry-run files deleted afterwards.

## Task 2: Hand-made graph features, set C (M, leakage risk) — DONE 2026-10-29

**Description:** Add `SET_C` to `features.py` as DuckDB window and join queries (list under Decisions), with frames ending 1 minute before the row. Extend the brute-force check (`features.py --check`) to C. Two leaky variants must fail: a frame ending at `CURRENT ROW`, and a 3-cycle that ignores time order (`ts(w→u)` allowed after `ts`).

**Acceptance criteria:**
- [x] C columns for all 5,078,345 rows, no unexpected NULLs; build < 15 min at `memory_limit = 3GB`
- [x] Brute-force check: 0 mismatches on the sampled rows (hubs, same-minute ties, Sep 1, split edges, laundering); both leaky variants caught
- [x] Drift check (share of test accounts above the train max) < 1% per column; single-feature val PR-AUC < 0.5 for every C column

**Verification:** `python src/features.py`, then `python src/features.py --check`.
**Dependencies:** T1
**Files:** `src/features.py`
**Outcome:**- **Columns:** 20 C columns (`SET_C` in `features.py`): `out_/in_cpty_{1h,24h,3d}`; `out_recv_{cnt,usd}_{1h,24h}`, `out_pass_ratio_24h`, `in_send_{cnt,usd}_{1h,24h}`, `in_pass_ratio_24h`; `pair_cnt_3d`, `pair_rev_cnt_3d`, `pair_min_since_last`; `cyc3_cnt_3d`. Pass-through comes from one 10M-row stream (each transaction as a send and a receipt). Pairs use one window per unordered pair with a direction flag. 3-cycles join from the sender's past receipts (62.8M candidates), with an ASOF match for an earlier v → w.- **Build:** `features.py` builds A + B + C in **4 min 16 s**. C runs as one temp table per step: as a single query it ran out of memory, and so did the stream window at 3 GB with all threads. **`features.py` now connects with `memory_limit = 1.5GB`, `threads = 4`.**- **Checks:** no unexpected NULLs; distinct counts ≤ totals and nested windows; drift ≤ 0.001% of test accounts. `--check` (2.5 min, now 277 rows including 40 CYCLE-scheme rows): **0 mismatches for B and C**; leaky variants caught: B frame 7,465 cells, B lag 144, C frame 2,389 (18 columns), C cycle-order 5 (`cyc3_cnt_3d`). Single-feature val PR-AUC: highest C column `pair_rev_cnt_3d` 0.021, all far below 0.07.- **Early signal (all splits):** 16% of laundering goes to a receiver the sender paid in the past 3d (licit 76%). 4.7% goes to someone who paid the sender back (licit 0.05%). Only 212 transactions close a 3-cycle, and 72 of them are laundering.

## Task 3: `lgbm/ABC` on val (S) — DONE 2026-10-29

**Description:** Add `"ABC"` to `FEATURE_SETS`, train with the Phase 2 grid via `train.py lgbm ABC --phase 3` (new name; never `all`), and evaluate on val, including the paired val difference against `lgbm/AB`.

**Acceptance criteria:**
- [x] `lgbm/ABC` in `scores`, model saved as `lgbm_ABC.txt`, grid in `reports/phase3-tuning.csv`
- [x] `lgbm/AB` rows equal the frozen copy; Phase 2 report files unchanged
- [x] Val table: PR-AUC, F1, P/R@500, recall by pattern incl. UNLISTED and non-ACH UNLISTED, gain importance of the C columns

**Verification:** `python src/evaluate.py --split val --phase 3 --boot 200`.
**Dependencies:** T2
**Files:** `src/train.py`, `src/evaluate.py`
**Outcome (val, 1,000 resamples):** `lgbm/ABC` PR-AUC **0.444** [0.413, 0.474] (cluster [0.381, 0.499]) vs `lgbm/AB` 0.393. **Paired +0.051 [0.026, 0.075], cluster [0.025, 0.079]: significant.** At the operating point the gain is small (F1 0.479 vs 0.466, P/R@500 0.456 / 0.440 vs 0.446 / 0.431), and per-pattern recall is mixed (FAN-IN 0.80 vs 0.75 up; BIPARTITE 0.35 vs 0.49, STACK 0.33 vs 0.49 down). **UNLISTED 2/379, non-ACH UNLISTED 0/146**; 100% of flagged rows are ACH. Inside ACH, PR-AUC is 0.517 vs 0.459. Gain importance: `in_cpty_3d` (fan-in) 37%, `out_ach_share_24h` 22%, `out_recv_*` (pass-through) 16%. No single feature above 0.023. `lgbm/AB` unchanged; Phase 2 files byte-identical.Changes made on the way:- **Model choice:** the selection rule picked a **1-tree** run (val 0.457, 63 coarse scores). With triet's OK, `train.py` now ignores runs with fewer than `MIN_TREES = 10` trees. It doesn't change the Phase 2 picks (102 and 237 trees). The chosen ABC run is 15 leaves, lr 0.05, min_child 500, 160 trees.- **Memory:** `train.py` loads train and val separately and scores split by split, with categories fixed from the whole table (tail has only ACH, so per-split categories would have shifted codes). DuckDB is capped at **700MB, 2 threads**: the Windows commit limit (RAM + 4 GB page file) left only ~2.9 GB for the process, so higher caps failed OS allocations.

## Checkpoint A (after T0–T3)
- [x] Set C rebuilds in one command and passes its leakage check
- [x] `lgbm/ABC` vs `lgbm/AB` on val written down; frozen baseline intact
- [ ] Quick look with triet: is C worth keeping before the GNN starts

---

## Task 4: GNN data, snapshots and a timing slice (M, highest risk)

**Description:**
- **Install** torch and PyG, pinned in `requirements.txt` (CUDA wheel if the MX550 wins the bench, CPU otherwise).
- **Export:** `gnn.py export` writes edges sorted by `ts` to `data/gnn/edges.npz`: source/target account index, encoded A features, `tx_id`, hour index, split, label.
- **Snapshots:** a function returns the edge slice for `[h − W, h)` via `searchsorted`.
- **Timing slice:** forward and backward of the planned model on 5 train snapshots, on CPU and on the MX550, logging seconds and peak RAM/VRAM.

**Acceptance criteria:**
- [ ] Export round-trips: 5,078,345 rows, `tx_id` order matches `ts` order, no label or pattern columns among model inputs
- [ ] Leakage assert on every snapshot: `max(snapshot ts) < hour start ≤ min(target ts)`; a deliberately leaky snapshot (`[h − W, h]`, including the hour) fails it
- [ ] An epoch over 144 train hours projected < 20 min, with peak RAM < 4 GB (VRAM < 1.8 GB) at W = 24h. **If not:** compact active-node snapshots, hidden 32, or W = 6h, recorded; if still not, stop and raise Colab with triet

**Verification:** `python src/gnn.py export`, then `python src/gnn.py bench`.
**Dependencies:** T1 (can start alongside T2)
**Files:** `src/gnn.py`, `requirements.txt`

## Task 5: Train `gnn/AB` and `mlp/AB`, score every row (M)

**Description:** A training loop over train-hour snapshots in time order, early stopping on val PR-AUC (patience 3 epochs), fixed seed. Score every split (tail too, from earlier edges only) into `scores` with `train.py`'s `save_scores`. `mlp/AB` uses the same loop with the graph embeddings switched off.

**Acceptance criteria:**
- [ ] `gnn/AB` and `mlp/AB` each have one score for every `tx_id` (5,078,345), no NaNs
- [ ] `gnn/AB` val PR-AUC beats `lgbm/A` (0.075); if not, debug before tuning (feature scaling, positive weight; `mlp/AB` shows what the head alone gets)
- [ ] Runtime and peak memory logged; the same seed reproduces val PR-AUC within 0.005

**Verification:** `python src/gnn.py train`, then `python src/evaluate.py --split val --phase 3 --boot 200`.
**Dependencies:** T4
**Files:** `src/gnn.py`

## Task 6: Small GNN grid and val analysis (S)

**Description:** At most about 8 runs: W ∈ {24h, 3d if it fits}, layers ∈ {2, 3}, hidden ∈ {32, 64}, then 3 seeds of the best. Log to `reports/phase3-tuning.csv`. On val:
- paired differences against `lgbm/AB`, `lgbm/ABC` and `mlp/AB`;
- recall by pattern, with UNLISTED split into ACH and non-ACH;
- the share of flagged rows that are ACH.

If `gnn/AB` beats `lgbm/AB` on val, expand `lgbm/ABG` into its own tasks (with the stacking fix) before Checkpoint B.

**Acceptance criteria:**
- [ ] Tuning log saved; chosen config named with its seed spread
- [ ] Val section of `reports/phase3-results.md` written, with the UNLISTED and ACH-share findings, and the `lgbm/ABG` go / no-go stated
- [ ] `lgbm/AB` rows still equal the frozen copy

**Verification:** `python src/evaluate.py --split val --phase 3`.
**Dependencies:** T3, T5
**Files:** `src/gnn.py`, `reports/phase3-results.md`

## Checkpoint B: review val with triet (hard stop)
- [ ] Val results for `lgbm/ABC`, `mlp/AB`, `gnn/AB` (and `lgbm/ABG` if built) reviewed **before** touching test
- [ ] triet confirms which models go to test

---

## Task 7: Score test once and write results (S)

**Description:** `python src/evaluate.py --split test --final --phase 3 --models …`, then `--split tail`. Val thresholds, 1,000 resamples, iid and attempt-cluster, paired differences as listed, recall by pattern with CIs. Write the test, tail and conclusion sections of `reports/phase3-results.md`, and one line in `decisions.md` naming the model Phase 4 explains.

**Acceptance criteria:**
- [ ] Each new model scored on test once and logged with the date; Phase 2 files untouched
- [ ] Headline answer stated plainly: is each new model significantly better than `lgbm/AB` (CI excludes 0) under iid **and** cluster CIs; UNLISTED recall with CIs
- [ ] `decisions.md` names the Phase 4 model (if the GNN wins but `lgbm/ABC` or `lgbm/ABG` is close, prefer the explainable one and say why)

**Verification:** numbers regenerate from `scores` with the logged commands.
**Dependencies:** Checkpoint B
**Files:** `src/evaluate.py` (if needed), `reports/phase3-results.md`, `docs/decisions.md`

## Checkpoint C: Phase 3 complete
- [ ] `features.py`, `train.py lgbm ABC --phase 3` and `gnn.py export train` rebuild everything; `lgbm/AB` still equals the frozen copy
- [ ] Results reviewed by triet

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| GNN sees same-minute or future edges | High | Snapshots end at the hour start, asserted per snapshot; a leaky variant must fail (T4) |
| GNN memorizes dirty accounts across the split cuts | High | No per-account embeddings or IDs; node input is a constant plus in-window degree |
| Laptop runs out of RAM or time (5M edges) | High | Timing slice on CPU and MX550 before training (T4), with stated fallbacks, then Colab |
| Retrain overwrites the frozen baseline | High | T0 backup (`scores_frozen_lgbm_AB`, `lgbm_AB.frozen.txt`); every task re-checks `lgbm/AB`; never run `train.py all` or `train.py lgbm AB` before Checkpoint C |
| Phase 2 report files silently rewritten | Med | `--phase 3` routing (T1); `git diff reports/` after every task |
| "Graph" confused with "MLP vs trees" | Med | `mlp/AB` control; `gnn/AB − mlp/AB` isolates the graph |
| "Graph information" confused with "the GNN" | Med | `lgbm/ABC` as the middle comparison |
| `lgbm/ABG` inflated by stacking on train embeddings | Med | Time-split stacking (GNN on Sep 1–4, LightGBM on Sep 5–6), only if opened |
| 3-cycle query too slow | Med | Join from the sender's past receipts; 3d → 24h if over 10 min (T2) |
| GNN seed variance bigger than the effect | Med | 3 seeds, spread reported; significance from the paired test bootstrap |
| Val optimism (early stopping, grid, threshold) | Low | As in Phase 2, only test numbers are headline |
| torch / PyG install on Python 3.13 Windows | Low | Pure-Python PyG; checked in T4 before any model code |

## Revision notes (2026-10-29)

Review against the code and the machine, then a full rewrite:
- **Phase 2 files would have been overwritten** by `train.py`'s tuning log and `evaluate.py --split val`, in T0 and T3. Fixed by `--phase 3` routing (T1), and T0 compares scores without a val run.
- **3-cycles** are time-ordered and joined from the sender's past receipts (bounded in-degree), not the receiver's sends (hub fan-out).
- **`mlp/AB` control** added, so a GNN-vs-trees difference isn't read as a graph effect.
- **`lgbm/ABG`** stays conditional (triet), with the stacking trap and a time-split fix written down. It gets expanded into tasks only if opened.
- **Checked on the machine:** an MX550 with 2 GB exists (T4 benches CPU and GPU), and `COUNT(DISTINCT)` over `RANGE` windows works in DuckDB 2.0.
- **Smaller fixes:** log-scaled node degrees; pairing reuses `lgbm/AB`'s existing test scores; the paper's F1 marked "verify before quoting"; Phase 2 test runtime corrected to 67 min.
