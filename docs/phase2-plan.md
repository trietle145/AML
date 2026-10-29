# Phase 2 plan: Baseline

*Written 2026-10-28; revised the same day after review against the data (see "Revision notes" at the end). Based on `docs/decisions.md` (locked), `src/metrics.py` and `docs/plan-review.md`. Planning only; nothing implemented yet.*

## Goal

Build the number Phase 3 has to beat: a per-transaction + behavioral baseline, scored exactly as `docs/decisions.md` says. Phase 2 is done when the test-period results for logistic regression and LightGBM are written down with bootstrap CIs, and every transaction's baseline score is stored so Phase 3 can run a **paired** bootstrap against it on the same rows.

## Phase 1 review (what Phase 2 builds on)

Checked against `docs/phase1-plan.md` and `docs/plan-review.md` by reading the code, the executed notebook and `decisions.md`. Nothing was re-run (the device shell was unavailable).

**Solid:**
- Bank and account IDs are loaded as VARCHAR with an explicit schema; account key is `bank:account`. Row and laundering counts are asserted.
- Time split by calendar day, cut dates in one place (`CUTS` in `src/split.py`), non-overlap asserted. The thin Sep 11–18 tail is its own `tail` split with a written reason.
- Feature rule is written: strictly earlier `ts` only, look-back across splits allowed, pattern types / raw IDs / `tx_id` / `split` never features.
- Patterns: 100% match, 0 ambiguous, 38% UNLISTED bucket; evaluation-only.
- `metrics.py` covers PR-AUC, val-chosen F1 threshold, P/R at 500 alerts/day, recall by pattern, bootstrap and paired bootstrap, with a self-check and a random / "is ACH" floor.

**Small fixes (none block Phase 2), all done 2026-10-28:**
1. [x] `data/raw/README.md` is never committed, because `.gitignore` ignores all of `data/`. Replace `data/` with `data/*`, `!data/raw/`, `data/raw/*`, `!data/raw/README.md`.
2. [x] Notebook section 2 prints a pandas `UserWarning` (`.round` on a datetime column). Round only the numeric columns.
3. [x] `CLAUDE.md` still points "Current phase" at `phase1-plan.md`; point it at this file.

**One methodological question (resolved 2026-10-28):** the bootstrap resamples transactions independently, but transactions in one laundering attempt are correlated, so the CIs are probably too narrow. T7 also reports an **attempt-cluster bootstrap** as a sensitivity check next to the locked iid one: resample whole attempts, and treat every other row as its own cluster. A day-block bootstrap is ruled out because test has only 2 days, i.e. 2 blocks.

## Decisions this plan assumes

- **Two feature sets, cleanly separated so Phase 3's ablation is honest:**
  - **A, per-transaction:** USD log amount, payment format, payment and receiving currency, hour, `is_cross_bank`, `is_currency_mismatch`, `is_self_transfer`. *(A round-amount flag was planned and dropped in T1: amounts ≥ 100 that are a multiple of 100 in the paid currency are only 0.00–0.05% of rows in every currency, laundering or not, so the flag would be almost always 0.)*
  - **B, behavioral:** each account's *own* past activity on *one side* only. For the sender: its past outgoing transactions. For the receiver: its past incoming transactions. Counts, USD sums, mean / max, amount vs own past mean, time since last transaction (capped at 3d), share of past ACH. Windows **1h / 24h / 3d**.
  - **Saved for Phase 3 (graph), not in B:** anything about counterparties or flows through an account. Distinct counterparties (fan-in/out), in-vs-out on the same account (pass-through), pair history ("sent to this receiver before"), cycles. Pair history stays here: it is a property of the edge between two accounts, not of one account's own activity.
- **No feature may grow with absolute time.** All accounts appear from Sep 1, so "how long has this account been seen" measures how long the dataset has been running. Median hours since an account's first send are 37 in train, 166 in val and 204 in test, with a train maximum of 144. Test values would sit beyond anything the model trained on. So: no "time since first seen"; windows no longer than 3d (the longest that fills up inside train's 6 days); "time since last" capped at 3d.
- **Past-only in SQL:** timestamps are whole minutes (verified: 0 rows with non-zero seconds), so "strictly earlier" is a DuckDB `RANGE BETWEEN INTERVAL 'w' PRECEDING AND INTERVAL 1 MINUTE PRECEDING` window per account. T1 asserts zero seconds. **Never `LAG`** for "time since last": the previous row is usually in the same minute (5.06M of 5.08M rows share a minute). Use `MAX(ts)` over a frame ending 1 minute before the row.
- **No day-of-week feature.** With 10 days, weekday is confounded with the split (test has one weekend-like day).
- **Exchange rates** come from the data itself: on currency-mismatch rows of the **train** split, take the median `amount_received / amount_paid` for **USD → currency** and invert it. Every one of the 14 non-USD currencies has a USD pair in train in both directions. USD → X is tight (min/max ≥ 0.98 for most currencies); X → USD is noisy (min/max down to 0.46, from rounding on tiny amounts). The X → USD median is only a cross-check. Treated as public reference data.
- **Models fit on train only**, tuned and thresholded on val, scored once on test (and tail). No refit on train+val, so the val threshold stays valid for test as `decisions.md` requires. Val is used three times (early stopping, grid choice, threshold), so val numbers are optimistic; **test is the only unbiased number**.
- **Outputs:** feature table `features` and score table `scores(model, feature_set, tx_id, split, score)` in `data/aml.duckdb`. Train-split scores are in-sample: `evaluate.py` refuses `--split train`. LightGBM models saved under `data/models/` for SHAP in Phase 4.
- Account-level rollup and metrics wait for Phase 4.
- New dependency: `lightgbm` (pinned in `requirements.txt`).

## Dependency graph

```
T1 per-tx features ─ T2 LR on A, scored on val ─┬─ T3 behavioral features ─ T4 leakage check ─┐
                                                └──────────────────────────────────────────────┴─ T5 LightGBM + LR on A and A+B ─ T6 val analysis ─ T7 test run + results
```

All sequential. T2 is a thin end-to-end slice (features → model → metrics) so the scoring harness exists before the risky feature work.

---

## Task 1: Exchange rates and per-transaction features (S) — DONE 2026-10-28

**Description:** `src/features.py` builds table `features` with `tx_id` and feature set A. USD amounts use per-currency rates derived from train-split currency-mismatch rows.

**Acceptance criteria:**
- [x] One row per transaction (`COUNT(*) = COUNT(DISTINCT tx_id) = 5,078,345`), no nulls in A
- [x] A rate for all 15 currencies; USD = 1; rates printed. Each USD → X rate agrees with the inverted X → USD median within 1% (asserted)
- [x] All `ts` have zero seconds (asserted; the window trick in T3 depends on it)

**Outcome:** builds in about 3 s. Rates are stored in table `fx_rates` (EUR 1.1718, GBP 1.2917, BTC 11,875 USD; largest gap to the cross-check is Bitcoin at 0.05%). `features` has 8 set-A columns. Spot-checked BTC, Yen and Euro rows by hand.

**Verification:** `python src/features.py` runs its asserts; spot-check 3 non-USD rows by hand (EUR should be ≈ 1.17 USD).

**Dependencies:** None
**Files:** `src/features.py`

## Task 2: End-to-end slice: logistic regression on A, scored on val (M) — DONE 2026-10-28

**Description:** `src/train.py` loads a split's features from DuckDB (float32), fits a scikit-learn logistic regression (one-hot categoricals, standardized numerics, `class_weight="balanced"`) on train, writes scores for all rows to `scores`. `src/evaluate.py` reads `scores` and runs every `metrics.py` function on a chosen split (val by default), plus recall by pattern from `tx_pattern`. Also stores the "is ACH" floor as a scorer so every table has it.

**Acceptance criteria:**
- [x] `python src/train.py lr A` then `python src/evaluate.py --split val` prints PR-AUC [CI], best F1 + threshold, P/R@500/day, recall by pattern, for `is_ach` and `lr/A`
- [x] `lr/A` val PR-AUC beats the `is_ach` floor (0.0067)
- [x] `evaluate.py` refuses `--split test` unless `--final` is given (test is scored once), and refuses `--split train` (in-sample)

**Verification:** both commands run; the `is_ach` PR-AUC point estimate matches 0.0067 in `decisions.md`. CI bounds may differ slightly: the floor used 200 resamples, `evaluate.py` uses 1,000.

**Dependencies:** T1
**Files:** `src/train.py`, `src/evaluate.py`, `requirements.txt`
**Outcome (val, 1,000 resamples, about 4 min):** `is_ach` PR-AUC 0.0067 [0.0062, 0.0072], F1 0.015. `lr/A` PR-AUC 0.0145 [0.0130, 0.0161], F1 0.036 at threshold 0.916, P/R@500/day 0.019 / 0.018. `lr/A` recall at its val threshold is 0.32–0.51 on listed patterns but only 0.09 on UNLISTED. `is_ach` P@500/day reads 0.012 here against 0.008 in the decisions floor: with a binary score, the top 500 is a random draw among tied ACH rows, so it moves with row order. Its PR-AUC is unaffected. `python src/train.py all` currently runs `is_ach` + `lr A` in about 25 s.

## Task 3: Behavioral features (M, highest risk) — DONE 2026-10-29

**Description:** Add feature set B to `features.py` as DuckDB window queries: per `from_acct` over its past outgoing rows and per `to_acct` over its past incoming rows, for 1h / 24h / 3d, all with the frame ending 1 minute before the row. "Time since last" is `ts - MAX(ts)` over that frame (never `LAG`), capped at 3d. Accounts with no history get 0 counts, NULL ratios, the 3d cap for "time since last", and a `has_history` flag for LR.

**Acceptance criteria:**
- [x] B columns exist for all rows; same-minute rows never count each other (a row's 1h count and "time since last" ignore rows with the same `ts`)
- [x] No B column drifts with absolute time: for each column, the share of test **accounts** with a value above the train maximum is printed and is < 1% *(changed from rows, see Outcome)*
- [x] Builds in < 10 min on the laptop; hub accounts (168k sends) don't blow up; no column uses counterparties, the other side's flow, or anything under "Saved for Phase 3"

**Verification:** `python src/features.py` runs; print feature summary per split and check that Sep 1 rows have small history values (warm-up) but no NULL-for-everyone columns.

**Dependencies:** T2
**Files:** `src/features.py`

**Outcome:** 34 B columns (`SET_B` in `features.py`). Per side (`out` = sender's past sends, `in` = receiver's past receipts) and per window (1h, 24h, 3d): count, USD sum, USD max, amount vs own past mean, ACH share; plus `min_since_last` (capped at 4,320) and `has_history`. A + B build in about 33 s. Built-in checks: no unexpected NULLs; every account's first-ever minute on a side has count 0 and the cap, including same-minute peers; 1h ≤ 24h ≤ 3d counts. The drift check first failed on rows: 2.3% of test rows have `out_cnt_3d` above the train max, all from one hub, `070:100428660`. It sends about 59.8k on Sep 7–9 against at most 44.7k in any 3 train days, so it is a volume quirk, not time drift. The check now counts accounts: 0.0% for every column. Trees treat above-max values as the top bin, so the hub is harmless. Warm-up looks as expected: Sep 1 averages 342 past sends per row (3d) against 2,100–3,200 from Sep 2 on. DuckDB is capped at `memory_limit = 3GB`: the machine has about 4 GB free of 16, and the default 12.5 GiB limit made the final join fail an OS allocation instead of spilling to disk.

## Task 4: Leakage check (S) — DONE 2026-10-29

**Description:** A brute-force recomputation of B in pandas for ~300 sampled `tx_id`s (oversampling hubs, same-minute ties, Sep 1, split boundaries and laundering rows), using only raw rows with `ts` strictly earlier. Must equal the SQL values, "time since last" included.

**Acceptance criteria:**
- [x] All sampled values match within float tolerance
- [x] Two deliberately leaky variants fail the check, proving the check can fail: a frame ending at `CURRENT ROW`, and "time since last" computed with `LAG`

**Verification:** `python src/features.py --check` (or an `assert` block in `features.py`) passes.

**Dependencies:** T3
**Files:** `src/features.py`
**Outcome:** `python src/features.py --check` takes about 30 s. It recomputes set B in pandas for 244 sampled rows (fewer than 300 because the categories overlap) using only rows with `ts` strictly earlier: random, 3 top senders and 3 top receivers, same-minute ties on each side, Sep 1, ±2h around both split cuts, and 50 laundering rows. Result: **0 mismatched cells** across all 34 columns. Both leaky variants are caught: frame ending at `CURRENT ROW` gives 6,443 bad cells in 34 columns, and `LAG` for "time since last" gives 127 bad cells in the 2 `min_since_last` columns. Window SQL now lives in `b_query(leak=None)`, so the leaky variants share the real query.

## Checkpoint A (after T1–T4)
- [x] `python src/features.py` rebuilds A + B from the database in one command and its checks pass
- [x] LR on A is scored on val end-to-end

---

## Task 5: LightGBM and LR on A and A+B (M) — DONE 2026-10-29

**Description:** Extend `train.py`: LightGBM with early stopping on val average precision, a small grid (num_leaves, learning rate, min_child_samples; ~10 runs), fixed seed; LR on A+B. Four scored models: `lr/A`, `lr/AB`, `lgbm/A`, `lgbm/AB`. Best LightGBM boosters saved to `data/models/`.

**Acceptance criteria:**
- [x] All four models in `scores`, for every split
- [x] Grid results (params → val PR-AUC) logged to `reports/phase2-tuning.csv`
- [x] `lgbm/AB` val PR-AUC ≥ `lgbm/A`; if not, investigate before moving on

**Verification:** `python src/evaluate.py --split val` shows all four plus the floor.

**Dependencies:** T4
**Files:** `src/train.py`, `requirements.txt`

**Outcome (val, 1,000 resamples):**

| Model | PR-AUC [95% CI] | F1 @ val threshold | P@500/day | R@500/day |
|---|---|---|---|---|
| is_ach | 0.0067 [0.0062, 0.0072] | 0.015 | 0.012 | 0.012 |
| lr/A | 0.0145 [0.0130, 0.0161] | 0.036 | 0.019 | 0.018 |
| lr/AB | 0.0348 [0.0302, 0.0418] | 0.068 | 0.055 | 0.053 |
| lgbm/A | 0.0749 [0.0633, 0.0893] | 0.163 | 0.154 | 0.149 |
| lgbm/AB | 0.3926 [0.3596, 0.4255] | 0.466 | 0.446 | 0.431 |

- Best grid points: `lgbm/A` 15 leaves, lr 0.05, min_child 500, 102 trees; `lgbm/AB` 63 leaves, lr 0.05, min_child 500, 237 trees. Runs with `min_child_samples = 50` peak after 1–65 trees (overfitting), so more capacity or compute won't help. Grid: about 2 min (A) and 5 min (AB).
- Recall at the val threshold for `lgbm/AB` is 0.37–0.83 on listed patterns (FAN-IN 0.75, GATHER-SCATTER 0.80, SCATTER-GATHER 0.83) but **0.008 on UNLISTED (3 of 379)**. Logistic regression stays far behind, so the signal lives in interactions.
- No single B feature is a shortcut: the best single-feature val PR-AUC is 0.023 (`in_amt_vs_mean_3d`). The jump from A to AB comes mostly from accounts with no history: about 2.3% of val rows (no past sends or no past receipts) hold 53% of val laundering (551 of 1,036), a mule-account-like effect. It is legal (strictly past) and goes in the T6 write-up.
- Memory: scores are written in 500k-row chunks, and the string columns load as `category` (the first A+B run ran out of RAM in `load`).

## Task 6: Validation analysis (S)

**Description:** On val only: feature importance (gain) for `lgbm/AB`, recall by pattern per model, and a leakage smell test on single features (each feature's own val PR-AUC, both signs). Above **0.07** (about 10× "is ACH") the feature is looked at and explained. Above **0.5** it is treated as a leak until proven otherwise. Short written findings.

**Acceptance criteria:**
- [ ] Importance table and per-pattern recall saved in `reports/`
- [ ] Each suspicious feature is either explained or removed, with the reason written
- [ ] Written note on how much of listed-pattern recall comes from ACH alone (see `decisions.md` § Baseline floor)

**Verification:** findings readable in `reports/phase2-results.md` (val section).

**Dependencies:** T5
**Files:** `src/evaluate.py`, `reports/phase2-results.md`

## Checkpoint B
- [ ] Review val results with triet **before** touching test

---

## Task 7: Score test once and write results (S)

**Description:** `python src/evaluate.py --split test --final`: all four models and the floor on test and tail (separate columns), val-chosen thresholds, 1,000× bootstrap CIs, paired bootstrap `lgbm/AB − lr/AB` and `lgbm/AB − lgbm/A`, recall by pattern with CIs. Plus the attempt-cluster bootstrap as a sensitivity column next to the locked iid CIs.

**Acceptance criteria:**
- [ ] `reports/phase2-results.md` has the headline test table, tail table, paired differences, per-pattern recall, and iid vs attempt-cluster CI widths side by side
- [ ] Thresholds are the val ones; the test run happened once (logged with date in the results file)
- [ ] `decisions.md` gets one line naming `lgbm/AB` (or whichever wins on val) as **the Phase 3 baseline**

**Verification:** numbers in the report regenerate from `scores` with one command.

**Dependencies:** T6
**Files:** `src/evaluate.py`, `reports/phase2-results.md`, `docs/decisions.md`

## Checkpoint C: Phase 2 complete
- [ ] `python src/features.py && python src/train.py all && python src/evaluate.py --split val` rebuilds everything
- [ ] Results reviewed by triet; baseline scores frozen in `scores` for Phase 3's paired comparison

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Same-minute or future rows leak into windows | High | 1-minute frame offset, zero-seconds assert (T1), brute-force check that is shown to fail on a leaky variant (T4) |
| Behavioral features quietly include graph signal, shrinking Phase 3's gain | High | Written A / B / graph boundary above; reviewed in T3 |
| Features that grow with absolute time shift between train and test (time since first seen, 7d windows) | High | Banned in Decisions; windows ≤ 3d, "time since last" capped; drift check in T3 |
| Account IDs memorized via near-ID features (e.g. exact first-seen time) | Med | No absolute or first-seen times at all; importance check in T6 |
| Window queries slow on hub accounts | Med | DuckDB range windows; materialize `transactions` as a table if needed (the `ponytail` note in `clean.sql`) |
| Sep 1 warm-up rows have no history | Low | Keep them; `has_history` flag; check per-day val PR-AUC in T6 |
| iid bootstrap CIs too narrow (attempt clustering) | Med | Attempt-cluster bootstrap in T7 as sensitivity |
| Boundary (label-delay) leakage: attempts crossing a cut, so the model has seen labels from the start of a scheme | Low | Measured 2026-10-29: 26% of test pattern rows come from attempts that started in train, and 31% of test laundering rows touch an account dirty in train (val: 64% / 51%). Accepted by triet; it hits all models equally, so the ablation stays fair. Val's 2-day gap protects test |
| Val numbers optimistic (val used for early stopping, grid and threshold) | Low | Stated in Decisions; only test numbers are headline |
| 3.2M × ~50 features in pandas on a laptop | Low | float32, load one split at a time |

## Open questions (resolved 2026-10-28)

1. Pair history ("has this sender paid this receiver before"): **Phase 3 graph feature**, not behavioral.
2. Extra bootstrap in T7: **attempt-cluster only**; day-block is ruled out with 2 test days.

## Revision notes (2026-10-28)

Checked against `data/aml.duckdb` before any code:
- Windows changed from 1h / 24h / 7d to 1h / 24h / 3d, and "time since first seen" was dropped, because both drift with absolute time (see Decisions). T3 has a drift check.
- "Time since last" must not use `LAG` (same-minute leak). T4 tests a `LAG` variant to show the check catches it.
- The exchange-rate check changed from "min/max within a few %", which fails on the real data (down to 0.46), to "USD → X agrees with the inverted X → USD median within 1%".
- Floor match in T2 is on the point estimate only; train scores are in-sample and not evaluated; smell-test thresholds are concrete; val is marked as optimistic.
