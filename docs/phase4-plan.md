# Phase 4 plan: improve detection, then explain it (DRAFT)

*Draft written 2026-10-30, while Phase 3's T6 seed runs were still running on Colab. The UNLISTED exploration script is in `notes/explore_unlisted.py` (git-ignored; Task 1 moves it into `src/`). **It will be updated once Phase 3 closes** (T6 val analysis, Checkpoint B, test scored once, Checkpoint C): the base model, the numbers below and the task order may change with those results. Task 1 done on branch `phase4`; nothing else started.*

## Goal

Push detection past the Phase 3 result within the locked rules (`docs/decisions.md`), aiming at what every model misses, then make the chosen model explainable to an investigator. Phase 4 is done when the finalists are scored once on test against the Phase 3 winner, with SHAP-based alert reasons and account-level alerts. The IBM Multi-GNN reference run comes last.

## Where Phase 3 leaves us (val, to be confirmed by T6 and test)

| Model | Val PR-AUC | Note |
|---|---|---|
| `lgbm/AB` (frozen Phase 2 baseline) | 0.393 | test 0.292 |
| `lgbm/ABC` | 0.444 | set C adds +0.051 (paired, significant) |
| `mlp/ABC` (baseline settings) | 0.429 | |
| `gnn/ABC` (baseline settings, CPU) | 0.462 | |
| `gnn_ibm` (IBM-style settings, Colab) | 0.483 | single seed; seeds 1–2 pending |
| `gnn_pw10` / `gnn_w72` | 0.426 / 0.448 | `gnn_w72` diverged at epoch 10 (gradient explosion); not rerun |
| `mlp_ibm` (IBM-style settings, no message passing) | 0.477 | the GNN's lead over it is only +0.006 |

- **Where the score comes from:** set B (an account's own past) gives the big step (+0.32 over set A); set C (hand-made graph features) +0.05. Better training settings lift both neural nets about equally (MLP 0.429 → 0.477, GNN 0.462 → 0.483), and **message passing itself adds only ~+0.006** over the MLP with the same inputs, possibly within seed noise.
- **The ceiling is UNLISTED laundering:** 379 of 1,036 val laundering rows (36.6%) belong to no listed scheme, and every model catches ~0 of them (non-ACH UNLISTED: 0 of 146). Realistic recall at useful precision tops out around 63%.

### UNLISTED exploration (2026-10-30, train + val; script in the session scratchpad, log `logs/explore_unlisted.log`)

| | Licit | **UNLISTED** | Listed schemes |
|---|---|---|---|
| Rows | 4,210,879 | **1,597** | 1,969 |
| ACH | 11% | **64%** | 100% |
| Median USD amount | 924 | **2,372** | 10,192 |
| Amount $10–20k | 6% | **3%** | 47% |
| Self-transfer | 14% | **0%** | 0.3% |
| Sender is a new account | 13% | **28%** | 31% |
| Sender paid this receiver before (3d) | 75% | **33%** | 7% |

- **Formats of UNLISTED:** ACH 64%, Cheque 17%, Credit Card 11%, Cash 6%, Bitcoin 3%.
- **Links to known schemes:** **1.5%** share an account with a listed scheme within train + val, the same as licit rows (1.4%). They are separate actors. *(The first pass said 2.1% because it counted scheme accounts from every split; corrected in Task 1.)*
- **Repeat accounts:** **34.6%** involve an account that had earlier laundering. This is the main lever.
- **Upper bound:** a LightGBM trained only on UNLISTED vs licit (A + B + C) reaches val PR-AUC **0.0147** (37× the 0.0004 base rate, tiny in absolute terms); **non-ACH UNLISTED 0.0016** (base 0.00017), close to noise. Top features: payment format, the sender's recent amounts and counts, pair history.

## Decisions this draft assumes (revisit after Phase 3)

- **Base model:** the Phase 3 winner on val, likely `lgbm/ABC` or the best `gnn`. Tree models stay the explainable default for Phase 4's alert reasons.
- **Same rules as Phases 2–3:** strictly-past features, windows ≤ 3d, no IDs or pattern data as features, val for all tuning, **test once per finalist**. Each extra model scored on test spends test's credibility, so at most 2 finalists go to test.
- **Outputs** use a `--phase 4` prefix (`reports/phase4-*`); the Phase 2 and 3 files are never touched.
- **GNN runs from now on use gradient clipping** (`clip_grad_norm_(…, 1.0)`).

## Tasks

### Task 1: UNLISTED diagnosis (S) — DONE 2026-10-30
**Description:** move the exploration into `src/` (`evaluate.py --analyze-unlisted`, train + val only) and write `reports/unlisted-exploration.md` with the profile, scheme links, repeat accounts and upper bound. Then answer the Task 2 go/no-go question: how many val UNLISTED rows have a sender or receiver whose earlier transactions (strictly earlier, ≤ 3d) the current models (`lgbm/ABC`, `gnn/ABC`) already scored high (≥ the model's val best-F1 threshold), compared with licit val rows (lift). Earlier rows in train carry in-sample scores (the stacking trap), so hits from val-period history are reported separately.
**Acceptance:**
- [x] `evaluate.py --analyze-unlisted` regenerates the profile, scheme-link, repeat-account and upper-bound tables and writes `reports/phase4-unlisted-*.csv`; it reads only train + val rows and asserts that.
- [x] Past-score table: share of val UNLISTED vs licit rows with an earlier high-scored transaction on either account, per model, val-only history vs any history, and the lift.
- [x] The report states the recall ceiling with its evidence and a go/no-go for Task 2.
**Verification:** the command runs end to end with its asserts; the upper-bound numbers match `logs/explore_unlisted.log` (0.0147 / 0.0016).
**Dependencies:** none. **Task 2 depends on its go/no-go.**
**Outcome:** `evaluate.py --analyze-unlisted` (~3 min, asserts train + val only) writes `reports/phase4-unlisted-*.csv`; report in `reports/unlisted-exploration.md`. Upper bound reproduces 0.0147 / 0.0016. Ceiling: listed share of val laundering is 63.4%; UNLISTED caught 2/379 (`lgbm/ABC`), 4/379 (`gnn/ABC`). **Past scores:** with val-only history, an UNLISTED row's account had an earlier alerted transaction (val best-F1 threshold, ≤ 3d) in 1/379 (`lgbm/ABC`) and 4/379 (`gnn/ABC`) rows, against 0.3% of licit rows: a handful of rows either way. At a looser top-1% cut the lift is 4–7×, but UNLISTED is at most 0.28% of those 39k–119k rows. The 34.6% repeat-account signal exists in the labels, not in the scores. **Task 2 verdict: no-go as an UNLISTED lever.** Optional re-scope: a single "account alerted in past 3d" flag on `lgbm/ABC` for listed recall (32–53 missed listed val rows have such a hit, against ~2.8k licit rows).
**Files:** `src/evaluate.py`, `reports/unlisted-exploration.md`

### Task 2: Account-risk features from past scores (M, leakage risk)
**Description:** for each transaction, the sender's and receiver's past alarm level: max and mean of their earlier transactions' model scores over 24h / 3d, strictly earlier. **Stacking trap:** scores on train rows come from a model trained on those rows. **Fix:** time-split stacking, as for `lgbm/ABG`. A model trained on Sep 1–4 scores Sep 5 onward; the account-risk features exist from Sep 5; the final model trains on Sep 5–6.
**Acceptance:** brute-force leakage check passes, with a leaky variant (in-sample scores) that must fail; val PR-AUC and UNLISTED recall vs the base model, paired.
**Files:** `src/features.py`, `src/train.py`

### Task 3: Features round 2, set D (M, leakage risk)
**Description:** forwarding speed (minutes from an account's receipt to its next send), structuring (recent sends just under $20k), amount vs the account's own usual per currency, burstiness (sends within 5 min), 2-hop counts in SQL. All strictly past, windows ≤ 3d, extending `features.py --check` with leaky variants.
**Acceptance:** leakage check 0 mismatches; drift < 1% of test accounts; `lgbm/ABCD` vs `lgbm/ABC` on val, paired, with UNLISTED split by ACH.
**Files:** `src/features.py`, `src/train.py`

### Checkpoint A
- [ ] Do Tasks 2–3 move UNLISTED recall at all? If not, the write-up states the ceiling and Phase 4 shifts to Tasks 4–6.

### Task 4: Ensemble (S)
**Description:** rank-average the best tree model and the best GNN; compare with each part on val (paired).
**Acceptance:** ensemble kept only if the paired val gain over the best single model excludes 0.
**Files:** `src/evaluate.py`

### Task 5: Account-level alerts (S)
**Description:** roll transaction scores up to one alert per account per day (max score). Account metrics use the account label already defined in `decisions.md` (≥ 1 laundering transaction in the period): precision and recall at K accounts per day.
**Acceptance:** account-level table on val next to transaction-level; K chosen and written down.
**Files:** `src/evaluate.py`, `docs/decisions.md` (account K)

### Task 6: Explainability (M)
**Description:** SHAP on the chosen tree model; plain-English alert reasons from each alert's top SHAP features (e.g. "received from 14 accounts in 3 days, sent on 95% within 1 h, new receiver for this sender"); 10 example alerts checked by hand.
**Acceptance:** reasons generated for every val alert; the hand-checked examples read correctly.
**Files:** `src/explain.py`, `reports/phase4-alert-examples.md`

### Checkpoint B (hard stop)
- [ ] triet reviews val results and picks at most 2 finalists for test

### Task 7: Test once and results (S)
**Description:** score the finalists once on test (`--phase 4 --final`), paired against the Phase 3 winner and the frozen `lgbm/AB`; write `reports/phase4-results.md`.
**Files:** `src/evaluate.py`, `reports/phase4-results.md`, `docs/decisions.md`

### Task 8 (end): IBM Multi-GNN reference run (M)
**Description:** run IBM's code unchanged on HI-Small in a separate Colab notebook (`--model gin --emlps --reverse_mp --ego --ports`), map their test predictions to our `tx_id`s, evaluate with `metrics.py` including recall by pattern. Label it "IBM reference (transductive: sees later edges)". Check the repo's licence before reusing any code.
**Acceptance:** their score next to ours, with the time-rule difference explained; their UNLISTED recall reported.
**Files:** a separate notebook; `reports/phase4-results.md`

## Optional, if time allows
- 2 extra seeds of `lgbm/ABC` (named e.g. `lgbm/ABC_s1`) to measure LightGBM's seed spread.
- `gnn_w72` rerun with clipping, to see whether the 3-day window helps.
- LI-Small robustness run (lower laundering ratio).
- Phase 5 (thin dashboard), as in the original plan.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Account-risk features leak through in-sample scores | High | time-split stacking; leaky variant must fail (Task 2) |
| UNLISTED is simply not separable in this synthetic data | High | Task 1 states it with evidence; Checkpoint A decides whether to stop chasing it |
| Test credibility worn down by many models | Med | ≤ 2 finalists, scored once, all logged |
| GNN instability (explosions) | Med | gradient clipping in every new GNN run |
| Memory on the laptop | Med | Colab for GNNs; DuckDB caps as in Phase 3 |

## Open questions (after Phase 3)
1. Which model is Phase 4's base: `lgbm/ABC` (explainable) or the best `gnn` (if the seeds confirm a real gain)?
2. Keep Phase 5 (dashboard) in scope, or end with Phase 4's report?
