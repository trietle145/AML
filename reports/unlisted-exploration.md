# UNLISTED laundering: diagnosis (Phase 4, Task 1)

*Train + val only; test was not read. Regenerate with `.venv/Scripts/python -u src/evaluate.py --analyze-unlisted > logs/analyze_unlisted.log 2>&1` (about 3 min). Tables: `reports/phase4-unlisted-{profile,formats,links,past-scores,upper-bound}.csv`. Written 2026-10-30, before Phase 3's T6 seeds finished, so the base models here are `lgbm/ABC` and the baseline `gnn/ABC`.*

## Summary

- **Recall ceiling of about 63%.** Of val's 1,036 laundering rows, 379 (36.6%) belong to no listed scheme. Every model catches about none of them: at the val best-F1 threshold, `lgbm/ABC` flags 2 of 379 and `gnn/ABC` 4. Non-ACH UNLISTED: 0 of 146 for both. Unless UNLISTED becomes detectable, recall at useful precision stops at the listed share, 657 / 1,036 = **63.4%**.
- **UNLISTED is a separate population.** Smaller amounts, less ACH, no self-transfers, and new sender–receiver pairs. It shares accounts with listed schemes no more often than licit traffic does (1.5% vs 1.4%).
- **Repeat accounts are real but invisible to the models.** 34.6% of UNLISTED rows touch an account with earlier laundering (licit: 9.6%), but the models did not score those earlier transactions high.
- **Task 2 (account-risk features from past scores): no-go as an UNLISTED lever.** Details below. It may still help listed schemes, as a smaller re-scoped task.
- **Even a dedicated model barely separates UNLISTED.** LightGBM trained on UNLISTED vs licit only reaches val PR-AUC 0.0147 (37× base rate, still tiny); on non-ACH UNLISTED 0.0016.

## 1. Profile (train + val)

| | Licit | **UNLISTED** | Listed schemes |
|---|---|---|---|
| Rows | 4,210,879 | **1,597** | 1,969 |
| ACH | 11% | **64%** | 100% |
| Median USD amount | 924 | **2,372** | 10,192 |
| Amount $10–20k | 6% | **3%** | 47% |
| Cross-bank | 84% | **99%** | 98% |
| Self-transfer | 14% | **0%** | 0.3% |
| Sender is a new account | 13% | **28%** | 31% |
| Receiver is a new account | 11% | **12%** | 21% |
| Sender paid this receiver before (3d) | 75% | **33%** | 7% |

UNLISTED payment formats: ACH 64%, Cheque 17%, Credit Card 11%, Cash 6%, Bitcoin 3%. Listed schemes are 100% ACH. That is why "is ACH" alone finds every listed pattern and only 64% of UNLISTED.

UNLISTED sits between licit and listed on every column. It is closer to licit on amount and receiver age, and closer to listed on cross-bank and sender age. No single feature separates it.

## 2. Links to listed schemes and to earlier laundering

| | Licit | **UNLISTED** | Listed |
|---|---|---|---|
| Shares an account with a listed scheme (train + val) | 1.4% | **1.5%** | 100% |
| ...and that scheme account appeared earlier | 0.6% | **0.8%** | 74% |
| An account had earlier laundering (any type) | 9.6% | **34.6%** | 74% |

- **Correction to the draft plan:** last session's 2.1% counted scheme accounts from every split, including test and tail. Its "seen earlier = 100%" came from `AVG` skipping NULLs, so it only covered the rows that do share an account. Restricted to train + val with NULLs counted as "no", UNLISTED shares accounts with schemes at the licit rate. These are separate actors.
- **Repeat accounts:** an UNLISTED row is 3.6× as likely as a licit row to touch an account that laundered before. This uses labels, which a model never sees at scoring time. Section 3 checks whether the models' own scores carry the same signal.

## 3. Do past model scores flag UNLISTED accounts? (Task 2 go/no-go)

For each val row: does its sender or receiver have an earlier transaction (strictly earlier, within 3 days, either side) that the model scored "high"? Two definitions of "high":

- **best_f1:** at or above the model's val alert threshold.
- **top1pct:** in the top 1% of val scores, a much looser cut.

"Val history" counts only earlier transactions from val. Train-period scores are in-sample (the stacking trap) and would flatter train laundering.

Val, history from val only (the honest version):

| Model | High = | UNLISTED rows with a hit | Licit rows with a hit | Lift | UNLISTED share of hit rows |
|---|---|---|---|---|---|
| `lgbm/ABC` | best_f1 | 1 / 379 (0.3%) | 0.29% | 0.9× | 0.03% |
| `gnn/ABC` | best_f1 | 4 / 379 (1.1%) | 0.30% | 3.6× | 0.13% |
| `lgbm/ABC` | top1pct | 200 / 379 (53%) | 12.2% | 4.3× | 0.17% |
| `gnn/ABC` | top1pct | 109 / 379 (29%) | 4.0% | 7.2× | 0.28% |

(UNLISTED base rate in val: 0.04%. Including in-sample train history gives the same picture: best_f1 3 and 6 rows; full table in `phase4-unlisted-past-scores.csv`.)

- **At the alert threshold, there is nothing to work with.** The accounts behind UNLISTED laundering were almost never alerted on before. Their earlier laundering was mostly UNLISTED too, and the models miss that as well. The 34.6% repeat-account signal from section 2 does not survive the step from labels to scores.
- **Below the threshold, the lift is 4–7×.** But the hit set is large: 39k–119k val rows for 2 days, against a queue of 1,000 alerts (K = 500/day). At most 0.28% of hit rows are UNLISTED. A 4–7× flag cannot lift UNLISTED rows into a 500/day queue when the dedicated UNLISTED model (section 4), which already gets 37×, reaches only PR-AUC 0.015.
- **Where it could help instead:** listed rows. 32 (`gnn/ABC`) and 53 (`lgbm/ABC`) val listed rows are missed by the model but have an alerted account in the past 3 days (5–8% of listed val laundering). Against them stand about 2.8k licit rows with a hit, so the realistic gain is small.

**Verdict:** Task 2 as written (motivated by UNLISTED) is **no-go**. A re-scoped version aimed at listed recall is optional and low-value. It is cheapest to test as a single "account alerted in past 3d" flag on top of `lgbm/ABC`, before building the full stacking pipeline.

## 4. Upper bound: a model trained only on UNLISTED

LightGBM (15 leaves, learning rate 0.05, min_child_samples 500, early stopping on val) on A + B + C. Trained on train UNLISTED vs licit, with listed positives dropped. Scored on val UNLISTED vs licit.

| Val rows | Positives | PR-AUC | Base rate | Lift |
|---|---|---|---|---|
| All | 379 | 0.0147 | 0.0004 | 37× |
| Non-ACH | 146 | 0.0016 | 0.00017 | 9× |

Best iteration 94. Top features by gain: payment_format 13%, out_usd_max_3d 8%, out_usd_sum_24h 8%, out_cnt_24h 7%, out_cnt_3d 7%, pair_cnt_3d 7%. The model mostly relearns "ACH, larger than the sender's usual, new pair". Without ACH it is close to noise.

Early stopping on val makes this slightly optimistic, which is fine for an upper bound.

## What this means for Phase 4

1. State the ~63% recall ceiling in the write-up, with this file as the evidence.
2. Drop Task 2 as an UNLISTED lever. Keep at most the cheap one-flag check for listed recall.
3. Task 3 (set D) is the last place UNLISTED could move. Candidates by the profile: amount vs the sender's own usual, new-pair rate, cross-bank. Check UNLISTED recall split by ACH, as the plan already says. If it doesn't move (Checkpoint A), Phase 4 shifts to the ensemble, account alerts and explainability.

## Caveats

- Val history for a Sep 7 row covers only hours, so "val history" hit rates are a lower bound. The "any history" rates give the upper side: 3 and 6 rows at best_f1.
- `gnn/ABC` here is the baseline-settings run (val 0.462), not `gnn_ibm` or the T6 seeds. The best_f1 result is the same for both models, so a better GNN is unlikely to change the verdict.
- Thresholds and the top-1% cut come from val. This is a diagnosis, not a model selection.
