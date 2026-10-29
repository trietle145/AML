# Phase 2 results: per-transaction + behavioral baseline

*Rules: `docs/decisions.md`. Plan: `docs/phase2-plan.md`. All tables regenerate with `python src/evaluate.py --split val` (metrics, recall by pattern) and `python src/evaluate.py --analyze` (importance, single features, history), which write the CSVs in `reports/`.*

## Val (Sep 7–8; 965,524 rows, 1,036 laundering)

Val was used for early stopping, grid choice and thresholds, so these numbers are optimistic. Test is the unbiased one (T7).

### Headline (`val-metrics.csv`)

| Model | PR-AUC [95% CI] | F1 @ val threshold | Threshold | P@500/day | R@500/day |
|---|---|---|---|---|---|
| is_ach (floor) | 0.0067 [0.0062, 0.0072] | 0.015 | 1.000 | 0.012 | 0.012 |
| lr/A | 0.0145 [0.0130, 0.0161] | 0.036 | 0.916 | 0.019 | 0.018 |
| lr/AB | 0.0348 [0.0302, 0.0418] | 0.068 | 0.983 | 0.055 | 0.053 |
| lgbm/A | 0.0749 [0.0633, 0.0893] | 0.163 | 0.056 | 0.154 | 0.149 |
| **lgbm/AB** | **0.3926 [0.3596, 0.4255]** | **0.466** | 0.165 | **0.446** | **0.431** |

- **Behavior + trees is the jump.** B lifts LightGBM 5.2× (0.075 → 0.393) but logistic regression only 2.4× (0.015 → 0.035). The signal is in bands and combinations (e.g. ACH, no history, amount just under $20k) that a linear model can't express without hand-built bins and interactions.
- **`lgbm/AB` on a 500-alert daily queue:** about 45% of alerts are real and 43% of val laundering is caught.

### Recall by pattern at the val threshold (`val-recall-by-pattern.csv`)

| Pattern | n | is_ach | lr/A | lr/AB | lgbm/A | lgbm/AB |
|---|---|---|---|---|---|---|
| BIPARTITE | 49 | 1.000 | 0.469 | 0.102 | 0.429 | 0.490 |
| CYCLE | 66 | 1.000 | 0.333 | 0.106 | 0.318 | 0.439 |
| FAN-IN | 60 | 1.000 | 0.417 | 0.100 | 0.383 | 0.750 |
| FAN-OUT | 64 | 1.000 | 0.422 | 0.203 | 0.375 | 0.641 |
| GATHER-SCATTER | 96 | 1.000 | 0.510 | 0.229 | 0.448 | 0.802 |
| RANDOM | 38 | 1.000 | 0.316 | 0.079 | 0.158 | 0.368 |
| SCATTER-GATHER | 134 | 1.000 | 0.410 | 0.142 | 0.381 | 0.828 |
| STACK | 150 | 1.000 | 0.360 | 0.133 | 0.360 | 0.493 |
| **UNLISTED** | **379** | 0.615 | 0.090 | 0.053 | 0.003 | **0.008** |

- **`lgbm/AB` finds the generated schemes and almost nothing else.** It reaches 0.75–0.83 on fan-in and the gather/scatter types, 0.37–0.49 on cycles, stacks, bipartite and random, and **3 of 379 UNLISTED (0.8%)**.
- **Reading `lr/AB` against `lr/A`:** per-pattern recall is *lower* for `lr/AB` because its F1-optimal threshold sits at 0.983 and it flags fewer rows. Its PR-AUC (threshold-free) is still 2.4× higher, so this is a threshold effect, not a worse model.

### How much is "ACH"

- **`lgbm/AB` ranks well inside ACH, so it isn't just using ACH.** Restricted to ACH rows only, its PR-AUC is **0.459** against an ACH base rate of 0.0076, about 60× the base rate.
- **It only ever alerts on ACH.** **99.9%** of the rows it flags at the val threshold are ACH. 3,208 of the 3,209 pattern transactions are ACH, so the listed-pattern recall above comes from ranking *within* ACH. But 38.5% of UNLISTED laundering in val is not ACH (Cheque, Credit Card, Cash, Bitcoin), and that part is essentially never flagged. The ACH-only UNLISTED rows are missed too, since only 3 UNLISTED are caught in total.
- **Implication for Phase 3:** UNLISTED is where graph features have to prove themselves. Listed-pattern recall is already high, and partly inflated because every listed scheme is ACH.

### Leakage smell test (`val-single-feature.csv`)

Each of the 42 A+B features was scored alone on val, both signs (categoricals scored by their val laundering rate).

- **The best single feature reaches PR-AUC 0.0227** (`in_amt_vs_mean_3d`), then `out_ach_share_3d` 0.0136 and `in_cnt_3d` 0.0129. **None is above the 0.07 "look" bar**, let alone the 0.5 "leak" bar, so no feature is removed.
- **That a single LightGBM tree already reaches 0.35** is explained by combinations, mainly the history effect below, not by any one feature.

### What `lgbm/AB` relies on (`val-importance.csv`, gain)

| Feature | Share of gain |
|---|---|
| out_ach_share_24h | 28.4% |
| out_ach_share_1h | 14.6% |
| in_usd_max_24h | 7.2% |
| out_cnt_24h | 5.9% |
| log_amount_usd | 5.5% |
| out_min_since_last | 4.8% |
| is_currency_mismatch | 4.1% |
| in_cnt_3d | 3.1% |
| in_min_since_last | 2.6% |
| out_amt_vs_mean_3d | 2.6% |

- **The sender's recent ACH share is 43% of all gain.** A sender whose last 1–24h of sends are all ACH is the strongest signal. That fits the generator: laundering accounts run ACH schemes. It's the behavioral version of the "is ACH" shortcut.
- **Amount and the currency-mismatch flag (a perfect negative)** are the main set-A contributors.

### Accounts without history (`val-history.csv`)

| Sender has past sends | Receiver has past receipts | Rows | Laundering | Rate |
|---|---|---|---|---|
| yes | yes | 943,605 | 485 | 0.05% |
| no | yes | 14,524 | 279 | 1.92% |
| yes | no | 4,847 | 159 | 3.28% |
| no | no | 2,548 | 113 | 4.43% |

- **2.3% of val rows hold 53% of val laundering** (551 of 1,036): a transaction where the sender or the receiver hasn't been active on that side in the past 3 days. This looks like mule accounts that only become active for a scheme.
- **It is legal, not a leak.** "No history" means no strictly earlier rows (verified in T4), so nothing uses the future.
- **It will help any model.** It's part of why A+B is a strong baseline for Phase 3.

### Known optimism (accepted)

- **Val numbers are tuned on val.** Early stopping, the grid and thresholds were all chosen there.
- **Boundary leakage:** 64% of val pattern rows come from attempts that started in train (`docs/phase2-plan.md`, Risks). Test has a 2-day gap, and only 26% of its pattern rows come from attempts that started in train.

## Test (Sep 9–10; 862,792 rows, 956 laundering): scored once on 2026-10-29 at 17:08–18:12

`python src/evaluate.py --split test --final`. Thresholds are the val ones. Bootstrap: 1,000 resamples, both iid and attempt-cluster (one laundering attempt resampled as a unit). `evaluate.py` refuses to score test again while `reports/test-metrics.csv` exists. To regenerate the numbers, delete the `test-*.csv` files first; everything is seeded, so the numbers reproduce exactly.

### Headline (`test-metrics.csv`)

| Model | PR-AUC | iid 95% CI | Attempt-cluster 95% CI | F1 | P@500/day | R@500/day |
|---|---|---|---|---|---|---|
| is_ach (floor) | 0.0062 | [0.0058, 0.0067] | [0.0054, 0.0072] | 0.014 | 0.004 | 0.004 |
| lr/A | 0.0146 | [0.0131, 0.0165] | [0.0120, 0.0177] | 0.034 | 0.031 | 0.032 |
| lr/AB | 0.0259 | [0.0227, 0.0305] | [0.0211, 0.0327] | 0.066 | 0.066 | 0.069 |
| lgbm/A | 0.0703 | [0.0591, 0.0864] | [0.0553, 0.0906] | 0.140 | 0.151 | 0.158 |
| **lgbm/AB** | **0.2922** | **[0.2606, 0.3278]** | **[0.2315, 0.3511]** | **0.351** | **0.337** | **0.353** |

- **Val was optimistic for `lgbm/AB`, as expected:** 0.393 on val, 0.292 on test. Val tuned early stopping, the grid and the threshold, and it had no gap after train. On a 500-alert queue, about a third of alerts are real and about 35% of test laundering is caught.
- **The other models barely move between val and test** (`lgbm/A` 0.075 → 0.070, `lr/A` 0.0145 → 0.0146). Most of `lgbm/AB`'s drop is its own val optimism, and it is still the strongest model by far.
- **Attempt-cluster CIs are about 1.3–2× wider than iid** (lgbm/AB 1.8×), so transactions within an attempt are correlated and the locked iid CIs understate uncertainty. Every conclusion below holds under both.

### Paired differences (`test-paired.csv`, same rows, same resamples)

| Comparison | PR-AUC difference | iid 95% CI | Attempt-cluster 95% CI | Significant? |
|---|---|---|---|---|
| lgbm/AB − lr/AB | +0.266 | [0.235, 0.300] | [0.207, 0.320] | yes, both |
| lgbm/AB − lgbm/A | +0.222 | [0.192, 0.251] | [0.164, 0.272] | yes, both |

- **Behavior features help, and so does the tree model.** Both CIs exclude 0 by a wide margin.

### Recall by pattern at the val threshold (`test-recall-by-pattern.csv`; CIs for every model in `test-recall-by-pattern-ci.csv`)

| Pattern | n | is_ach | lr/A | lr/AB | lgbm/A | lgbm/AB [95% CI] |
|---|---|---|---|---|---|---|
| BIPARTITE | 44 | 1.000 | 0.523 | 0.159 | 0.386 | 0.364 [0.227, 0.500] |
| CYCLE | 55 | 1.000 | 0.382 | 0.164 | 0.273 | 0.418 [0.291, 0.545] |
| FAN-IN | 57 | 1.000 | 0.421 | 0.105 | 0.386 | 0.561 [0.421, 0.684] |
| FAN-OUT | 65 | 1.000 | 0.492 | 0.185 | 0.400 | 0.769 [0.646, 0.862] |
| GATHER-SCATTER | 127 | 1.000 | 0.346 | 0.181 | 0.307 | 0.654 [0.567, 0.740] |
| RANDOM | 43 | 1.000 | 0.488 | 0.233 | 0.256 | 0.302 [0.163, 0.442] |
| SCATTER-GATHER | 110 | 1.000 | 0.391 | 0.200 | 0.282 | 0.727 [0.645, 0.809] |
| STACK | 84 | 0.988 | 0.357 | 0.131 | 0.381 | 0.500 [0.393, 0.607] |
| **UNLISTED** | **371** | 0.674 | 0.094 | 0.070 | 0.008 | **0.024 [0.008, 0.040]** |

- **The same picture as val.** `lgbm/AB` is strongest on the fan and gather/scatter schemes (0.56–0.77), weaker on bipartite, random and cycles (0.30–0.42), and catches **9 of 371 UNLISTED (2.4%)**.
- **Why `lr/A` sometimes recalls more** (bipartite 0.52, random 0.49): its val threshold flags far more rows, so its recall at that threshold is high but its precision is very low (F1 0.034). PR-AUC is the fair comparison.
- **Per-pattern CIs are wide** (±0.1–0.15 on 43–127 rows), so recall by pattern is read as a direction, as `decisions.md` says.

## Tail (Sep 11–18; 1,108 rows, 655 laundering, 59%): reported separately, never pooled with test

| Model | PR-AUC | iid 95% CI | F1 @ val threshold |
|---|---|---|---|
| is_ach | 0.591 | [0.563, 0.620] | 0.743 |
| lr/A | 0.953 | [0.932, 0.972] | 0.588 |
| lr/AB | 0.956 | [0.939, 0.972] | 0.327 |
| lgbm/A | 0.974 | [0.957, 0.989] | 0.507 |
| lgbm/AB | 0.963 | [0.949, 0.976] | 0.818 |

- **The tail can't separate the models.** With 59% laundering, even the floor scores 0.59 and every model gets 0.95–0.97. The paired differences straddle 0 (`lgbm/AB − lgbm/A` −0.011 [−0.023, 0.002]). Precision and recall at 500/day are meaningless here: there are fewer than 500 rows per day, so every row is an alert. This confirms the T7 split decision to keep the tail out of headline metrics.

## Phase 2 conclusion

- **The Phase 3 baseline is `lgbm/AB`:** test PR-AUC 0.292 [0.261, 0.328], F1 0.351, about 34% precision and 35% recall at 500 alerts/day. Recorded in `docs/decisions.md`; scores frozen in `scores`.
- **What Phase 3 has to beat:**
  - **On headline PR-AUC:** the gap from A to AB is +0.22 and significant.
  - **On UNLISTED laundering:** the baseline catches 2.4%. Every listed scheme is ACH and the baseline has learned "ACH + new account + amount band", so this is the clearest test of whether graph features add something behavior can't.
