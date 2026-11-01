# Decisions

*The single place Phases 2–5 read the split, target and metrics from. Approved by triet on 2026-10-28; locked from here on.*

## Target

- **Transaction level (what models predict):** `is_laundering` for each row of `transactions`.
- **Account level (for alerts and account metrics):** an account (`bank:account`) is positive for a period if it sends or receives at least one laundering transaction in that period. Account scores are aggregated from transaction scores (e.g. max per account per day), and the aggregation is stated wherever account metrics are reported.

## Feature rule

- A feature for a transaction may only use transactions with `ts` **strictly earlier** than its own. Same-minute rows don't see each other; 5.06M of 5.08M rows share a minute.
- Train, val, test and tail rows all compute features the same way, looking back over all earlier data regardless of split. A test row may see train-period history, which is realistic.
- **Never features:** pattern types or anything from `tx_pattern` (evaluation only), raw account or bank IDs (attempts cross the split cuts, so IDs would let a model memorize dirty accounts), `tx_id`, `src_row`, and `split`.

## Metrics

All live in `src/metrics.py`. Run `python src/metrics.py` for the self-check and the floors below.

| Metric | Function | Rule |
|---|---|---|
| PR-AUC (primary) | `pr_auc` | average precision |
| Minority-class F1 (primary) | `best_f1_threshold`, `f1_at` | threshold chosen on val, applied unchanged to test |
| Precision / recall at alert budget | `precision_recall_at_k_per_day` | top **K = 500 alerts per day**, totalled over the split's days; ties broken by a seeded shuffle |
| Recall by pattern | `recall_by_pattern` | recall at the val-chosen threshold, per pattern type plus `UNLISTED` |
| Uncertainty | `bootstrap_ci`, `paired_bootstrap_ci` | 1,000 resamples of the test set, 95% percentile CI |

- **Why K = 500:** test averages about 431k transactions and 478 laundering transactions per day. 500 alerts per day is about 0.12% of volume and roughly one day's positives, a realistic investigator queue where a near-perfect model could catch most of a day's laundering.
- **Account level: K = 800 account alerts per day** (approved by triet 2026-11-01). Each account-day is scored by the **max** score over the account's transactions that day, sent or received (`evaluate.py --accounts`). Same rule as the transaction budget: val has about 225k active accounts and 815 positive accounts per day, so 800 is roughly one day's positive accounts (0.36% of active accounts).
- **"Significantly better":** the paired-bootstrap 95% CI of the difference (model A minus model B, same resamples) excludes 0.
- **Reporting:** headline numbers come from `test`. `tail` is reported in its own column and never pooled with test. Test is scored once per final model; all tuning happens on val.
- **Runtime:** a bootstrap over about 860k rows takes roughly 0.3 s per resample, so 1,000× is a few minutes per model.

## Baseline floor (val)

| Scorer | PR-AUC [95% CI] | Best F1 | P@500/day | R@500/day |
|---|---|---|---|---|
| random | 0.0011 [0.0010, 0.0012] | 0.003 | 0.001 | 0.001 |
| is ACH | 0.0067 [0.0062, 0.0073] | 0.015 | 0.008 | 0.008 |

CIs come from 200 resamples, which is enough for a floor. The random PR-AUC matches the val base rate (0.107%), as it should. "Is ACH" flags 100% of every listed pattern type in val but only 61.5% of UNLISTED laundering: 3,208 of the 3,209 pattern transactions in the whole dataset are ACH. So a per-transaction model gets the listed patterns' payment format for free, and recall by pattern has to be read with that in mind.

## Data quality (from T4)

No nulls, no zero or negative amounts, and no rows dropped. 9 exact duplicate pairs exist; all are licit micro-amount Bitcoin/ACH transfers, kept as real repeat payments.

## Shortcut features (from T6)

ACH is strongly positive (87% of laundering). Wire, Reinvestment and currency mismatch have 0 laundering. Self-transfer has 11 laundering rows in 591k. Saudi Riyal is about 4× the base rate. Laundering amounts drop off sharply just under $20k. These are real signal in the generator, not leaks; the per-transaction baseline uses them, and the graph features have to beat that. Full table: `notebooks/01_eda.ipynb`, "Shortcut-feature candidates".

## Split

**Decision:** split by time on calendar days. The cut dates live in one place, `CUTS` in `src/split.py`; `load.py` writes them into the `split` column of the `transactions` view.

| Split | Days | Rows | % rows | Laundering | % laundering | Use |
|---|---|---|---|---|---|---|
| train | Sep 1–6 | 3,248,921 | 64.0 | 2,530 | 0.078 | fit models |
| val | Sep 7–8 | 965,524 | 19.0 | 1,036 | 0.107 | tune, pick thresholds |
| test | Sep 9–10 | 862,792 | 17.0 | 956 | 0.111 | headline metrics, scored once |
| tail | Sep 11–18 | 1,108 | 0.0 | 655 | 59.1 | reported separately, never in headline metrics |

Checked by `python src/split.py`: splits don't overlap in time (`max(ts)` of each split < `min(ts)` of the next), every row is in exactly one split, and val and test each have ≥ 300 positives.

### How the cut dates were chosen

1. **Time split, not k-fold.** A laundering attempt spans 1–8 days (median 1–6 depending on type) and 5–14 transactions. Random k-fold would put parts of the same attempt into both train and test, so a model could memorize the accounts instead of learning the behaviour. Features are also computed from past transactions, and random folds would put test-period data into a training row's "past". Rolling-origin CV is leak-free, but with 10 usable days each fold is tiny; the test-set bootstrap (T8) covers uncertainty instead.
2. **Cuts by transaction count, on day boundaries.** Cumulative volume reaches 64% at the end of Sep 6 and 83% at the end of Sep 8. That is the closest whole-day match to 60/20/20. Splitting by laundering count instead would tie the split to the label.
3. **The thin tail becomes its own split.** The generator produced normal activity and unlisted laundering for Sep 1–10 only. After Sep 10 it only lets attempts already under way finish: 109 attempts still have transactions on Sep 11–18, and 0 unlisted laundering occurs there. Those 1,108 rows are 59% laundering, against about 0.1% on every other day. Keeping them in test would inflate PR-AUC with easy, unrepresentative positives. Dropping them would lose the endings of 109 attempts, so they are kept as a separate `tail` split: scored and reported alongside test, never mixed into the headline numbers.
4. **Attempts crossing a cut are kept.** About 96 attempts span the train/val cut and 97 span the val/test cut. That is realistic, since you see the start of a scheme before it ends. As a consequence, raw account IDs must never be features, or the model learns which accounts are dirty.

### Laundering by pattern and split

| Pattern | train | val | test | tail |
|---|---|---|---|---|
| BIPARTITE | 148 | 49 | 44 | 22 |
| CYCLE | 122 | 66 | 55 | 44 |
| FAN-IN | 135 | 60 | 57 | 66 |
| FAN-OUT | 147 | 64 | 65 | 66 |
| GATHER-SCATTER | 239 | 96 | 127 | 254 |
| RANDOM | 69 | 38 | 43 | 41 |
| SCATTER-GATHER | 253 | 134 | 110 | 129 |
| STACK | 199 | 150 | 84 | 33 |
| UNLISTED | 1,218 | 379 | 371 | 0 |

Per-pattern test counts are small (43–127), so recall by pattern needs bootstrap CIs and is read as a direction, not a precise number.

### Known caveats

- **Warm-up:** Sep 1 has the least history for rolling features, and pattern laundering is low that day (94 rows) because attempts are only starting.
- **Weekday mix:** test (Sep 9–10) includes one low-volume day (Sep 10, about 208k rows), like train's Sep 3–4. Val (Sep 7–8) has none.

## Phase 3 baseline (from Phase 2, T7)

**`lgbm/AB` is the Phase 3 baseline**: LightGBM on feature sets A + B (`data/models/lgbm_AB.txt`; 63 leaves, learning rate 0.05, min_child_samples 500, 237 trees; val threshold 0.1652). Test PR-AUC 0.292 [0.261, 0.328] (attempt-cluster CI [0.232, 0.351]). Its per-row scores stay frozen in `scores` (`model = 'lgbm'`, `feature_set = 'AB'`) for Phase 3's paired bootstrap. Phase 3 must not retrain or overwrite it. Results: the README and `reports/test-metrics.csv`.

## Phase 4 base model (from Phase 3, T7; approved by triet 2026-10-31)

**`lgbm/ABC` is the Phase 4 base model**: LightGBM on feature sets A + B + C (`data/models/lgbm_ABC.txt`; 15 leaves, learning rate 0.05, min_child_samples 500, 160 trees; val threshold 0.1675). Test PR-AUC 0.345 [0.313, 0.381] (attempt-cluster CI [0.286, 0.400]), paired +0.053 over `lgbm/AB`; P / R @500/day 0.389 / 0.407. The GNN (0.375 ± 0.003) and MLP (0.369 ± 0.002) rank slightly better overall but tie or lose at 500/day, and message passing added nothing measurable over the MLP. `lgbm/ABC` was chosen for being as good at the operating point, explainable (TreeSHAP via `pred_contrib`) and stable. **No new neural networks are trained in Phase 4**; existing MLP scores may only enter an ensemble. Its `scores` rows and model file are never retrained or overwritten (new feature sets get new names, e.g. `lgbm/ABCD`). Results: the README and `reports/phase3-test-*.csv`.

## Look-ahead experiment (exception to the feature rule; approved by triet 2026-11-01)

Set **C+** (`SET_C_AHEAD`, columns `*_ahead`) is set C computed with look-ahead: every window spans the same interval back **and forward** (1h / 24h / 3d), and 3-cycles may close in either time order within ±3 days. Set B stays strictly past. It exists only to measure how much seeing future transactions is worth, mirroring IBM Multi-GNN's transductive protocol (and the past-only rescoring of IBM's model). `lgbm/ABC_ahead` is **never a headline or deployable model**: it is reported only in the comparison with published results, labelled as look-ahead. It lives on branch `lookahead-features`.
