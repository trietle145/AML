# Decisions

*The single place Phases 2–5 read the split, target and metrics from. After triet approves it, nothing here changes. Target, metrics and the feature rule are added in T8.*

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
