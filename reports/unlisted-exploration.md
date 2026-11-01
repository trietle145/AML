# UNLISTED laundering: why a third of laundering is out of reach

*Train and val only; test was not read. Regenerate with `python -u src/evaluate.py --analyze-unlisted` (about 3 minutes). The models examined here are `lgbm/ABC` and an early GNN run (`gnn/ABC`, val PR-AUC 0.462); later models give the same picture.*

## Summary

- **Recall ceiling of about 63%.** Of val's 1,036 laundering rows, 379 (36.6%) belong to no listed scheme. Every model catches almost none of them: at the val alert threshold, `lgbm/ABC` flags 2 of 379 and the GNN 4; on non-ACH UNLISTED, 0 of 146 for both. So recall at useful precision stops at the listed share: 657 / 1,036 = **63.4%**.
- **UNLISTED is a separate population.** It has smaller amounts, less ACH, no self-transfers and new sender–receiver pairs. It shares accounts with listed schemes no more often than licit traffic does (1.5% vs 1.4%).
- **Repeat accounts are real but invisible to the models.** 34.6% of UNLISTED rows touch an account that laundered earlier (licit: 9.6%). But that comes from the labels, and the models never scored those earlier transactions high, so features built from past scores can't recover it.
- **Even a dedicated model barely separates UNLISTED.** LightGBM trained only on UNLISTED vs licit reaches val PR-AUC 0.0147: 37× the base rate, but tiny. On non-ACH UNLISTED it gets 0.0016.
- **The simulator explains why.** AMLworld labels criminals' ordinary-looking spending (payroll, supplies) and every downstream transfer of dirty money as laundering. Nothing in a single payment, or in its graph neighbourhood, marks it.

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

UNLISTED payment formats are ACH 64%, Cheque 17%, Credit Card 11%, Cash 6% and Bitcoin 3%. Listed schemes are 100% ACH, which is why "is ACH" alone finds every listed pattern but only 64% of UNLISTED. UNLISTED sits between licit and listed on every column, and no single feature separates it.

## 2. Links to listed schemes and to earlier laundering

| | Licit | **UNLISTED** | Listed |
|---|---|---|---|
| Shares an account with a listed scheme (train + val) | 1.4% | **1.5%** | 100% |
| ...and that scheme account appeared earlier | 0.6% | **0.8%** | 74% |
| An account had earlier laundering (any type) | 9.6% | **34.6%** | 74% |

UNLISTED shares accounts with schemes at the licit rate: these are separate actors. An UNLISTED row is 3.6× as likely as a licit row to touch an account that laundered before. That signal uses labels, though, which a model never sees at scoring time.

## 3. Do past model scores flag UNLISTED accounts?

For each val row, does its sender or receiver have an earlier transaction (strictly earlier, within 3 days) that the model scored "high"? "High" means either at the model's val alert threshold, or in the top 1% of val scores. Only val-period history is counted, because train-period scores are in-sample and would flatter the result.

| Model | High = | UNLISTED rows with a hit | Licit rows with a hit | Lift | UNLISTED share of hit rows |
|---|---|---|---|---|---|
| `lgbm/ABC` | alert threshold | 1 / 379 (0.3%) | 0.29% | 0.9× | 0.03% |
| GNN | alert threshold | 4 / 379 (1.1%) | 0.30% | 3.6× | 0.13% |
| `lgbm/ABC` | top 1% | 200 / 379 (53%) | 12.2% | 4.3× | 0.17% |
| GNN | top 1% | 109 / 379 (29%) | 4.0% | 7.2× | 0.28% |

- **At the alert threshold there is nothing to work with.** The accounts behind UNLISTED laundering were almost never alerted on before: their earlier laundering was mostly UNLISTED too, and the models miss that as well.
- **Below the threshold the lift is 4–7×,** but the hit set is 39k–119k val rows, against a queue of 500 alerts a day. At most 0.28% of hit rows are UNLISTED.

So "this account was alerted recently" features can't lift UNLISTED; they were not built.

## 4. Upper bound: a model trained only on UNLISTED

LightGBM on A + B + C (15 leaves, learning rate 0.05, min_child_samples 500, early stopping on val), trained on UNLISTED vs licit with listed positives dropped:

| Val rows | Positives | PR-AUC | Base rate | Lift |
|---|---|---|---|---|
| All | 379 | 0.0147 | 0.0004 | 37× |
| Non-ACH | 146 | 0.0016 | 0.00017 | 9× |

Top features by gain: payment format 13%, the sender's largest and total recent amounts (8% each), the sender's recent counts (7% each), pair history 7%. The model mostly relearns "ACH, larger than the sender's usual, new pair". Without ACH it is close to noise. Early stopping on val makes this slightly optimistic, which is fine for an upper bound.

## What happened next

- **Amount-vs-own-usual features (set D),** the last candidate the profile suggested, barely moved UNLISTED (5 → 8 of 379 rows caught at 500 alerts/day on val) and didn't improve the model overall. They were dropped.
- **Look-ahead doesn't help:** even models allowed to see future transactions (branch `lookahead-features`, and IBM's Multi-GNN) catch about 2% of UNLISTED.
- **The ~63% recall ceiling stands:** about 37% of laundering is ordinary-looking spending of dirty money that strictly-past transaction data can't separate.

Thresholds and the top-1% cut come from val; this is a diagnosis, not model selection.
