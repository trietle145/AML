# Features

62 columns in three sets, all built in SQL (DuckDB) by [`src/features.py`](../src/features.py) into one `features` table, one row per transaction. The models use A, A+B or A+B+C. The rules behind them are in [`decisions.md`](decisions.md).

## The rules every feature follows

- **Strictly past.** A feature may only use transactions from strictly earlier minutes. 5.06M of the 5.08M rows share their minute with another row, so every window ends **one minute before** the scored row (`RANGE BETWEEN … PRECEDING AND INTERVAL 1 MINUTE PRECEDING`). "Last seen" is computed as `MAX(ts)` over that window, never with `LAG`, which would pick a same-minute row.
- **Nothing that grows with absolute time.** All accounts appear from Sep 1, so "time since first seen" or cumulative counts would drift between train and test. Windows are at most 3 days.
- **No identities, no labels.** No account or bank IDs, no scheme labels, no row IDs.
- **USD everywhere.** Amounts are converted with exchange rates measured on training-period USD↔X transfers (`fx_rates`, 15 currencies, cross-checked within 1%).

## Set A: the transaction itself (8 columns)

| Column | Meaning |
|---|---|
| `log_amount_usd` | log10 of the amount paid, in USD |
| `payment_format` | ACH, Cheque, Credit Card, Cash, Bitcoin, Wire, Reinvestment (categorical) |
| `payment_currency`, `receiving_currency` | categorical |
| `hour` | hour of day |
| `is_cross_bank`, `is_currency_mismatch`, `is_self_transfer` | 0/1 flags |

## Set B: each account's own past, one side only (34 columns)

For the **sender's past sends** (`out_`) and the **receiver's past receipts** (`in_`), over **1 hour, 24 hours and 3 days**:

| Column pattern | Meaning |
|---|---|
| `{out,in}_cnt_{w}` | number of transactions |
| `{out,in}_usd_sum_{w}`, `{out,in}_usd_max_{w}` | total and largest USD amount |
| `{out,in}_amt_vs_mean_{w}` | this amount ÷ the account's mean in the window (empty without history) |
| `{out,in}_ach_share_{w}` | share of those transactions paid by ACH |

Plus `{out,in}_min_since_last` (minutes since the account's previous transaction on that side, capped at 3 days) and `{out,in}_has_history` (any activity in 3 days).

**Why it matters:** rows where the sender has no past sends or the receiver no past receipts are about 2% of rows but over half of laundering. That's legal to use, because it's strictly past, and it drives most of the jump from A to A+B (test PR-AUC 0.070 → 0.292).

## Set C: the graph around the transaction (20 columns)

Anything about counterparties, money flowing *through* an account, the pair's history or cycles is in C, never in B.

| Column | Meaning |
|---|---|
| `{out,in}_cpty_{1h,24h,3d}` | **distinct** counterparties: whom the sender paid (fan-out), who paid the receiver (fan-in) |
| `out_recv_{cnt,usd}_{1h,24h}` | what the **sender received** just before sending (pass-through) |
| `in_send_{cnt,usd}_{1h,24h}` | what the **receiver sent on** before this payment |
| `out_pass_ratio_24h`, `in_pass_ratio_24h` | USD sent ÷ USD received in 24h, per account |
| `pair_cnt_3d`, `pair_min_since_last` | how often and how recently this sender paid this receiver |
| `pair_rev_cnt_3d` | how often the receiver paid the sender (money coming back: a 2-cycle) |
| `cyc3_cnt_3d` | **time-ordered 3-cycles** closed by this payment: middle accounts w with v → w, then w → u, both earlier and within 3 days |

**What it adds:** +0.053 test PR-AUC on top of A+B (significant). The biggest single feature in `lgbm/ABC` is `in_cpty_3d` (fan-in, 37% of gain), and fan-in recall jumps from 0.56 to 0.86.

## How the features are checked

- **Brute-force leakage check** (`features.py --check`): about 300 stress-test rows (hub accounts, same-minute ties, split boundaries, laundering, cycles) are recomputed row by row in pandas from raw transactions, strictly earlier only. Every B and C column must match exactly (0 mismatched cells).
- **The check must be able to fail:** deliberately leaky versions are built and must be caught. These are windows ending at the current row (so they see same-minute rows), `LAG` instead of a time window, and 3-cycles that ignore time order.
- **Drift:** for every column, under 1% of test accounts may exceed the training maximum.
- **Sanity asserts:** no unexpected empty values, nested windows (1h ≤ 24h ≤ 3d), distinct counts ≤ totals, and an account's first-ever minute sees no history.

## Tried and not kept

- **Set D, amount vs the account's own usual** (z-score, vs median, vs max; 12 columns). Built and leak-checked, but no gain: −0.005 val PR-AUC, not significant, because set B's `amt_vs_mean` already held that signal. Archived on branch `feature-set-d`.
- **Longer cycles (4–10 hops), bicliques and scatter-gather counts:** dropped before building, after a quick check of how many val scheme rows they could ever fire on. Long cycles reached 2 of 66 cycle rows; bicliques reached 0 of 49 bipartite and 4 of 150 stack rows. A strictly-past feature can only fire on the payment that *completes* a structure, and most schemes take about 3 days while val and test are 2 days each.
- **Set C+, set C with look-ahead** (windows reaching forward as well): an experiment that breaks the strictly-past rule on purpose, to measure what seeing the future is worth. It is never a real model. See branch `lookahead-features`.
