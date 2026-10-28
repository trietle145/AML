# Phase 1 plan: Data and EDA

*Written 2026-10-28. Based on `docs/plan-review.md`. Planning only; nothing implemented yet.*

## Goal

Get IBM AML **HI-Small** into DuckDB, clean it, explore it, and **write down the time split, target and metrics before any feature work starts**. Phase 1 is done when `docs/decisions.md` exists and Phases 2–5 can read the split and metrics from one place instead of re-deciding them.

## Decisions this plan already assumes

- **Dataset:** HI-Small only (`HI-Small_Trans.csv` + `HI-Small_Patterns.txt`). LI-Small is a later robustness check, not Phase 1.
- **Engine:** DuckDB file at `data/aml.duckdb`, queried from Python. No server.
- **Code style:** plain Python scripts in `src/` for anything later phases reuse (load, clean, split, metrics); one notebook for EDA. No package, no framework.
- **Account key:** `bank || ':' || account`. Account IDs are only unique within a bank.
- **Raw data is never edited.** `data/` is git-ignored; everything is rebuilt from `data/raw/` by re-running the scripts.

## Dependency graph

```
T1 setup ─ T2 download ─ T3 load ─ T4 clean ─┬─ T5 patterns ─────────┐
                                             └─ T6 EDA ─ T7 split ───┴─ T8 decisions + metrics
```

T5 and T6 can run in parallel. Everything else is sequential.

---

## Task 1: Project setup (XS) — DONE 2026-10-28

**Description:** Minimal skeleton so later tasks have a place to live.

**Acceptance criteria:**
- [x] `requirements.txt` with `duckdb`, `pandas`, `pyarrow`, `matplotlib`, `scikit-learn`, `jupyter` (pinned)
- [x] `.gitignore` excludes `data/`, `.venv/`, `*.duckdb`
- [x] Folders: `data/raw/`, `src/`, `notebooks/`, `reports/figures/`; `git init` done

**Verification:** `pip install -r requirements.txt` succeeds in a fresh venv; `python -c "import duckdb; print(duckdb.__version__)"` prints a version.

**Dependencies:** None
**Files:** `requirements.txt`, `.gitignore`

## Task 2: Download raw data (XS, needs triet) — DONE 2026-10-28

*Files are in `data/raw/`: 5,078,345 transaction rows, 370 pattern attempts across 8 types. `HI-Small_accounts.csv` was also downloaded and is unused in Phase 1. README written: 5,177 laundering rows.*

**Description:** Fetch the two HI-Small files from Kaggle (`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`). Either via the Kaggle CLI with your API token (`kaggle datasets download ... -f HI-Small_Trans.csv`, same for `HI-Small_Patterns.txt`) or by hand from the website. Do **not** download the whole dataset; Large is ~16 GB.

**Acceptance criteria:**
- [x] `data/raw/HI-Small_Trans.csv` and `data/raw/HI-Small_Patterns.txt` exist
- [x] Row count recorded in `data/raw/README.md` (expect roughly 5.08M rows, roughly 5.2k laundering)

**Verification:** Count lines in the CSV; check the header has the 11 expected columns (Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering).

**Dependencies:** T1
**Files:** `data/raw/*` (not committed), `data/raw/README.md`

## Task 3: Load into DuckDB (S) — DONE 2026-10-28

**Description:** `src/load.py` reads the CSV into table `raw_transactions` with an **explicit schema** (no type sniffing; the duplicate `Account` header must be renamed to `from_account` / `to_account`).

**Acceptance criteria:**
- [x] `python src/load.py` builds `data/aml.duckdb` from scratch, idempotently (drop + recreate)
- [x] Row count in DuckDB equals CSV row count from T2
- [x] `SUM(is_laundering)` matches the expected count

**Verification:** Script prints and `assert`s both counts; rerunning gives the same numbers.

**Dependencies:** T2
**Files:** `src/load.py`

## Task 4: Clean into a `transactions` view (S) — DONE 2026-10-28

**Description:** One SQL view on top of `raw_transactions` that every later phase uses. Adds: `tx_id` (stable row number ordered by timestamp, then original row order), parsed `ts`, `from_acct` / `to_acct` keys (bank+account), `day`, `hour`, `is_self_transfer`, `is_cross_bank`, `is_currency_mismatch`, `amount_paid` and `amount_received` as doubles. Checks for nulls, exact duplicates, negative or zero amounts, and records what it finds.

**Acceptance criteria:**
- [x] `src/clean.sql` (run by `load.py`) creates the view; the view has the same row count as the raw table
- [x] A data-quality summary (nulls, duplicates, zero amounts, self-transfers) is printed and copied into `docs/decisions.md` later
- [x] Any row dropped is dropped for a written reason; default is to drop nothing

**Verification:** `SELECT COUNT(*), COUNT(DISTINCT tx_id) FROM transactions` gives equal numbers; spot-check 5 rows against the CSV.

**Dependencies:** T3
**Files:** `src/clean.sql`, `src/load.py`

**Findings (copy into decisions.md in T8):** no nulls; no zero or negative amounts; 9 exact duplicate pairs, all licit micro-amount Bitcoin/ACH transfers, kept as real repeat payments; nothing dropped. Self-transfers: 591,212 rows, 11 laundering. Cross-bank: 4,387,090 rows, 5,074 of 5,177 laundering. Currency mismatch: 72,170 rows, **0 laundering**, so it is a negative shortcut, not the positive one the review expected. Time range: 2022-09-01 00:00 to 2022-09-18 16:18.

## Checkpoint A (after T1–T4)
- [x] `python src/load.py` rebuilds the database from raw files in one command
- [x] Counts match the CSV; data-quality summary looks sane

---

## Task 5: Parse the patterns file (S) — DONE 2026-10-28

**Description:** `HI-Small_Patterns.txt` lists laundering attempts as blocks (`BEGIN LAUNDERING ATTEMPT - FAN-IN ...` … `END`), each followed by transaction lines in the same format as the CSV. Parse it into table `pattern_tx(attempt_id, pattern_type, ...)` and join to `transactions` on (timestamp, accounts, amount, currency, format) to get `tx_id → pattern_type`. **Evaluation only; never a feature.**

**Acceptance criteria:**
- [x] `src/patterns.py` loads the table; pattern types listed with counts
- [x] Match rate reported: share of pattern rows found in `transactions`, and share of laundering transactions that have **no** pattern (the "unlisted" bucket)
- [x] Ambiguous matches (one pattern row matching several transactions) counted, not silently dropped

**Verification:** Every matched transaction has `is_laundering = 1`; script `assert`s it.

**Dependencies:** T4
**Files:** `src/patterns.py`
**Findings:** 370 attempts and 3,209 pattern rows. All 3,209 match exactly one transaction: 100% match rate, 0 ambiguous, and no transaction belongs to two attempts. **1,968 of the 5,177 laundering transactions (38%) are unlisted**, so the unlisted bucket is the largest group in any recall-by-pattern breakdown. Tables: `pattern_tx` (the parsed rows) and `tx_pattern` (pattern_row, attempt_id, pattern_type, tx_id).

## Task 6: EDA notebook (M)

**Description:** `notebooks/01_eda.ipynb`, querying DuckDB directly. Each section ends with one or two sentences of findings. Figures saved to `reports/figures/`.

Sections:
1. **Volume over time:** transactions per day and per hour; first and last timestamps.
2. **Daily laundering rate:** count and rate per day. Look specifically at the **last days of the period**, which are known to be thin in volume with an unusually high laundering share; that decides T7.
3. **Amounts:** distributions (log scale) for licit vs laundering, per currency.
4. **Currencies:** counts, laundering rate per currency, rate of currency mismatch.
5. **Payment formats:** counts and laundering rate per format (the review flags ACH as a likely shortcut feature).
6. **Accounts:** number of accounts, in/out degree distribution, share of accounts ever involved in laundering.
7. **Patterns:** laundering count by pattern type including "unlisted" (uses T5).

**Acceptance criteria:**
- [ ] Notebook runs top to bottom from a fresh kernel without errors
- [ ] All 7 sections present, each with a written finding
- [ ] Shortcut-feature candidates (format, currency mismatch, cross-bank) named explicitly

**Verification:** "Restart and run all" passes; figures exist in `reports/figures/`.

**Dependencies:** T4 (T5 for section 7 only)
**Files:** `notebooks/01_eda.ipynb`, `reports/figures/*`

## Checkpoint B (after T5–T6)
- [ ] Review EDA findings with triet before fixing the split

---

## Task 7: Choose and implement the time split (S)

**Description:** Pick cut-offs for train / validation / test, roughly **60/20/20**, from the daily plot in T6. Default rule: split by **calendar day on transaction count**, not by laundering count, and decide explicitly what happens to the thin tail days (keep in test, or trim and say so). Implement as a `split` column (`train`/`val`/`test`) on the `transactions` view, driven by two constant dates in `src/split.py`.

**Acceptance criteria:**
- [ ] Cut-off timestamps are constants in one place, and the view uses them
- [ ] Table of rows, laundering count and rate per split; every split has enough positives to evaluate (aim for several hundred in val and test)
- [ ] No split overlaps in time: `max(ts)` of train < `min(ts)` of val < `min(ts)` of test

**Verification:** `src/split.py` `assert`s non-overlap and prints the per-split table.

**Dependencies:** T6
**Files:** `src/split.py`, `src/clean.sql`

## Task 8: Lock target and metrics (S)

**Description:** Write `docs/decisions.md` and a small `src/metrics.py` so Phases 2–4 all score the same way.

`docs/decisions.md` records:
- **Target:** `is_laundering` per transaction. Account-level label = account (bank+account) has ≥1 laundering transaction as sender or receiver in the evaluated period.
- **Split:** the dates from T7 and the per-split table.
- **Feature rule:** features may only use transactions with `ts` ≤ the scored transaction's `ts` (strictly earlier for same-timestamp ties). Pattern types never used as features.
- **Primary metrics:** PR-AUC and minority-class F1 (threshold chosen on validation, applied to test).
- **Business metric:** precision and recall at a fixed **alert budget**, with K picked from test-period daily volume (e.g. top 100 transactions/day).
- **Breakdown:** recall by pattern type, with an "unlisted" bucket.
- **Uncertainty:** 1,000× bootstrap on the test set, 95% CI; "significantly better" means the CIs of the difference exclude 0 (paired bootstrap).
- **Data-quality notes** from T4 and shortcut features from T6.

`src/metrics.py` has `pr_auc`, `best_f1_threshold`, `f1_at`, `precision_recall_at_k_per_day`, `recall_by_pattern`, `paired_bootstrap_ci`, plus an `assert`-based self-check on a toy example under `if __name__ == "__main__"`.

**Acceptance criteria:**
- [ ] `docs/decisions.md` covers every bullet above
- [ ] `python src/metrics.py` runs its self-check and passes
- [ ] A trivial scorer (random scores, and "is ACH") is run through `metrics.py` on val, giving the floor any Phase 2 model has to beat

**Verification:** Self-check passes; random-score PR-AUC ≈ the base rate.

**Dependencies:** T5, T7
**Files:** `docs/decisions.md`, `src/metrics.py`

## Checkpoint C: Phase 1 complete
- [ ] `python src/load.py && python src/patterns.py && python src/split.py` rebuilds everything from raw files
- [ ] EDA notebook runs clean
- [ ] `docs/decisions.md` reviewed and approved by triet; after this, split and metrics don't change

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Thin, laundering-heavy tail days distort the test set | High | Inspect in T6; decide explicitly in T7 and write the reason down |
| Pattern file rows don't match transactions exactly | Med | Match on all fields; report match rate and ambiguous matches (T5) |
| Same-timestamp transactions leak into "past-only" features later | Med | Tie rule written in `decisions.md` (T8) |
| 5M rows in pandas runs out of memory on a laptop | Low | Aggregate in DuckDB, only pull summaries into pandas |
| Kaggle download needs a login / API token | Low | Manual download is fine; only two files needed |

## Open questions for triet

1. Do you have a Kaggle API token set up, or will you download the two files by hand?
2. Which Python setup do you use on that machine (venv, conda, uv)? The plan assumes a plain venv.
3. Should the project be a git repo (and pushed to GitHub)? The plan assumes yes, local only for now.
