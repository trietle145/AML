# Follow the Money: AML detection

Transaction-level laundering detection on IBM AML **HI-Small** (synthetic). Headline question: do past-only graph features (fan-in/out, pass-through, short cycles) beat per-transaction + behavioral features? Answered by an ablation on a fixed time split.

- Why and scope: [docs/plan-review.md](docs/plan-review.md)
- Current phase, tasks, acceptance criteria: [docs/phase1-plan.md](docs/phase1-plan.md)
- Locked target, split dates, metrics: `docs/decisions.md` (written in Phase 1, Task 8). Once triet approves it, it's fixed; read from it, never re-decide.

## Workflow

- Coding (writing, fixing, refactoring, reviewing): invoke the `ponytail` skill first.
- Planning (new phase, task breakdown, re-planning): invoke the `agent-skills:plan` skill; save plans as `docs/phaseN-plan.md`.

## Data gotchas

- `data/raw/` is read-only. Everything downstream rebuilds from it via scripts; `data/` and `*.duckdb` stay out of git.
- `HI-Small_Trans.csv` header has `Account` twice: rename to `from_account` / `to_account`.
- Bank IDs have leading zeros (`010`, `021174`): load as VARCHAR with an explicit schema, never type-sniffed.
- Account key is `bank || ':' || account`. Account IDs are only unique within a bank.
- Timestamps are minute resolution, so ties are common. Features use only transactions strictly earlier than the scored one.
- Pattern types from `HI-Small_Patterns.txt` are evaluation-only (recall by pattern, plus an "unlisted" bucket), never features.
- Split is by time only. No random splits anywhere.
- 5M rows: aggregate in DuckDB (`data/aml.duckdb`), pull only summaries into pandas.
- `HI-Small_accounts.csv` (bank, account → entity) is unused in Phase 1.

## Environment

Windows, Python 3.13, uv. Plain scripts in `src/`, one EDA notebook in `notebooks/`. Each non-trivial script carries an `assert`-based self-check.
