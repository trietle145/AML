from pathlib import Path

import duckdb

from split import CUTS

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data/raw/HI-Small_Trans.csv"
DB = ROOT / "data/aml.duckdb"

# from data/raw/README.md
EXPECTED_ROWS = 5_078_345
EXPECTED_LAUNDERING = 5_177

# explicit schema; banks/accounts stay text (leading zeros), the duplicate Account header is renamed
COLUMNS = {
    "timestamp": "TIMESTAMP",
    "from_bank": "VARCHAR",
    "from_account": "VARCHAR",
    "to_bank": "VARCHAR",
    "to_account": "VARCHAR",
    "amount_received": "DOUBLE",
    "receiving_currency": "VARCHAR",
    "amount_paid": "DOUBLE",
    "payment_currency": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "TINYINT",
}


def main():
    con = duckdb.connect(str(DB))
    # src_row = position in the CSV (0-based), relies on DuckDB preserving insertion order
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_transactions AS
        SELECT row_number() OVER () - 1 AS src_row, *
        FROM read_csv(?, header = true, columns = {COLUMNS}, timestampformat = '%Y/%m/%d %H:%M')
    """, [str(CSV)])
    rows, laundering = con.execute("SELECT COUNT(*), SUM(is_laundering) FROM raw_transactions").fetchone()
    print(f"raw_transactions: {rows:,} rows, {laundering:,} laundering")
    assert rows == EXPECTED_ROWS, rows
    assert laundering == EXPECTED_LAUNDERING, laundering

    con.execute((ROOT / "src/clean.sql").read_text().format(**CUTS))
    n, ids = con.execute("SELECT COUNT(*), COUNT(DISTINCT tx_id) FROM transactions").fetchone()
    assert n == ids == rows, (n, ids, rows)
    quality_summary(con, rows)


# no rows are dropped; this only reports what is there
def quality_summary(con, rows):
    nulls = con.execute("SELECT " + ", ".join(f"COUNT(*) - COUNT({c})" for c in COLUMNS) + " FROM raw_transactions").fetchone()
    print("nulls:", {c: v for c, v in zip(COLUMNS, nulls) if v} or "none")
    distinct = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT * EXCLUDE (src_row) FROM raw_transactions)").fetchone()[0]
    print(f"exact duplicate rows (beyond first copy): {rows - distinct:,}")
    for name, cond in [
        ("amount_paid <= 0", "amount_paid <= 0"),
        ("amount_received <= 0", "amount_received <= 0"),
        ("self-transfers", "is_self_transfer"),
        ("cross-bank", "is_cross_bank"),
        ("currency mismatch", "is_currency_mismatch"),
    ]:
        n, pos = con.execute(f"SELECT COUNT(*), COALESCE(SUM(is_laundering), 0) FROM transactions WHERE {cond}").fetchone()
        print(f"{name}: {n:,} rows, {pos:,} laundering")


if __name__ == "__main__":
    main()
