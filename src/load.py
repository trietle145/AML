from pathlib import Path

import duckdb

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


if __name__ == "__main__":
    main()
