import duckdb

from load import COLUMNS, DB, ROOT

PATTERNS = ROOT / "data/raw/HI-Small_Patterns.txt"
TMP_CSV = ROOT / "data/pattern_tx.csv"
JOIN_COLS = [c for c in COLUMNS if c != "is_laundering"]


# one CSV row per pattern transaction, prefixed with attempt_id and pattern_type
def parse(path):
    rows, attempt, ptype = [], -1, None
    for line in path.read_text().splitlines():
        if line.startswith("BEGIN"):
            attempt += 1
            ptype = line.split(" - ", 1)[1].split(":")[0].strip()
        elif line.startswith("END"):
            ptype = None
        elif line.strip():
            assert ptype, f"transaction outside a block: {line}"
            rows.append(f"{attempt},{ptype},{line}")
    return attempt + 1, rows


def main():
    attempts, rows = parse(PATTERNS)
    # reuse load.py's schema so amounts and timestamps parse exactly like raw_transactions
    TMP_CSV.write_text("\n".join(rows) + "\n")
    con = duckdb.connect(str(DB))
    cols = {"attempt_id": "INTEGER", "pattern_type": "VARCHAR", **COLUMNS}
    con.execute(f"""
        CREATE OR REPLACE TABLE pattern_tx AS
        SELECT row_number() OVER () - 1 AS pattern_row, *
        FROM read_csv(?, header = false, columns = {cols}, timestampformat = '%Y/%m/%d %H:%M')
    """, [str(TMP_CSV)])
    TMP_CSV.unlink()

    on = " AND ".join(f"p.{c} = r.{c}" for c in JOIN_COLS)
    con.execute(f"""
        CREATE OR REPLACE TABLE tx_pattern AS
        SELECT p.pattern_row, p.attempt_id, p.pattern_type, t.tx_id, t.is_laundering
        FROM pattern_tx p
        JOIN raw_transactions r ON {on}
        JOIN transactions t USING (src_row)
    """)

    print(f"{attempts} attempts, {len(rows):,} pattern rows")
    print(con.sql("""
        SELECT pattern_type, COUNT(DISTINCT attempt_id) AS attempts, COUNT(*) AS rows
        FROM pattern_tx GROUP BY 1 ORDER BY 1
    """))
    matched, ambiguous = con.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (n > 1)
        FROM (SELECT pattern_row, COUNT(*) n FROM tx_pattern GROUP BY 1)
    """).fetchone()
    print(f"pattern rows matched: {matched:,} / {len(rows):,} ({matched / len(rows):.1%}), ambiguous (>1 tx): {ambiguous}")
    shared = con.execute("SELECT COUNT(*) FROM (SELECT tx_id FROM tx_pattern GROUP BY 1 HAVING COUNT(DISTINCT attempt_id) > 1)").fetchone()[0]
    print(f"transactions in more than one attempt: {shared}")
    total, unlisted = con.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (tx_id NOT IN (SELECT tx_id FROM tx_pattern))
        FROM transactions WHERE is_laundering = 1
    """).fetchone()
    print(f"laundering transactions with no pattern (unlisted): {unlisted:,} / {total:,} ({unlisted / total:.1%})")

    bad = con.execute("SELECT COUNT(*) FROM tx_pattern WHERE is_laundering <> 1").fetchone()[0]
    assert bad == 0, f"{bad} matched transactions are not labelled laundering"
    assert attempts == 370, attempts


if __name__ == "__main__":
    main()
