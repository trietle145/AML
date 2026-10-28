import duckdb

# start of each split; train is everything before VAL_START. Chosen in docs/decisions.md
CUTS = {"val_start": "2022-09-07", "test_start": "2022-09-09", "tail_start": "2022-09-11"}
ORDER = ["train", "val", "test", "tail"]


def main():
    from load import DB  # imported here because load.py imports CUTS from this file

    con = duckdb.connect(str(DB), read_only=True)
    df = con.sql("""
        SELECT split, MIN(ts) first_ts, MAX(ts) last_ts, COUNT(*) n,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) pct_rows,
               SUM(is_laundering) pos, ROUND(100 * AVG(is_laundering), 3) pct_laundering
        FROM transactions GROUP BY 1
    """).df().set_index("split").loc[ORDER]
    print(df.to_string())
    print(con.sql("""
        SELECT COALESCE(p.pattern_type, 'UNLISTED') pattern_type, t.split, COUNT(*) n
        FROM transactions t LEFT JOIN tx_pattern p USING (tx_id)
        WHERE t.is_laundering = 1 GROUP BY 1, 2
    """).df().pivot(index="pattern_type", columns="split", values="n").fillna(0).astype(int)[ORDER].to_string())

    for a, b in zip(ORDER, ORDER[1:]):
        assert df.last_ts[a] < df.first_ts[b], f"{a} overlaps {b}"
    assert df.n.sum() == con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert (df.pos[["val", "test"]] >= 300).all(), "val/test need several hundred positives"


if __name__ == "__main__":
    main()
