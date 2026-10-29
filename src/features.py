import duckdb

from load import DB, EXPECTED_ROWS

SET_A = ["log_amount_usd", "payment_format", "payment_currency", "receiving_currency", "hour",
         "is_cross_bank", "is_currency_mismatch", "is_self_transfer"]

# set B: each account's own past on one side only. out = sender's past sends, in = receiver's past receipts
SIDES = {"out": ("from_acct", "usd_paid"), "in": ("to_acct", "usd_recv")}
WINDOWS = {"1h": "1 HOUR", "24h": "24 HOUR", "3d": "3 DAY"}
STATS = ["cnt", "usd_sum", "usd_max", "amt_vs_mean", "ach_share"]
CAP_MINUTES = 3 * 24 * 60
SET_B = [f"{s}_{st}_{w}" for s in SIDES for w in WINDOWS for st in STATS] + \
        [f"{s}_{c}" for s in SIDES for c in ("min_since_last", "has_history")]
NULLABLE_B = [c for c in SET_B if "amt_vs_mean" in c or "ach_share" in c]


# usd_per_unit per currency, from train-split USD -> X conversions (tight); X -> USD only cross-checks
def build_fx(con):
    con.execute("""
        CREATE OR REPLACE TABLE fx_rates AS
        WITH pairs AS (
            SELECT payment_currency pay, receiving_currency rec, MEDIAN(amount_received / amount_paid) r
            FROM transactions
            WHERE split = 'train' AND is_currency_mismatch AND 'US Dollar' IN (payment_currency, receiving_currency)
            GROUP BY 1, 2
        )
        SELECT o.rec currency, 1 / o.r usd_per_unit, i.r usd_per_unit_check
        FROM pairs o JOIN pairs i ON o.rec = i.pay AND i.rec = 'US Dollar'
        WHERE o.pay = 'US Dollar'
        UNION ALL SELECT 'US Dollar', 1, 1
    """)
    fx = con.sql("SELECT * FROM fx_rates ORDER BY currency").df()
    print(fx.to_string(index=False))
    n_cur = con.execute("SELECT COUNT(DISTINCT payment_currency) FROM transactions").fetchall()[0][0]
    assert len(fx) == n_cur == 15, (len(fx), n_cur)
    off = (fx.usd_per_unit / fx.usd_per_unit_check - 1).abs()
    assert (off < 0.01).all(), fx[off >= 0.01]


def build_a(con):
    con.execute("""
        CREATE OR REPLACE TABLE features AS
        SELECT t.tx_id,
               log10(t.amount_paid * fx.usd_per_unit) log_amount_usd,
               t.payment_format, t.payment_currency, t.receiving_currency, t.hour,
               t.is_cross_bank::TINYINT is_cross_bank,
               t.is_currency_mismatch::TINYINT is_currency_mismatch,
               t.is_self_transfer::TINYINT is_self_transfer
        FROM transactions t JOIN fx_rates fx ON fx.currency = t.payment_currency
        ORDER BY t.tx_id
    """)


# joins the 34 set-B columns onto `features`: per side and window count, USD sum/max, amount vs own mean, ACH share; plus min_since_last, has_history
def build_b(con):
    inner, outer, windows = [], [], []
    for s, (acct, usd) in SIDES.items():
        for w, interval in WINDOWS.items():
            # frame ends 1 minute before the row: whole-minute timestamps, so same-minute rows never see each other
            windows.append(f"{s}_{w} AS (PARTITION BY {acct} ORDER BY ts "
                           f"RANGE BETWEEN INTERVAL {interval} PRECEDING AND INTERVAL 1 MINUTE PRECEDING)")
            inner += [f"COUNT(*) OVER {s}_{w} {s}_cnt_{w}", f"SUM({usd}) OVER {s}_{w} {s}_sum_{w}",
                      f"MAX({usd}) OVER {s}_{w} {s}_max_{w}", f"SUM(ach) OVER {s}_{w} {s}_ach_{w}"]
            outer += [f"{s}_cnt_{w}", f"COALESCE({s}_sum_{w}, 0) {s}_usd_sum_{w}", f"COALESCE({s}_max_{w}, 0) {s}_usd_max_{w}",
                      f"{usd} / NULLIF({s}_sum_{w} / {s}_cnt_{w}, 0) {s}_amt_vs_mean_{w}",
                      f"{s}_ach_{w} / NULLIF({s}_cnt_{w}, 0) {s}_ach_share_{w}"]
        # MAX(ts) over the 3d frame, never LAG: the previous row is usually in the same minute
        inner.append(f"MAX(ts) OVER {s}_3d {s}_last_ts")
        outer += [f"COALESCE(date_diff('minute', {s}_last_ts, ts), {CAP_MINUTES}) {s}_min_since_last",
                  f"({s}_cnt_3d > 0)::TINYINT {s}_has_history"]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE features_b AS
        WITH base AS (
            SELECT t.tx_id, t.ts, t.from_acct, t.to_acct, (t.payment_format = 'ACH')::INT ach,
                   t.amount_paid * fp.usd_per_unit usd_paid, t.amount_received * fr.usd_per_unit usd_recv
            FROM transactions t
            JOIN fx_rates fp ON fp.currency = t.payment_currency
            JOIN fx_rates fr ON fr.currency = t.receiving_currency
        ), w AS (
            SELECT tx_id, ts, usd_paid, usd_recv, {", ".join(inner)}
            FROM base WINDOW {", ".join(windows)}
        )
        SELECT tx_id, {", ".join(outer)} FROM w
    """)
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features JOIN features_b USING (tx_id) ORDER BY tx_id")


def check_b(con):
    n = con.execute("SELECT COUNT(*) FROM features").fetchall()[0][0]
    assert n == EXPECTED_ROWS, n
    nulls = con.execute("SELECT " + ", ".join(f"COUNT(*) - COUNT({c})" for c in SET_B) + " FROM features").fetchall()[0]
    bad = {c: v for c, v in zip(SET_B, nulls) if (v and c not in NULLABLE_B) or v == n}
    assert not bad, f"unexpected NULLs: {bad}"
    for s, (acct, _) in SIDES.items():
        # an account's first-ever minute on this side has no strictly-earlier history, even with same-minute peers
        leaked = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT f.{s}_cnt_3d c, f.{s}_min_since_last m FROM features f JOIN transactions t USING (tx_id)
                QUALIFY t.ts = MIN(t.ts) OVER (PARTITION BY t.{acct})
            ) WHERE c <> 0 OR m <> {CAP_MINUTES}
        """).fetchall()[0][0]
        assert leaked == 0, f"{s}: {leaked} first-minute rows see history"
        bad_order = con.execute(f"SELECT COUNT(*) FROM features WHERE NOT ({s}_cnt_1h <= {s}_cnt_24h AND {s}_cnt_24h <= {s}_cnt_3d)").fetchall()[0][0]
        assert bad_order == 0, f"{s}: window counts not nested"

    # time drift pushes every account past the train range; one busy hub only moves its own rows,
    # so count accounts, not rows (070:100428660 alone puts 2.3% of test rows above the train max of out_cnt_3d)
    acct = {c: SIDES[c.split("_")[0]][0] for c in SET_B}
    drift = con.sql(f"""
        WITH m AS (SELECT {", ".join(f"MAX(f.{c}) {c}" for c in SET_B)} FROM features f JOIN transactions t USING (tx_id) WHERE t.split = 'train')
        SELECT {", ".join(f"COUNT(DISTINCT CASE WHEN f.{c} > m.{c} THEN t.{acct[c]} END) / COUNT(DISTINCT t.{acct[c]}) {c}" for c in SET_B)}
        FROM features f JOIN transactions t USING (tx_id), m WHERE t.split = 'test'
    """).df().T[0].sort_values(ascending=False)
    print("share of test accounts with values above the train max (top 5):")
    print(drift.head(5).round(5).to_string())
    assert (drift < 0.01).all(), drift[drift >= 0.01]

    print(con.sql("""
        SELECT t.day, ROUND(AVG(out_cnt_3d), 1) out_cnt_3d, ROUND(AVG(in_cnt_3d), 1) in_cnt_3d,
               ROUND(AVG(out_has_history), 3) out_hist, ROUND(AVG(in_has_history), 3) in_hist,
               ROUND(AVG(out_min_since_last), 0) out_min_last
        FROM features f JOIN transactions t USING (tx_id) GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False))


def check_a(con):
    # the strictly-earlier window trick in T3 relies on whole-minute timestamps
    assert con.execute("SELECT COUNT(*) FROM transactions WHERE second(ts) <> 0").fetchall()[0][0] == 0
    n, ids = con.execute("SELECT COUNT(*), COUNT(DISTINCT tx_id) FROM features").fetchall()[0]
    assert n == ids == EXPECTED_ROWS, (n, ids)
    nulls = con.execute("SELECT " + ", ".join(f"COUNT(*) - COUNT({c})" for c in SET_A) + " FROM features").fetchall()[0]
    assert not any(nulls), dict(zip(SET_A, nulls))
    print(con.sql("""
        SELECT t.tx_id, t.amount_paid, t.payment_currency, round(10 ^ f.log_amount_usd, 2) usd
        FROM transactions t JOIN features f USING (tx_id)
        WHERE t.payment_currency IN ('Euro', 'Yen', 'Bitcoin') AND t.tx_id % 1000 = 7
        QUALIFY row_number() OVER (PARTITION BY t.payment_currency ORDER BY t.tx_id) = 1
    """).df().to_string(index=False))


def main():
    # below free RAM (~4 GB of 16) so DuckDB spills to disk instead of failing the allocation
    con = duckdb.connect(str(DB), config={"memory_limit": "3GB"})
    build_fx(con)
    build_a(con)
    check_a(con)
    build_b(con)
    check_b(con)
    print(f"features: {EXPECTED_ROWS:,} rows, set A = {len(SET_A)} columns, set B = {len(SET_B)} columns")


if __name__ == "__main__":
    main()
