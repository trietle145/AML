import duckdb

from load import DB, EXPECTED_ROWS

SET_A = ["log_amount_usd", "payment_format", "payment_currency", "receiving_currency", "hour",
         "is_cross_bank", "is_currency_mismatch", "is_self_transfer"]


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
    con = duckdb.connect(str(DB))
    build_fx(con)
    build_a(con)
    check_a(con)
    print(f"features: {EXPECTED_ROWS:,} rows, set A = {len(SET_A)} columns")


if __name__ == "__main__":
    main()
