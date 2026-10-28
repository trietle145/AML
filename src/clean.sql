-- ponytail: view recomputes the tx_id sort on every query (~1s); materialize as a table if feature queries get slow
CREATE OR REPLACE VIEW transactions AS
SELECT
    row_number() OVER (ORDER BY timestamp, src_row) - 1 AS tx_id,
    src_row,
    timestamp AS ts,
    CAST(timestamp AS DATE) AS day,
    hour(timestamp) AS hour,
    from_bank || ':' || from_account AS from_acct,
    to_bank || ':' || to_account AS to_acct,
    from_bank,
    to_bank,
    amount_paid,
    payment_currency,
    amount_received,
    receiving_currency,
    payment_format,
    from_acct = to_acct AS is_self_transfer,
    from_bank <> to_bank AS is_cross_bank,
    payment_currency <> receiving_currency AS is_currency_mismatch,
    is_laundering,
    -- cut dates come from src/split.py (CUTS), filled in by load.py
    CASE
        WHEN timestamp < TIMESTAMP '{val_start}' THEN 'train'
        WHEN timestamp < TIMESTAMP '{test_start}' THEN 'val'
        WHEN timestamp < TIMESTAMP '{tail_start}' THEN 'test'
        ELSE 'tail'
    END AS split
FROM raw_transactions;
