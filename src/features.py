import sys

import duckdb
import numpy as np
import pandas as pd

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

# set C: the graph around the transaction, strictly past. out_/in_ = sender/receiver side, as in B
PASS_WINDOWS = {"1h": "1 HOUR", "24h": "24 HOUR"}
SET_C = [f"{s}_cpty_{w}" for s in SIDES for w in WINDOWS] + \
        [f"out_recv_{st}_{w}" for w in PASS_WINDOWS for st in ("cnt", "usd")] + ["out_pass_ratio_24h"] + \
        [f"in_send_{st}_{w}" for w in PASS_WINDOWS for st in ("cnt", "usd")] + ["in_pass_ratio_24h"] + \
        ["pair_cnt_3d", "pair_rev_cnt_3d", "pair_min_since_last", "cyc3_cnt_3d"]
NULLABLE_C = ["out_pass_ratio_24h", "in_pass_ratio_24h"]


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


# one row per transaction with the USD amounts and ACH flag the set-B windows aggregate
BASE = """
    SELECT t.tx_id, t.ts, t.day, t.from_acct, t.to_acct, t.is_laundering, (t.payment_format = 'ACH')::INT ach,
           t.amount_paid * fp.usd_per_unit usd_paid, t.amount_received * fr.usd_per_unit usd_recv
    FROM transactions t
    JOIN fx_rates fp ON fp.currency = t.payment_currency
    JOIN fx_rates fr ON fr.currency = t.receiving_currency
"""


# SELECT of tx_id + SET_B; leak="frame" (frame ends at CURRENT ROW) or "lag" are deliberately wrong, for the leakage check
def b_query(leak=None):
    end = "CURRENT ROW" if leak == "frame" else "INTERVAL 1 MINUTE PRECEDING"
    inner, outer, windows = [], [], []
    for s, (acct, usd) in SIDES.items():
        for w, interval in WINDOWS.items():
            # frame ends 1 minute before the row: whole-minute timestamps, so same-minute rows never see each other
            windows.append(f"{s}_{w} AS (PARTITION BY {acct} ORDER BY ts RANGE BETWEEN INTERVAL {interval} PRECEDING AND {end})")
            inner += [f"COUNT(*) OVER {s}_{w} {s}_cnt_{w}", f"SUM({usd}) OVER {s}_{w} {s}_sum_{w}",
                      f"MAX({usd}) OVER {s}_{w} {s}_max_{w}", f"SUM(ach) OVER {s}_{w} {s}_ach_{w}"]
            outer += [f"{s}_cnt_{w}", f"COALESCE({s}_sum_{w}, 0) {s}_usd_sum_{w}", f"COALESCE({s}_max_{w}, 0) {s}_usd_max_{w}",
                      f"{usd} / NULLIF({s}_sum_{w} / {s}_cnt_{w}, 0) {s}_amt_vs_mean_{w}",
                      f"{s}_ach_{w} / NULLIF({s}_cnt_{w}, 0) {s}_ach_share_{w}"]
        # MAX(ts) over the 3d frame, never LAG: the previous row is usually in the same minute
        last = f"LAG(ts) OVER (PARTITION BY {acct} ORDER BY ts)" if leak == "lag" else f"MAX(ts) OVER {s}_3d"
        inner.append(f"{last} {s}_last_ts")
        outer += [f"COALESCE(date_diff('minute', {s}_last_ts, ts), {CAP_MINUTES}) {s}_min_since_last",
                  f"({s}_cnt_3d > 0)::TINYINT {s}_has_history"]
    return f"""
        WITH base AS ({BASE}), w AS (
            SELECT tx_id, ts, usd_paid, usd_recv, {", ".join(inner)}
            FROM base WINDOW {", ".join(windows)}
        )
        SELECT tx_id, {", ".join(outer)} FROM w
    """


# joins the 34 set-B columns onto `features`: per side and window count, USD sum/max, amount vs own mean, ACH share; plus min_since_last, has_history
def build_b(con):
    con.execute(f"CREATE OR REPLACE TEMP TABLE features_b AS {b_query()}")
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features JOIN features_b USING (tx_id) ORDER BY tx_id")


# builds set C step by step into temp tables, returns the SELECT of tx_id + SET_C; leak="frame" (frames end at CURRENT ROW) or "cycle" (3-cycles ignore time order) are deliberately wrong
def c_query(con, leak=None, where=None):
    end = "CURRENT ROW" if leak == "frame" else "INTERVAL 1 MINUTE PRECEDING"
    cpty, cpty_w = [], []
    for s, (acct, _) in SIDES.items():
        other = "to_acct" if acct == "from_acct" else "from_acct"
        for w, interval in WINDOWS.items():
            cpty_w.append(f"{s}_{w} AS (PARTITION BY {acct} ORDER BY ts RANGE BETWEEN INTERVAL {interval} PRECEDING AND {end})")
            cpty.append(f"COUNT(DISTINCT {other}) OVER {s}_{w} {s}_cpty_{w}")
    # every transaction twice: as a send of from_acct and as a receipt of to_acct, so one window sees an account's other side
    flow, flow_w, pick = [], [], []
    for w, interval in PASS_WINDOWS.items():
        flow_w.append(f"f_{w} AS (PARTITION BY acct ORDER BY ts RANGE BETWEEN INTERVAL {interval} PRECEDING AND {end})")
        for d in ("in", "out"):
            flow += [f"COUNT(*) FILTER (WHERE side = '{d}') OVER f_{w} {d}_cnt_{w}",
                     f"COALESCE(SUM(usd) FILTER (WHERE side = '{d}') OVER f_{w}, 0) {d}_usd_{w}"]
        # sender view (side out) takes its receipts; receiver view (side in) takes its sends
        pick += [f"MAX(CASE WHEN side = 'out' THEN in_cnt_{w} END) out_recv_cnt_{w}",
                 f"MAX(CASE WHEN side = 'out' THEN in_usd_{w} END) out_recv_usd_{w}",
                 f"MAX(CASE WHEN side = 'in' THEN out_cnt_{w} END) in_send_cnt_{w}",
                 f"MAX(CASE WHEN side = 'in' THEN out_usd_{w} END) in_send_usd_{w}"]
    pick += ["MAX(CASE WHEN side = 'out' THEN out_usd_24h / NULLIF(in_usd_24h, 0) END) out_pass_ratio_24h",
             "MAX(CASE WHEN side = 'in' THEN out_usd_24h / NULLIF(in_usd_24h, 0) END) in_pass_ratio_24h"]
    # 3-cycle closing u -> v: some w with v -> w then w -> u, both inside the 3d before ts. Joined from u's past
    # receipts (in-degree <= ~1,084); ASOF finds the latest v -> w before w -> u
    if leak == "cycle":
        cyc = """SELECT c.tx_id, COUNT(DISTINCT c.w) n FROM (
                     SELECT t.tx_id, t.ts, t.to_acct v, e2.from_acct w FROM base t JOIN base e2 ON e2.to_acct = t.from_acct
                     WHERE e2.ts BETWEEN t.ts - INTERVAL 3 DAY AND t.ts + INTERVAL 3 DAY AND e2.from_acct NOT IN (t.from_acct, t.to_acct)) c
                 JOIN base e1 ON e1.from_acct = c.v AND e1.to_acct = c.w AND e1.ts BETWEEN c.ts - INTERVAL 3 DAY AND c.ts + INTERVAL 3 DAY
                 GROUP BY 1"""
    else:
        cyc = """SELECT c.tx_id, COUNT(DISTINCT c.w) n FROM (
                     SELECT t.tx_id, t.ts, t.to_acct v, e2.from_acct w, e2.ts t2 FROM base t JOIN base e2 ON e2.to_acct = t.from_acct
                     WHERE e2.ts >= t.ts - INTERVAL 3 DAY AND e2.ts < t.ts AND e2.from_acct NOT IN (t.from_acct, t.to_acct)) c
                 ASOF JOIN base e1 ON e1.from_acct = c.v AND e1.to_acct = c.w AND e1.ts < c.t2
                 WHERE e1.ts >= c.ts - INTERVAL 3 DAY GROUP BY 1"""
    # one temp table per step: run as a single query, the windows, the 10M-row stream and the cycle join don't fit in RAM together
    steps = {
        "base": f"{BASE} {'WHERE ' + where if where else ''}",
        "cpty": f"SELECT tx_id, {', '.join(cpty)} FROM base WINDOW {', '.join(cpty_w)}",
        "pass": f"""WITH stream AS (SELECT tx_id, ts, from_acct acct, 'out' side, usd_paid usd FROM base
                                    UNION ALL SELECT tx_id, ts, to_acct, 'in', usd_recv FROM base),
                    flow AS (SELECT tx_id, side, {', '.join(flow)} FROM stream WINDOW {', '.join(flow_w)})
                    SELECT tx_id, {', '.join(pick)} FROM flow GROUP BY tx_id""",
        # one window per unordered pair; fwd says which direction this row goes (self-transfers: fwd false, no reverse)
        "pair": f"""WITH p AS (
                        SELECT tx_id, ts, from_acct < to_acct fwd, from_acct = to_acct from_self,
                               COUNT(*) FILTER (WHERE from_acct < to_acct) OVER pw c_fwd, COUNT(*) FILTER (WHERE from_acct >= to_acct) OVER pw c_bwd,
                               MAX(ts) FILTER (WHERE from_acct < to_acct) OVER pw l_fwd, MAX(ts) FILTER (WHERE from_acct >= to_acct) OVER pw l_bwd
                        FROM base WINDOW pw AS (PARTITION BY least(from_acct, to_acct), greatest(from_acct, to_acct) ORDER BY ts
                                                RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND {end}))
                    SELECT tx_id, CASE WHEN fwd THEN c_fwd ELSE c_bwd END pair_cnt_3d,
                           CASE WHEN from_self THEN 0 WHEN fwd THEN c_bwd ELSE c_fwd END pair_rev_cnt_3d,
                           COALESCE(date_diff('minute', CASE WHEN fwd THEN l_fwd ELSE l_bwd END, ts), {CAP_MINUTES}) pair_min_since_last
                    FROM p""",
        "cyc": cyc,
    }
    for name, sql in steps.items():
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {sql}")
    return f"""SELECT tx_id, {', '.join(SET_C[:-1])}, COALESCE(cyc.n, 0) cyc3_cnt_3d
               FROM cpty JOIN pass USING (tx_id) JOIN pair USING (tx_id) LEFT JOIN cyc USING (tx_id)"""


# joins the 20 set-C columns onto `features`: distinct counterparties, pass-through, pair history, 3-cycles
def build_c(con):
    con.execute(f"CREATE OR REPLACE TEMP TABLE features_c AS {c_query(con)}")
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features JOIN features_c USING (tx_id) ORDER BY tx_id")


# ~300 rows that stress the windows: random, hub accounts, same-minute ties, Sep 1, split boundaries, laundering
SAMPLE = f"""
    WITH b AS ({BASE}),
    ties AS (SELECT tx_id, COUNT(*) OVER (PARTITION BY from_acct, ts) t_out, COUNT(*) OVER (PARTITION BY to_acct, ts) t_in FROM b),
    hubs AS ((SELECT from_acct a FROM b GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 3)
             UNION ALL (SELECT to_acct FROM b GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 3)),
    picks AS (
        (SELECT tx_id FROM b ORDER BY hash(tx_id) LIMIT 60)
        UNION (SELECT tx_id FROM b WHERE from_acct IN (SELECT a FROM hubs) ORDER BY hash(tx_id) LIMIT 30)
        UNION (SELECT tx_id FROM b WHERE to_acct IN (SELECT a FROM hubs) ORDER BY hash(tx_id) LIMIT 30)
        UNION (SELECT tx_id FROM ties WHERE t_out > 1 ORDER BY hash(tx_id) LIMIT 30)
        UNION (SELECT tx_id FROM ties WHERE t_in > 1 ORDER BY hash(tx_id) LIMIT 30)
        UNION (SELECT tx_id FROM b WHERE day = DATE '2022-09-01' ORDER BY hash(tx_id) LIMIT 30)
        UNION (SELECT tx_id FROM b WHERE ts BETWEEN TIMESTAMP '2022-09-06 22:00' AND TIMESTAMP '2022-09-07 02:00'
                  OR ts BETWEEN TIMESTAMP '2022-09-08 22:00' AND TIMESTAMP '2022-09-09 02:00' ORDER BY hash(tx_id) LIMIT 40)
        UNION (SELECT tx_id FROM b WHERE is_laundering = 1 ORDER BY hash(tx_id) LIMIT 50)
        UNION (SELECT tx_id FROM tx_pattern WHERE pattern_type = 'CYCLE' ORDER BY hash(tx_id) LIMIT 40)
    )
    SELECT * FROM b WHERE tx_id IN (SELECT tx_id FROM picks)
"""


# set B recomputed row by row in pandas from raw rows with ts strictly earlier: the independent reference
def brute_force(con, sample):
    out = sample[["tx_id"]].copy()
    for s, (acct, usd) in SIDES.items():
        hist = con.sql(f"SELECT b.{acct} acct, b.ts, b.{usd} usd, b.ach FROM ({BASE}) b WHERE b.{acct} IN (SELECT {acct} FROM sample)").df()
        groups = {a: (g.ts.to_numpy(), g.usd.to_numpy(), g.ach.to_numpy()) for a, g in hist.groupby("acct")}
        cols = {}
        for r in sample.itertuples():
            ts, amt, ach = groups[getattr(r, acct)]
            now = np.datetime64(r.ts)
            for w, delta in {"1h": np.timedelta64(1, "h"), "24h": np.timedelta64(24, "h"), "3d": np.timedelta64(3, "D")}.items():
                m = (ts < now) & (ts >= now - delta)
                n, total = m.sum(), amt[m].sum()
                mean = total / n if n else np.nan
                vals = {"cnt": n, "usd_sum": total, "usd_max": amt[m].max() if n else 0,
                        "amt_vs_mean": getattr(r, usd) / mean if n and mean else np.nan, "ach_share": ach[m].sum() / n if n else np.nan}
                for st, v in vals.items():
                    cols.setdefault(f"{s}_{st}_{w}", []).append(v)
            m3 = (ts < now) & (ts >= now - np.timedelta64(3, "D"))
            cols.setdefault(f"{s}_min_since_last", []).append((now - ts[m3].max()) // np.timedelta64(1, "m") if m3.any() else CAP_MINUTES)
            cols.setdefault(f"{s}_has_history", []).append(int(m3.any()))
        out = out.assign(**cols)
    return out


# set C recomputed row by row in pandas from the sampled accounts' raw rows with ts strictly earlier
def brute_force_c(con, sample):
    accts = pd.DataFrame({"a": sorted(set(sample.from_acct) | set(sample.to_acct))})
    hist = con.sql(f"SELECT * FROM ({BASE}) b WHERE b.from_acct IN (SELECT a FROM accts) OR b.to_acct IN (SELECT a FROM accts)").df()
    sends = {a: g for a, g in hist.groupby("from_acct")}
    recvs = {a: g for a, g in hist.groupby("to_acct")}
    empty = hist.iloc[:0]
    deltas = {w: np.timedelta64(int(i.split()[0]), "h" if "HOUR" in i else "D") for w, i in WINDOWS.items()}
    rows = []
    for r in sample.itertuples():
        now, u, v = np.datetime64(r.ts), r.from_acct, r.to_acct
        win = lambda g, w: g[(g.ts.to_numpy() < now) & (g.ts.to_numpy() >= now - deltas[w])]
        su, rv, ru, sv = (d.get(a, empty) for d, a in ((sends, u), (recvs, v), (recvs, u), (sends, v)))
        row = {"tx_id": r.tx_id}
        for w in WINDOWS:
            row[f"out_cpty_{w}"], row[f"in_cpty_{w}"] = win(su, w).to_acct.nunique(), win(rv, w).from_acct.nunique()
        for w in PASS_WINDOWS:
            row[f"out_recv_cnt_{w}"], row[f"out_recv_usd_{w}"] = len(win(ru, w)), win(ru, w).usd_recv.sum()
            row[f"in_send_cnt_{w}"], row[f"in_send_usd_{w}"] = len(win(sv, w)), win(sv, w).usd_paid.sum()
        ratio = lambda out, inn: out / inn if inn else np.nan
        row["out_pass_ratio_24h"] = ratio(win(su, "24h").usd_paid.sum(), win(ru, "24h").usd_recv.sum())
        row["in_pass_ratio_24h"] = ratio(win(sv, "24h").usd_paid.sum(), win(rv, "24h").usd_recv.sum())
        same, rev = win(su, "3d"), win(sv, "3d")
        same, rev = same[same.to_acct == v], rev[rev.to_acct == u]
        row["pair_cnt_3d"], row["pair_rev_cnt_3d"] = len(same), 0 if u == v else len(rev)
        row["pair_min_since_last"] = (now - same.ts.max().to_datetime64()) // np.timedelta64(1, "m") if len(same) else CAP_MINUTES
        # w -> u in the past 3d, and an earlier v -> w still inside the 3d
        legs, sv3 = win(ru, "3d"), win(sv, "3d")
        legs = legs[~legs.from_acct.isin([u, v])]
        row["cyc3_cnt_3d"] = len({w for w, t2 in zip(legs.from_acct, legs.ts) if ((sv3.to_acct == w) & (sv3.ts < t2)).any()})
        rows.append(row)
    return pd.DataFrame(rows)


# number of (row, column) cells where got differs from the brute-force reference
def mismatches(got, ref, cols=SET_B):
    m = ref.merge(got, on="tx_id", suffixes=("_ref", ""))
    return {c: int((~np.isclose(m[c].astype(float), m[c + "_ref"].astype(float), rtol=1e-6, equal_nan=True)).sum()) for c in cols}


def leakage_check(con):
    sample = con.sql(SAMPLE).df()
    ref = brute_force(con, sample)
    real = mismatches(con.sql(f"SELECT tx_id, {', '.join(SET_B)} FROM features WHERE tx_id IN (SELECT tx_id FROM sample)").df(), ref)
    print(f"leakage check on {len(sample)} rows ({int(sample.is_laundering.sum())} laundering): {sum(real.values())} mismatched cells")
    assert not any(real.values()), {c: v for c, v in real.items() if v}
    # the check must be able to fail: both deliberately leaky variants have to be caught
    for leak in ("frame", "lag"):
        got = con.sql(f"SELECT * FROM ({b_query(leak)}) WHERE tx_id IN (SELECT tx_id FROM sample)").df()
        bad = mismatches(got, ref)
        print(f"  leaky variant '{leak}': {sum(bad.values())} mismatched cells in {sum(1 for v in bad.values() if v)} columns")
        assert sum(bad.values()) > 0, f"check failed to catch the '{leak}' leak"

    ref = brute_force_c(con, sample)
    real = mismatches(con.sql(f"SELECT tx_id, {', '.join(SET_C)} FROM features WHERE tx_id IN (SELECT tx_id FROM sample)").df(), ref, SET_C)
    print(f"set C leakage check on the same rows ({int((ref.cyc3_cnt_3d > 0).sum())} close a 3-cycle): {sum(real.values())} mismatched cells")
    assert not any(real.values()), {c: v for c, v in real.items() if v}
    # sampled rows only depend on rows touching their accounts, so the leaky variants run on that slice
    con.register("accts", pd.DataFrame({"a": sorted(set(sample.from_acct) | set(sample.to_acct))}))
    near = "t.from_acct IN (SELECT a FROM accts) OR t.to_acct IN (SELECT a FROM accts)"
    for leak in ("frame", "cycle"):
        got = con.sql(f"SELECT * FROM ({c_query(con, leak, near)}) WHERE tx_id IN (SELECT tx_id FROM sample)").df()
        bad = mismatches(got, ref, SET_C)
        print(f"  leaky variant '{leak}': {sum(bad.values())} mismatched cells in {sum(1 for v in bad.values() if v)} columns")
        assert sum(bad.values()) > 0, f"check failed to catch the '{leak}' leak"


def assert_nulls(con, cols, nullable):
    n = con.execute("SELECT COUNT(*) FROM features").fetchall()[0][0]
    assert n == EXPECTED_ROWS, n
    nulls = con.execute("SELECT " + ", ".join(f"COUNT(*) - COUNT({c})" for c in cols) + " FROM features").fetchall()[0]
    bad = {c: v for c, v in zip(cols, nulls) if (v and c not in nullable) or v == n}
    assert not bad, f"unexpected NULLs: {bad}"


# time drift pushes every account past the train range; one busy hub only moves its own rows,
# so count accounts, not rows (070:100428660 alone puts 2.3% of test rows above the train max of out_cnt_3d)
def assert_no_drift(con, cols):
    acct = {c: "to_acct" if c.startswith("in_") else "from_acct" for c in cols}
    drift = con.sql(f"""
        WITH m AS (SELECT {", ".join(f"MAX(f.{c}) {c}" for c in cols)} FROM features f JOIN transactions t USING (tx_id) WHERE t.split = 'train')
        SELECT {", ".join(f"COUNT(DISTINCT CASE WHEN f.{c} > m.{c} THEN t.{acct[c]} END) / COUNT(DISTINCT t.{acct[c]}) {c}" for c in cols)}
        FROM features f JOIN transactions t USING (tx_id), m WHERE t.split = 'test'
    """).df().T[0].sort_values(ascending=False)
    print("share of test accounts with values above the train max (top 5):")
    print(drift.head(5).round(5).to_string())
    assert (drift < 0.01).all(), drift[drift >= 0.01]


def check_c(con):
    assert_nulls(con, SET_C, NULLABLE_C)
    bad_order = con.execute("SELECT COUNT(*) FROM features WHERE NOT (out_cpty_1h <= out_cpty_24h AND out_cpty_24h <= out_cpty_3d "
                            "AND out_cpty_3d <= out_cnt_3d AND in_cpty_3d <= in_cnt_3d AND pair_cnt_3d <= out_cnt_3d)").fetchall()[0][0]
    assert bad_order == 0, f"{bad_order} rows where distinct counts exceed totals or windows aren't nested"
    assert_no_drift(con, SET_C)
    print(con.sql("""
        SELECT t.is_laundering, ROUND(AVG(out_cpty_24h), 1) out_cpty_24h, ROUND(AVG(in_cpty_24h), 1) in_cpty_24h,
               ROUND(AVG(out_recv_cnt_24h), 1) out_recv_24h, ROUND(AVG((pair_cnt_3d > 0)::INT), 3) seen_pair,
               ROUND(AVG((pair_rev_cnt_3d > 0)::INT), 4) rev_pair, SUM((cyc3_cnt_3d > 0)::INT) cyc3_rows
        FROM features f JOIN transactions t USING (tx_id) GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False))


def check_b(con):
    assert_nulls(con, SET_B, NULLABLE_B)
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
    assert_no_drift(con, SET_B)

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
    # well below free RAM (~3.7 GB of 16) so DuckDB spills to disk instead of failing the allocation;
    # set C's 10M-row stream window fails at 3GB / all threads and runs at 1.5GB / 4 threads
    con = duckdb.connect(str(DB), config={"memory_limit": "1.5GB", "threads": 4})
    if "--check" in sys.argv:
        return leakage_check(con)
    build_fx(con)
    build_a(con)
    check_a(con)
    build_b(con)
    check_b(con)
    build_c(con)
    check_c(con)
    print(f"features: {EXPECTED_ROWS:,} rows, sets A / B / C = {len(SET_A)} / {len(SET_B)} / {len(SET_C)} columns")


if __name__ == "__main__":
    main()
