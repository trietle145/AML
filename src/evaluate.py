import argparse

import duckdb
import pandas as pd

from load import DB
from metrics import (ALERTS_PER_DAY, best_f1_threshold, bootstrap_ci, f1_at, precision_recall_at_k_per_day,
                     recall_by_pattern)


def load_split(con, split):
    return con.sql(f"""
        SELECT s.model || '/' || s.feature_set name, s.tx_id, s.score, t.is_laundering y, t.day,
               CASE WHEN t.is_laundering = 1 THEN COALESCE(p.pattern_type, 'UNLISTED') END pattern
        FROM scores s JOIN transactions t USING (tx_id) LEFT JOIN tx_pattern p USING (tx_id)
        WHERE s.split = '{split}' ORDER BY name, s.tx_id
    """).df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test", "tail"])
    ap.add_argument("--final", action="store_true", help="required to score test or tail (done once)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    args = ap.parse_args()
    if args.split != "val" and not args.final:
        ap.error(f"--split {args.split} needs --final: test and tail are scored once, after val review")

    con = duckdb.connect(str(DB), read_only=True)
    # thresholds always come from val, then get applied unchanged to the evaluated split
    val = load_split(con, "val")
    data = val if args.split == "val" else load_split(con, args.split)
    rows, recalls = [], {}
    for name, d in data.groupby("name"):
        v = val[val.name == name]
        thr, _ = best_f1_threshold(v.y, v.score)
        auc, lo, hi = bootstrap_ci(d.y, d.score, n=args.boot)
        p_k, r_k = precision_recall_at_k_per_day(d.y, d.score, d.day)
        rows.append({"model": name, "pr_auc": auc, "ci_lo": lo, "ci_hi": hi, "f1": f1_at(d.y, d.score, thr),
                     "threshold": thr, f"p@{ALERTS_PER_DAY}/day": p_k, f"r@{ALERTS_PER_DAY}/day": r_k})
        by_pattern = recall_by_pattern(d.y, d.score >= thr, d.pattern)
        recalls[name] = by_pattern.recall
    print(f"split={args.split}  rows={len(d):,}  positives={int(d.y.sum()):,}  boot={args.boot}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nrecall by pattern at the val threshold")
    print(pd.DataFrame(recalls).assign(n=by_pattern.n).round(3).to_string())


if __name__ == "__main__":
    main()
