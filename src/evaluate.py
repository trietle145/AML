import argparse

import duckdb
import lightgbm as lgb
import pandas as pd

from load import DB, ROOT
from metrics import (ALERTS_PER_DAY, best_f1_threshold, bootstrap_ci, f1_at, pr_auc, precision_recall_at_k_per_day,
                     recall_by_pattern)
from train import CATEGORICAL, FEATURE_SETS, load

REPORTS = ROOT / "reports"


def load_split(con, split):
    return con.sql(f"""
        SELECT s.model || '/' || s.feature_set name, s.tx_id, s.score, t.is_laundering y, t.day,
               CASE WHEN t.is_laundering = 1 THEN COALESCE(p.pattern_type, 'UNLISTED') END pattern
        FROM scores s JOIN transactions t USING (tx_id) LEFT JOIN tx_pattern p USING (tx_id)
        WHERE s.split = '{split}' ORDER BY name, s.tx_id
    """).df()


# val only: what lgbm/AB relies on, whether any single feature is a leak, and how much is just "ACH"
def analyze(con):
    imp = lgb.Booster(model_file=str(ROOT / "data/models/lgbm_AB.txt"))
    imp = pd.DataFrame({"feature": imp.feature_name(), "gain": imp.feature_importance("gain")})
    imp = imp.assign(share=imp.gain / imp.gain.sum()).sort_values("gain", ascending=False)
    imp.to_csv(REPORTS / "val-importance.csv", index=False)
    print("lgbm/AB gain importance (top 10):")
    print(imp.head(10).round(4).to_string(index=False))

    # smell test: each feature alone as a score, both signs; categoricals scored by their val laundering rate
    v = load(con, FEATURE_SETS["AB"], split="val")
    single = {}
    for c in FEATURE_SETS["AB"]:
        s = v.groupby(c, observed=True).y.transform("mean") if c in CATEGORICAL else v[c].astype(float)
        s = s.fillna(s.min() - 1)
        single[c] = max(pr_auc(v.y, s), pr_auc(v.y, -s))
    single = pd.Series(single, name="val_pr_auc").sort_values(ascending=False)
    single.to_csv(REPORTS / "val-single-feature.csv", index_label="feature")
    print("\nsingle-feature val PR-AUC (top 5; look > 0.07, leak > 0.5):")
    print(single.head(5).round(4).to_string())
    assert single.max() < 0.5, single[single >= 0.5]

    # how much of lgbm/AB is "ACH": ranking quality inside ACH rows, and the ACH share of what it flags
    s = con.sql("""
        SELECT s.score, t.is_laundering y, t.payment_format = 'ACH' ach, f.out_has_history, f.in_has_history
        FROM scores s JOIN transactions t USING (tx_id) JOIN features f USING (tx_id)
        WHERE s.split = 'val' AND s.model = 'lgbm' AND s.feature_set = 'AB'
    """).df()
    thr, _ = best_f1_threshold(s.y, s.score)
    flagged = s[s.score >= thr]
    print(f"\nlgbm/AB inside ACH rows only: PR-AUC {pr_auc(s.y[s.ach], s.score[s.ach]):.4f} "
          f"(ACH base rate {s.y[s.ach].mean():.4f}); flagged rows that are ACH: {flagged.ach.mean():.3f}")
    hist = s.groupby(["out_has_history", "in_has_history"]).agg(rows=("y", "size"), laundering=("y", "sum"), rate=("y", "mean"))
    hist.to_csv(REPORTS / "val-history.csv")
    print(hist.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test", "tail"])
    ap.add_argument("--final", action="store_true", help="required to score test or tail (done once)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--analyze", action="store_true", help="val-only analysis of lgbm/AB instead of metrics")
    args = ap.parse_args()
    if args.analyze:
        return analyze(duckdb.connect(str(DB), read_only=True, config={"memory_limit": "3GB"}))
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
    metrics, patterns = pd.DataFrame(rows), pd.DataFrame(recalls).assign(n=by_pattern.n)
    metrics.to_csv(REPORTS / f"{args.split}-metrics.csv", index=False)
    patterns.to_csv(REPORTS / f"{args.split}-recall-by-pattern.csv", index_label="pattern")
    print(f"split={args.split}  rows={len(d):,}  positives={int(d.y.sum()):,}  boot={args.boot}")
    print(metrics.round(4).to_string(index=False))
    print("\nrecall by pattern at the val threshold")
    print(patterns.round(3).to_string())


if __name__ == "__main__":
    main()
