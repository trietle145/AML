import argparse
import datetime

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from load import DB, ROOT
from metrics import (ALERTS_PER_DAY, best_f1_threshold, bootstrap_ci, f1_at, paired_bootstrap_ci, pr_auc,
                     precision_recall_at_k_per_day, recall_by_pattern)
from train import CATEGORICAL, FEATURE_SETS, load

REPORTS = ROOT / "reports"
# phase -> (new model, partner) pairs; a pair is used when both have scores
PAIRS = {2: [("lgbm/AB", "lr/AB"), ("lgbm/AB", "lgbm/A")],
         3: [("lgbm/ABC", "lgbm/AB"), ("gnn/AB", "lgbm/AB"), ("gnn/AB", "lgbm/ABC"), ("gnn/AB", "mlp/AB"), ("lgbm/ABG", "lgbm/AB")]}


# cluster = laundering attempt for listed-pattern rows, the row itself otherwise
def load_split(con, split):
    return con.sql(f"""
        SELECT s.model || '/' || s.feature_set name, s.tx_id, s.score, t.is_laundering y, t.day,
               CASE WHEN t.is_laundering = 1 THEN COALESCE(p.pattern_type, 'UNLISTED') END pattern,
               COALESCE('a' || p.attempt_id, 't' || s.tx_id) cluster
        FROM scores s JOIN transactions t USING (tx_id) LEFT JOIN tx_pattern p USING (tx_id)
        WHERE s.split = '{split}' ORDER BY name, s.tx_id
    """).df()


# final run only: paired differences (iid and attempt-cluster) and recall-by-pattern CIs
def final_extras(data, thresholds, split, boot, pairs, prefix, names):
    by = {name: d for name, d in data.groupby("name")}
    paired = []
    for a, b in pairs:
        da, db = by[a], by[b]
        assert (da.tx_id.to_numpy() == db.tx_id.to_numpy()).all(), "paired scores must cover the same rows"
        row = {"a": a, "b": b}
        for kind, cl in (("iid", None), ("cluster", da.cluster.to_numpy())):
            diff, lo, hi = paired_bootstrap_ci(da.y.to_numpy(), da.score.to_numpy(), db.score.to_numpy(), n=boot, clusters=cl)
            row.update({"diff": diff, f"lo_{kind}": lo, f"hi_{kind}": hi})
        paired.append(row)
    paired = pd.DataFrame(paired)
    paired.to_csv(REPORTS / f"{prefix}{split}-paired.csv", index=False)
    print("\npaired PR-AUC differences (a - b):")
    print(paired.round(4).to_string(index=False))

    rng, rows = np.random.default_rng(0), []
    for name in names:
        d = by[name]
        pos = d[d.y == 1]
        for pattern, g in pos.groupby("pattern"):
            hit = (g.score >= thresholds[name]).to_numpy()
            b = hit[rng.integers(0, len(hit), (boot, len(hit)))].mean(axis=1)
            rows.append({"model": name, "pattern": pattern, "n": len(hit), "recall": hit.mean(),
                         "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
    pd.DataFrame(rows).to_csv(REPORTS / f"{prefix}{split}-recall-by-pattern-ci.csv", index=False)


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
    ap.add_argument("--phase", type=int, default=2, choices=[2, 3], help="3 writes reports/phase3-* and never touches Phase 2 files")
    ap.add_argument("--models", help="comma-separated models to evaluate (default: all in scores); required for a phase 3 --final run")
    args = ap.parse_args()
    if args.analyze:
        return analyze(duckdb.connect(str(DB), read_only=True, config={"memory_limit": "3GB"}))
    final = args.split != "val"
    prefix = "" if args.phase == 2 else f"phase{args.phase}-"
    models = args.models.split(",") if args.models else None
    if final and not args.final:
        ap.error(f"--split {args.split} needs --final: test and tail are scored once, after val review")
    if final and (REPORTS / f"{prefix}{args.split}-metrics.csv").exists():
        ap.error(f"{args.split} was already scored (reports/{prefix}{args.split}-metrics.csv exists); it is scored once")
    if final and args.phase > 2:
        if not models:
            ap.error("a phase 3 --final run needs --models (the new models only)")
        earlier = REPORTS / f"{args.split}-metrics.csv"
        done = set(pd.read_csv(earlier).model) & set(models) if earlier.exists() else set()
        if done:
            ap.error(f"{sorted(done)} already scored on {args.split} in Phase 2; they are pairing partners, not new models")

    con = duckdb.connect(str(DB), read_only=True)
    # thresholds always come from val, then get applied unchanged to the evaluated split
    val = load_split(con, "val")
    data = val if not final else load_split(con, args.split)
    names = models or sorted(data.name.unique())
    missing = set(names) - set(data.name)
    if missing:
        ap.error(f"no scores for {sorted(missing)}")
    # phase 3 always adds cluster CIs, pairs and recall CIs; phase 2 only on its final run
    extras = final or args.phase > 2
    rows, recalls, thresholds = [], {}, {}
    for name, d in data[data.name.isin(names)].groupby("name"):
        v = val[val.name == name]
        thr, _ = best_f1_threshold(v.y, v.score)
        thresholds[name] = thr
        auc, lo, hi = bootstrap_ci(d.y, d.score, n=args.boot)
        p_k, r_k = precision_recall_at_k_per_day(d.y, d.score, d.day)
        row = {"model": name, "pr_auc": auc, "ci_lo": lo, "ci_hi": hi, "f1": f1_at(d.y, d.score, thr),
               "threshold": thr, f"p@{ALERTS_PER_DAY}/day": p_k, f"r@{ALERTS_PER_DAY}/day": r_k}
        if extras:
            _, row["ci_lo_cluster"], row["ci_hi_cluster"] = bootstrap_ci(d.y, d.score, n=args.boot, clusters=d.cluster.to_numpy())
            row["scored_at"] = datetime.datetime.now().isoformat(timespec="minutes")
        rows.append(row)
        by_pattern = recall_by_pattern(d.y, d.score >= thr, d.pattern)
        recalls[name] = by_pattern.recall
    if extras:
        pairs = [(a, b) for a, b in PAIRS[args.phase] if a in names and b in set(data.name)]
        final_extras(data, thresholds, args.split, args.boot, pairs, prefix, names)
    metrics, patterns = pd.DataFrame(rows), pd.DataFrame(recalls).assign(n=by_pattern.n)
    metrics.to_csv(REPORTS / f"{prefix}{args.split}-metrics.csv", index=False)
    patterns.to_csv(REPORTS / f"{prefix}{args.split}-recall-by-pattern.csv", index_label="pattern")
    print(f"split={args.split}  rows={len(d):,}  positives={int(d.y.sum()):,}  boot={args.boot}")
    print(metrics.round(4).to_string(index=False))
    print("\nrecall by pattern at the val threshold")
    print(patterns.round(3).to_string())


if __name__ == "__main__":
    main()
