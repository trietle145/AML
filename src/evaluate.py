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
         3: [("lgbm/ABC", "lgbm/AB"), ("gnn/ABC", "lgbm/ABC"), ("gnn/ABC", "lgbm/AB"), ("gnn/ABC", "mlp/ABC"),
             ("gnn/AB", "lgbm/AB"), ("gnn/AB", "mlp/AB"), ("lgbm/ABG", "lgbm/ABC")]}


# cluster = laundering attempt for listed-pattern rows, the row itself otherwise
def load_split(con, split, models=None):
    # only the requested models: with ~15 models in `scores`, loading them all would mean ~15M rows in pandas
    only = f"AND s.model || '/' || s.feature_set IN ({', '.join(repr(m) for m in models)})" if models else ""
    return con.sql(f"""
        SELECT s.model || '/' || s.feature_set name, s.tx_id, s.score, t.is_laundering y, t.day,
               CASE WHEN t.is_laundering = 1 THEN COALESCE(p.pattern_type, 'UNLISTED') END pattern,
               COALESCE('a' || p.attempt_id, 't' || s.tx_id) cluster
        FROM scores s JOIN transactions t USING (tx_id) LEFT JOIN tx_pattern p USING (tx_id)
        WHERE s.split = '{split}' {only} ORDER BY name, s.tx_id
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
        print(f"{datetime.datetime.now():%H:%M} pair {len(paired)}/{len(pairs)} {a} - {b}: {diff:+.4f}")
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


# val only: what an lgbm model relies on, whether any single feature is a leak, and how much is just "ACH"
def analyze(con, fs="AB", prefix=""):
    imp = lgb.Booster(model_file=str(ROOT / f"data/models/lgbm_{fs}.txt"))
    imp = pd.DataFrame({"feature": imp.feature_name(), "gain": imp.feature_importance("gain")})
    imp = imp.assign(share=imp.gain / imp.gain.sum()).sort_values("gain", ascending=False)
    imp.to_csv(REPORTS / f"{prefix}val-importance.csv", index=False)
    print(f"lgbm/{fs} gain importance (top 10):")
    print(imp.head(10).round(4).to_string(index=False))

    # smell test: each feature alone as a score, both signs; categoricals scored by their val laundering rate
    v = load(con, FEATURE_SETS[fs], split="val")
    single = {}
    for c in FEATURE_SETS[fs]:
        s = v.groupby(c, observed=True).y.transform("mean") if c in CATEGORICAL else v[c].astype(float)
        s = s.fillna(s.min() - 1)
        single[c] = max(pr_auc(v.y, s), pr_auc(v.y, -s))
    single = pd.Series(single, name="val_pr_auc").sort_values(ascending=False)
    single.to_csv(REPORTS / f"{prefix}val-single-feature.csv", index_label="feature")
    del v
    print("\nsingle-feature val PR-AUC (top 5; look > 0.07, leak > 0.5):")
    print(single.head(5).round(4).to_string())
    assert single.max() < 0.5, single[single >= 0.5]

    # how much of lgbm/AB is "ACH": ranking quality inside ACH rows, and the ACH share of what it flags
    s = con.sql("""
        SELECT s.score, t.is_laundering y, t.payment_format = 'ACH' ach, f.out_has_history, f.in_has_history,
               t.is_laundering = 1 AND p.tx_id IS NULL unlisted
        FROM scores s JOIN transactions t USING (tx_id) JOIN features f USING (tx_id) LEFT JOIN tx_pattern p USING (tx_id)
        WHERE s.split = 'val' AND s.model = 'lgbm' AND s.feature_set = ?
    """, params=[fs]).df()
    thr, _ = best_f1_threshold(s.y, s.score)
    flagged = s[s.score >= thr]
    print(f"\nlgbm/{fs} inside ACH rows only: PR-AUC {pr_auc(s.y[s.ach], s.score[s.ach]):.4f} "
          f"(ACH base rate {s.y[s.ach].mean():.4f}); flagged rows that are ACH: {flagged.ach.mean():.3f}")
    hist = s.groupby(["out_has_history", "in_has_history"]).agg(rows=("y", "size"), laundering=("y", "sum"), rate=("y", "mean"))
    hist.to_csv(REPORTS / f"{prefix}val-history.csv")
    print(hist.round(4).to_string())
    unl = s[s.unlisted].assign(flagged=lambda d: d.score >= thr).groupby("ach").flagged.agg(n="size", caught="sum", recall="mean")
    unl.to_csv(REPORTS / f"{prefix}val-unlisted-by-ach.csv")
    print(f"\nUNLISTED recall at the val threshold, by ACH:\n{unl.round(3).to_string()}")


# per val row: does its sender or receiver have an earlier (<= 3d) transaction this model scored >= thr?
# hit_val counts only val-period history; train-period scores are in-sample, so they flatter train laundering
PAST_SCORE_SQL = """
WITH hi AS (SELECT UNNEST([t.from_acct, t.to_acct]) a, t.ts, t.split = 'val' in_val
            FROM scores s JOIN transactions t USING (tx_id)
            WHERE s.model = ? AND s.feature_set = ? AND s.split IN ('train', 'val') AND s.score >= ?),
     va AS (SELECT tx_id, ts, UNNEST([from_acct, to_acct]) a FROM transactions WHERE split = 'val'),
     hit AS (SELECT va.tx_id, BOOL_OR(hi.in_val) hit_val FROM va JOIN hi
             ON hi.a = va.a AND hi.ts < va.ts AND hi.ts >= va.ts - INTERVAL 3 DAY GROUP BY 1)
SELECT CASE WHEN t.is_laundering = 0 THEN 'licit' WHEN p.tx_id IS NOT NULL THEN 'listed'
            WHEN t.payment_format = 'ACH' THEN 'UNLISTED ACH' ELSE 'UNLISTED non-ACH' END grp,
       COUNT(*) n, SUM((s.score >= ?)::INT) flagged,
       SUM((h.tx_id IS NOT NULL)::INT) hit_any, SUM(COALESCE(h.hit_val, FALSE)::INT) hit_val,
       SUM((h.tx_id IS NOT NULL AND s.score < ?)::INT) hit_any_nf, SUM(COALESCE(h.hit_val AND s.score < ?, FALSE)::INT) hit_val_nf
FROM transactions t JOIN scores s USING (tx_id) LEFT JOIN tx_pattern p USING (tx_id) LEFT JOIN hit h USING (tx_id)
WHERE t.split = 'val' AND s.model = ? AND s.feature_set = ? GROUP BY 1"""


# train + val only: who the UNLISTED launderers are, whether they touch listed schemes or repeat accounts,
# whether past model scores already flag their accounts, and how far a model trained on UNLISTED alone gets
def analyze_unlisted(con, models=("lgbm/ABC", "gnn/ABC")):
    def report(df, name):
        df.to_csv(REPORTS / f"phase4-unlisted-{name}.csv", index=False)
        print(f"\n== {name}\n{df.round(4).to_string(index=False)}")

    grp = "CASE WHEN t.is_laundering = 0 THEN 'licit' WHEN p.tx_id IS NULL THEN 'UNLISTED' ELSE 'listed' END grp"
    tv = "t.split IN ('train', 'val')"
    prof = con.sql(f"""
        SELECT {grp}, COUNT(*) n,
               AVG((t.payment_format = 'ACH')::INT) ach, MEDIAN(10 ^ f.log_amount_usd) med_usd,
               AVG((10 ^ f.log_amount_usd BETWEEN 10000 AND 20000)::INT) usd_10_20k,
               AVG(f.is_cross_bank::INT) cross_bank, AVG(f.is_self_transfer::INT) self,
               AVG((f.out_has_history = 0)::INT) sender_new, AVG((f.in_has_history = 0)::INT) receiver_new,
               MEDIAN(f.out_cnt_24h) out_cnt_24h, MEDIAN(f.in_cnt_24h) in_cnt_24h, MEDIAN(f.in_cpty_3d) in_cpty_3d,
               MEDIAN(f.out_recv_cnt_24h) sender_recv_24h, AVG((f.pair_cnt_3d > 0)::INT) seen_pair, AVG(f.hour) mean_hour
        FROM transactions t JOIN features f USING (tx_id) LEFT JOIN tx_pattern p USING (tx_id)
        WHERE {tv} GROUP BY 1 ORDER BY 1""").df()
    assert prof.n.sum() == 3_248_921 + 965_524, "train + val rows only"
    report(prof, "profile")
    report(con.sql(f"""
        SELECT t.payment_format, COUNT(*) n, 100 * COUNT(*) / SUM(COUNT(*)) OVER () pct
        FROM transactions t LEFT JOIN tx_pattern p USING (tx_id)
        WHERE {tv} AND t.is_laundering = 1 AND p.tx_id IS NULL GROUP BY 1 ORDER BY 2 DESC""").df(), "formats")

    # first time each account appears in a listed scheme / in any laundering, within train + val
    links = con.sql(f"""
        WITH sch AS (SELECT a, MIN(ts) first_ts FROM (SELECT UNNEST([t.from_acct, t.to_acct]) a, t.ts
                     FROM tx_pattern p JOIN transactions t USING (tx_id) WHERE {tv}) GROUP BY 1),
             lab AS (SELECT a, MIN(ts) first_ts FROM (SELECT UNNEST([t.from_acct, t.to_acct]) a, t.ts
                     FROM transactions t WHERE {tv} AND t.is_laundering = 1) GROUP BY 1)
        SELECT {grp}, COUNT(*) n,
               AVG((ss.a IS NOT NULL OR sr.a IS NOT NULL)::INT) shares_acct_with_scheme,
               AVG(COALESCE(ss.first_ts < t.ts OR sr.first_ts < t.ts, FALSE)::INT) scheme_acct_earlier,
               AVG(COALESCE(ls.first_ts < t.ts OR lr.first_ts < t.ts, FALSE)::INT) acct_had_earlier_laundering
        FROM transactions t LEFT JOIN tx_pattern p USING (tx_id)
             LEFT JOIN sch ss ON ss.a = t.from_acct LEFT JOIN sch sr ON sr.a = t.to_acct
             LEFT JOIN lab ls ON ls.a = t.from_acct LEFT JOIN lab lr ON lr.a = t.to_acct
        WHERE {tv} GROUP BY 1 ORDER BY 1""").df()
    report(links, "links")

    rows = []
    for m in models:
        model, fs = m.split("/")
        v = con.sql("""SELECT s.score, t.is_laundering y FROM scores s JOIN transactions t USING (tx_id)
                       WHERE s.split = 'val' AND s.model = ? AND s.feature_set = ?""", params=[model, fs]).df()
        # "high" = at the alert threshold (val best F1), or the looser top 1% of val scores; flagged is always the alert
        thr = float(best_f1_threshold(v.y, v.score)[0])
        for cut_name, cut in (("best_f1", thr), ("top1pct", float(v.score.quantile(0.99)))):
            d = con.sql(PAST_SCORE_SQL, params=[model, fs, cut, thr, thr, thr, model, fs]).df().set_index("grp")
            d.loc["UNLISTED"] = d.loc[["UNLISTED ACH", "UNLISTED non-ACH"]].sum()
            for hist in ("any", "val"):
                rate = d[f"hit_{hist}"] / d.n
                rows.append(pd.DataFrame({"model": m, "high": cut_name, "cut": cut, "history": hist, "n": d.n,
                                          "flagged": d.flagged, "hit": d[f"hit_{hist}"], "hit_rate": rate,
                                          "lift_vs_licit": rate / rate["licit"],
                                          "hit_not_flagged": d[f"hit_{hist}_nf"]}).reset_index())
    past = pd.concat(rows)
    assert (past[past.grp == "UNLISTED"].n == 379).all() and (past.hit_not_flagged <= past.hit).all()
    report(past, "past-scores")

    # upper bound: LightGBM on UNLISTED vs licit only (listed positives dropped), same settings as the exploration
    cols = FEATURE_SETS["ABC"]
    pat = con.sql("SELECT tx_id FROM tx_pattern").df().tx_id
    tr, va = load(con, cols, "train"), load(con, cols, "val")
    tr, va = tr[~tr.tx_id.isin(pat)], va[~va.tx_id.isin(pat)]
    m = lgb.LGBMClassifier(n_estimators=2000, num_leaves=15, learning_rate=0.05, min_child_samples=500, random_state=0, verbose=-1)
    m.fit(tr[cols], tr.y, eval_X=va[cols], eval_y=va.y, eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    del tr
    s = m.predict_proba(va[cols])[:, 1]
    na = (va.payment_format != "ACH").to_numpy()
    ub = pd.DataFrame([{"rows": k, "pos": int(va.y[mask].sum()), "pr_auc": pr_auc(va.y[mask], s[mask]),
                        "base_rate": va.y[mask].mean(), "best_iteration": m.best_iteration_}
                       for k, mask in (("all", np.ones(len(va), bool)), ("non-ACH", na))])
    assert ub.pos[0] == 379 and (ub.pr_auc > ub.base_rate).all()
    report(ub, "upper-bound")
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features:", ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(6).items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test", "tail"])
    ap.add_argument("--final", action="store_true", help="required to score test or tail (done once)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--analyze", action="store_true", help="val-only analysis of one lgbm model (--models, default lgbm/AB) instead of metrics")
    ap.add_argument("--phase", type=int, default=2, choices=[2, 3], help="3 writes reports/phase3-* and never touches Phase 2 files")
    ap.add_argument("--models", help="comma-separated models to evaluate (default: all in scores); required for a phase 3 --final run")
    ap.add_argument("--pairs", help="comma-separated a:b pairs for paired differences, instead of PAIRS[phase]; b may be outside --models")
    ap.add_argument("--analyze-unlisted", action="store_true", help="train + val diagnosis of UNLISTED laundering (--models: past-score models, default lgbm/ABC,gnn/ABC); writes reports/phase4-unlisted-*")
    args = ap.parse_args()
    if args.analyze_unlisted:
        con = duckdb.connect(str(DB), read_only=True, config={"memory_limit": "700MB", "threads": 2})
        return analyze_unlisted(con, args.models.split(",") if args.models else ("lgbm/ABC", "gnn/ABC"))
    if args.analyze:
        fs = (args.models or "lgbm/AB").split("/")[1]
        con = duckdb.connect(str(DB), read_only=True, config={"memory_limit": "700MB", "threads": 2})
        return analyze(con, fs, "" if args.phase == 2 else f"phase{args.phase}-")
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
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")] if args.pairs else PAIRS[args.phase]
    # pair partners (e.g. the frozen lgbm/AB on test) are loaded too, though not re-scored
    need = sorted(set(models) | {m for p in pairs for m in p}) if models else None
    # thresholds always come from val, then get applied unchanged to the evaluated split
    val = load_split(con, "val", need)
    data = val if not final else load_split(con, args.split, need)
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
        print(f"{datetime.datetime.now():%H:%M} model {len(rows)}/{len(names)} {name}: PR-AUC {auc:.4f}")
        by_pattern = recall_by_pattern(d.y, d.score >= thr, d.pattern)
        recalls[name] = by_pattern.recall
    if extras:
        if not args.pairs:
            pairs = [(a, b) for a, b in pairs if a in names and b in set(data.name)]
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
