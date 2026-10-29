import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

ALERTS_PER_DAY = 500  # alert budget K, chosen in docs/decisions.md


def pr_auc(y, score):
    return average_precision_score(y, score)


# threshold that maximises F1 on this data (pick on val, then apply to test with f1_at)
def best_f1_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = np.argmax(f1[:-1])
    return t[i], f1[i]


def f1_at(y, score, threshold):
    return f1_score(y, np.asarray(score) >= threshold)


# top-k scores per day; ties broken by a seeded shuffle so binary scorers aren't favoured by row order
def precision_recall_at_k_per_day(y, score, day, k=ALERTS_PER_DAY, seed=0):
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score), "day": np.asarray(day)})
    df["tie"] = np.random.default_rng(seed).random(len(df))
    top = df.sort_values(["s", "tie"], ascending=False).groupby("day").head(k)
    tp = top.y.sum()
    return tp / len(top), tp / df.y.sum()


# pattern: pattern type for laundering rows ("UNLISTED" if none), ignored for licit rows
def recall_by_pattern(y, flagged, pattern):
    df = pd.DataFrame({"y": np.asarray(y), "f": np.asarray(flagged), "p": np.asarray(pattern)})
    pos = df[df.y == 1]
    return pos.groupby("p").f.agg(recall="mean", n="size")


# clusters=None resamples rows; otherwise whole clusters (e.g. one laundering attempt) are drawn together
# ponytail: re-sorts scores on every resample (~0.18 s at 900k rows, ~55 min for the T7 test run); sort once + weighted cumsum AP if Phase 3 needs many runs
def _bootstrap(y, scores, metric, n, seed, clusters=None):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if clusters is not None:
        codes = pd.factorize(np.asarray(clusters))[0]
        order = np.argsort(codes, kind="stable")
        sizes = np.bincount(codes)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = []
    for _ in range(n):
        if clusters is None:
            i = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(sizes), len(sizes))
            lens = sizes[pick]
            # row positions of every picked cluster, concatenated
            i = order[np.repeat(starts[pick] - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())]
        out.append([metric(y[i], np.asarray(s)[i]) for s in scores])
    return np.array(out)


# 95% CI of metric(score)
def bootstrap_ci(y, score, metric=pr_auc, n=1000, seed=0, clusters=None):
    b = _bootstrap(y, [score], metric, n, seed, clusters)[:, 0]
    return metric(y, score), *np.percentile(b, [2.5, 97.5])


# 95% CI of metric(a) - metric(b) on the same resamples; "significantly better" = CI excludes 0
def paired_bootstrap_ci(y, score_a, score_b, metric=pr_auc, n=1000, seed=0, clusters=None):
    b = _bootstrap(y, [score_a, score_b], metric, n, seed, clusters)
    d = b[:, 0] - b[:, 1]
    return metric(y, score_a) - metric(y, score_b), *np.percentile(d, [2.5, 97.5])


def self_check():
    rng = np.random.default_rng(1)
    y = (rng.random(100_000) < 0.01).astype(int)
    noise = rng.random(len(y))
    assert pr_auc(y, y + 0.1 * noise) == 1
    assert abs(pr_auc(y, noise) - y.mean()) < 0.005
    t, f1 = best_f1_threshold(y, y + 0.1 * noise)
    assert f1 == 1 and f1_at(y, y + 0.1 * noise, t) == 1

    # 2 days, 2 positives each, k=2: the perfect scorer finds all 4
    yk = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    dk = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    assert precision_recall_at_k_per_day(yk, yk, dk, k=2) == (1, 1)
    assert precision_recall_at_k_per_day(yk, 1 - yk, dk, k=2) == (0, 0)

    rp = recall_by_pattern([1, 1, 1, 0], [1, 0, 1, 1], ["A", "A", "UNLISTED", None])
    assert rp.loc["A", "recall"] == 0.5 and rp.loc["UNLISTED", "recall"] == 1 and rp.n.sum() == 3

    small = slice(0, 20_000)
    d, lo, hi = paired_bootstrap_ci(y[small], (y + 0.1 * noise)[small], noise[small], n=200)
    assert lo > 0 and lo <= d <= hi
    v, lo, hi = bootstrap_ci(y[small], noise[small], n=200)
    assert lo <= v <= hi

    # cluster resampling keeps clusters whole: rows of a picked cluster always come together
    cl = np.arange(len(y)) // 5
    b = _bootstrap(cl, [cl], lambda a, s: float(np.all(np.bincount(a, minlength=cl.max() + 1)[np.unique(a)] % 5 == 0)), 20, 0, cl)
    assert b.min() == 1
    v, lo_c, hi_c = bootstrap_ci(y[small], (y + noise)[small], n=200, clusters=(np.arange(20_000) // 50))
    assert lo_c <= v <= hi_c
    print("metrics self-check passed")


# trivial scorers on val: the floor every Phase 2 model has to beat
def baselines():
    import duckdb

    from load import DB

    con = duckdb.connect(str(DB), read_only=True)
    val = con.sql("""
        SELECT t.is_laundering y, t.day, (t.payment_format = 'ACH')::INT is_ach,
               CASE WHEN t.is_laundering = 1 THEN COALESCE(p.pattern_type, 'UNLISTED') END pattern
        FROM transactions t LEFT JOIN tx_pattern p USING (tx_id) WHERE t.split = 'val'
    """).df()
    scorers = {"random": np.random.default_rng(0).random(len(val)), "is_ach": val.is_ach.to_numpy()}
    for name, s in scorers.items():
        v, lo, hi = bootstrap_ci(val.y, s, n=200)
        t, f1 = best_f1_threshold(val.y, s)
        p_k, r_k = precision_recall_at_k_per_day(val.y, s, val.day)
        print(f"{name:7s} PR-AUC {v:.4f} [{lo:.4f}, {hi:.4f}]  best F1 {f1:.4f}  P@{ALERTS_PER_DAY}/day {p_k:.3f}  R@{ALERTS_PER_DAY}/day {r_k:.3f}")
    print(recall_by_pattern(val.y, val.is_ach, val.pattern).round(3).to_string())


if __name__ == "__main__":
    self_check()
    baselines()
