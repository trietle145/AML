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


# mask of the top-k scores per day; ties broken by a seeded shuffle so binary scorers aren't favoured by row order
def top_k_per_day(score, day, k=ALERTS_PER_DAY, seed=0):
    df = pd.DataFrame({"s": np.asarray(score), "day": np.asarray(day)})
    df["tie"] = np.random.default_rng(seed).random(len(df))
    flag = np.zeros(len(df), bool)
    flag[df.sort_values(["s", "tie"], ascending=False).groupby("day").head(k).index] = True
    return flag


def precision_recall_at_k_per_day(y, score, day, k=ALERTS_PER_DAY, seed=0):
    y, top = np.asarray(y), top_k_per_day(score, day, k, seed)
    tp = y[top].sum()
    return tp / top.sum(), tp / y.sum()


# pattern: pattern type for laundering rows ("UNLISTED" if none), ignored for licit rows
def recall_by_pattern(y, flagged, pattern):
    df = pd.DataFrame({"y": np.asarray(y), "f": np.asarray(flagged), "p": np.asarray(pattern)})
    pos = df[df.y == 1]
    return pos.groupby("p").f.agg(recall="mean", n="size")


# AP of (y, s) for any row weights w (bootstrap draw counts), sorting s only once; same result as pr_auc on the resampled rows
def _weighted_ap(y, s):
    order = np.argsort(-np.asarray(s), kind="stable")
    ys, ss = np.asarray(y)[order], np.asarray(s)[order]
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])  # last row of each distinct score
    pos = np.flatnonzero(ys == 1)
    # recall only moves at score groups holding a positive, so precision is needed only there
    ends_p = np.unique(ends[np.searchsorted(ends, pos)])
    k = np.searchsorted(pos, ends_p, side="right") - 1
    pos_rows = order[pos]

    def ap(w):
        tp = np.cumsum(w[pos_rows])[k]
        seen = np.cumsum(w[order])[ends_p]
        step = np.diff(np.r_[0, tp])
        m = step > 0  # groups whose positives weren't drawn add nothing (and would be 0/0)
        return float(np.sum(step[m] / tp[-1] * tp[m] / seen[m]))
    return ap


# clusters=None resamples rows; otherwise whole clusters (e.g. one laundering attempt) are drawn together
def _bootstrap(y, scores, metric, n, seed, clusters=None):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if clusters is not None:
        codes = pd.factorize(np.asarray(clusters))[0]
        order = np.argsort(codes, kind="stable")
        sizes = np.bincount(codes)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    fast = [_weighted_ap(y, s) for s in scores] if metric is pr_auc else None
    out = []
    for _ in range(n):
        if clusters is None:
            i = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(sizes), len(sizes))
        if fast:
            # draw counts per row; for clusters every row of a picked cluster counts once per pick
            w = np.bincount(i, minlength=len(y)) if clusters is None else np.bincount(pick, minlength=len(sizes))[codes]
            out.append([ap(w) for ap in fast])
            continue
        if clusters is not None:
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

    # fast weighted AP must equal sklearn on the resampled rows, ties included (rounded scores), iid and cluster
    tied = np.round(noise[small] + 0.3 * y[small], 2)
    for cl in (None, np.arange(20_000) // 50):
        fast = _bootstrap(y[small], [tied], pr_auc, 30, 0, cl)
        slow = _bootstrap(y[small], [tied], lambda a, s: pr_auc(a, s), 30, 0, cl)
        assert np.allclose(fast, slow, atol=1e-12), np.abs(fast - slow).max()
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
