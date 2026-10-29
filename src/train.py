import sys

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import SET_A, SET_B
from load import DB, ROOT

FEATURE_SETS = {"A": SET_A, "AB": SET_A + SET_B}
CATEGORICAL = ["payment_format", "payment_currency", "receiving_currency", "hour"]
GRID = [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
        for nl in (15, 63) for lr in (0.05, 0.1) for mcs in (50, 500)]
MODELS_DIR = ROOT / "data/models"


def load(con, cols, split=None):
    # numerics cast to FLOAT in SQL so 5M x 42 never sits in memory as float64
    sel = ", ".join(f"f.{c}" if c in CATEGORICAL else f"CAST(f.{c} AS FLOAT) {c}" for c in cols)
    where = f"WHERE t.split = '{split}'" if split else ""
    df = con.sql(f"""
        SELECT f.tx_id, t.split, t.is_laundering y, {sel}
        FROM features f JOIN transactions t USING (tx_id) {where} ORDER BY f.tx_id
    """).df()
    strings = [c for c in cols if c in CATEGORICAL and c != "hour"]
    df[strings] = df[strings].astype("category")
    return df


# scored in chunks: LR's one-hot + scaled matrix for all 5M rows at once is ~3 GB of float64
def predict(model, X, chunk=500_000):
    return np.concatenate([model.predict_proba(X.iloc[i:i + chunk])[:, 1] for i in range(0, len(X), chunk)])


def save_scores(con, model, feature_set, df, score):
    out = df[["tx_id", "split"]].assign(model=model, feature_set=feature_set, score=np.asarray(score, dtype="float64"))
    con.execute("CREATE TABLE IF NOT EXISTS scores (model VARCHAR, feature_set VARCHAR, tx_id BIGINT, split VARCHAR, score DOUBLE)")
    con.execute("DELETE FROM scores WHERE model = ? AND feature_set = ?", [model, feature_set])
    con.execute("INSERT INTO scores SELECT model, feature_set, tx_id, split, score FROM out")
    print(f"scores: {model}/{feature_set} written for {len(out):,} rows")


def is_ach(con):
    df = load(con, ["payment_format"])
    save_scores(con, "is_ach", "-", df, (df.payment_format == "ACH").astype(int))


def lr(con, feature_set):
    cols = FEATURE_SETS[feature_set]
    df = load(con, cols)
    cat = [c for c in cols if c in CATEGORICAL]
    b = [c for c in cols if c in SET_B]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("a", StandardScaler(), [c for c in cols if c not in cat and c not in b]),
        # B is all non-negative and heavy-tailed (hubs send 100k+); NULL ratios mean "no history", has_history flags it
        ("b", make_pipeline(SimpleImputer(strategy="constant", fill_value=0), FunctionTransformer(np.log1p), StandardScaler()), b),
    ])
    model = make_pipeline(prep, LogisticRegression(class_weight="balanced", max_iter=1000))
    train = df[df.split == "train"]
    model.fit(train[cols], train.y)
    save_scores(con, "lr", feature_set, df, predict(model, df[cols]))


# small grid, early stopping on val average precision; the best run is saved and scores every row
def lgbm(con, feature_set, tuning_log):
    cols = FEATURE_SETS[feature_set]
    df = load(con, cols)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    runs, best = [], None
    for params in GRID:
        m = lgb.LGBMClassifier(n_estimators=2000, random_state=0, verbose=-1, **params)
        m.fit(tr[cols], tr.y, eval_X=va[cols], eval_y=va.y, eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        ap = m.best_score_["valid_0"]["average_precision"]
        runs.append({"feature_set": feature_set, **params, "best_iteration": m.best_iteration_, "val_pr_auc": ap})
        print(f"  lgbm/{feature_set} {params} -> val PR-AUC {ap:.4f} at {m.best_iteration_} trees")
        if best is None or ap > best[0]:
            best = (ap, m)
    log = pd.DataFrame(runs)
    if tuning_log.exists():
        old = pd.read_csv(tuning_log)
        log = pd.concat([old[old.feature_set != feature_set], log])
    log.to_csv(tuning_log, index=False)
    MODELS_DIR.mkdir(exist_ok=True)
    best[1].booster_.save_model(str(MODELS_DIR / f"lgbm_{feature_set}.txt"))
    save_scores(con, "lgbm", feature_set, df, predict(best[1], df[cols]))


def main():
    con = duckdb.connect(str(DB), config={"memory_limit": "3GB"})
    what = sys.argv[1:]
    phase = "2"
    # --phase N routes the LightGBM tuning log to reports/phaseN-tuning.csv
    if "--phase" in what:
        i = what.index("--phase")
        phase = what[i + 1]
        del what[i:i + 2]
    tuning_log = ROOT / f"reports/phase{phase}-tuning.csv"
    what = what or ["all"]
    if what == ["all"]:
        is_ach(con)
        for fs in FEATURE_SETS:
            lr(con, fs)
            lgbm(con, fs, tuning_log)
    elif what == ["is_ach"]:
        is_ach(con)
    elif len(what) == 2 and what[0] == "lr" and what[1] in FEATURE_SETS:
        lr(con, what[1])
    elif len(what) == 2 and what[0] == "lgbm" and what[1] in FEATURE_SETS:
        lgbm(con, what[1], tuning_log)
    else:
        sys.exit("usage: train.py [--phase N] all | is_ach | lr <set> | lgbm <set>   (sets: " + ", ".join(FEATURE_SETS) + ")")


if __name__ == "__main__":
    main()
