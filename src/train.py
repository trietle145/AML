import sys

import duckdb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import SET_A
from load import DB

FEATURE_SETS = {"A": SET_A}
CATEGORICAL = ["payment_format", "payment_currency", "receiving_currency", "hour"]


def load(con, cols):
    df = con.sql(f"""
        SELECT f.tx_id, t.split, t.is_laundering y, {", ".join("f." + c for c in cols)}
        FROM features f JOIN transactions t USING (tx_id) ORDER BY f.tx_id
    """).df()
    num = [c for c in cols if c not in CATEGORICAL]
    df[num] = df[num].astype("float32")
    return df


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
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), [c for c in cols if c not in cat]),
    ])
    model = make_pipeline(prep, LogisticRegression(class_weight="balanced", max_iter=1000))
    train = df[df.split == "train"]
    model.fit(train[cols], train.y)
    save_scores(con, "lr", feature_set, df, model.predict_proba(df[cols])[:, 1])


def main():
    con = duckdb.connect(str(DB))
    what = sys.argv[1:] or ["all"]
    if what == ["all"]:
        is_ach(con)
        lr(con, "A")
    elif what == ["is_ach"]:
        is_ach(con)
    elif what[0] == "lr" and len(what) == 2:
        lr(con, what[1])
    else:
        sys.exit("usage: train.py all | is_ach | lr <feature_set>")


if __name__ == "__main__":
    main()
