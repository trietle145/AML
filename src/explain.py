import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from load import DB, ROOT
from metrics import top_k_per_day
from train import FEATURE_SETS, load

COLS = FEATURE_SETS["ABC"]
REPORTS = ROOT / "reports"


# every lgbm/ABC feature in exactly one group. Features that encode the same fact split credit into large opposite
# values (a receiver with no past receipts has no pair history and no fan-in either), so they share a group:
# median sum|group| / |total| on val alerts is 1.22 here vs 2.23 per feature and 1.55 with activity, pair and fan-in apart
def group(c):
    if c.startswith("pair_rev") or c.startswith("cyc3"):
        return "loop back to sender"
    if c.startswith("pair_") or c.startswith("in_cpty") or c.startswith(("out_cnt", "out_min", "out_has", "in_cnt", "in_min", "in_has")):
        return "account history"
    if c.startswith("out_cpty"):
        return "sender fan-out"
    if c.startswith(("out_recv", "in_send")) or "pass_ratio" in c:
        return "pass-through"
    if c == "payment_format" or "ach_share" in c:
        return "payment format"
    if c == "log_amount_usd" or "_usd_" in c or "amt_vs_mean" in c:
        return "amount"
    if c == "hour":
        return "time of day"
    return "currency and bank"


# one plain-English sentence per group, from the alert's own feature values
def say(g, r):
    if g == "amount":
        x = lambda v: f"{v:.0f}x" if v >= 10 else f"{v:.2g}x"
        vs = [f"{x(getattr(r, f'{s}_amt_vs_mean_3d'))} the {who}'s 3-day average" for s, who in (("in", "receiver"), ("out", "sender"))
              if pd.notna(getattr(r, f"{s}_amt_vs_mean_3d"))]
        return f"amount ${10 ** r.log_amount_usd:,.0f}" + (f" ({', '.join(vs)})" if vs else ", neither account has 3-day history to compare with")
    if g == "payment format":
        return f"paid by {r.payment_format}" + (f"; {r.out_ach_share_3d:.0%} of the sender's recent sends are ACH" if pd.notna(r.out_ach_share_3d) else "")
    if g == "account history":
        if not r.in_has_history:
            facts = ["receiver has no receipts in the past 3 days (new or dormant account)"]
        else:
            n = int(r.in_cpty_3d)
            facts = [f"receiver got money from {n} account{'s' * (n != 1)} in 3 days ({r.in_cnt_24h:.0f} receipts in 24 h)"]
        facts.append(f"sender paid this receiver {r.pair_cnt_3d:.0f} times in 3 days" if r.pair_cnt_3d else "first payment from this sender to this receiver")
        if not r.out_has_history:
            facts.append("sender has no sends in the past 3 days")
        return "; ".join(facts)
    if g == "sender fan-out":
        return f"sender paid {r.out_cpty_3d:.0f} different accounts in the past 3 days"
    if g == "pass-through":
        if pd.notna(r.out_pass_ratio_24h):
            return f"sender received ${r.out_recv_usd_24h:,.0f} in the past 24 h and sent out {r.out_pass_ratio_24h:.0%} of it"
        return f"sender received nothing in the past 24 h; receiver sent ${r.in_send_usd_24h:,.0f} onward in that time"
    if g == "loop back to sender":
        if r.cyc3_cnt_3d:
            return f"closes a 3-account loop back to the sender ({r.cyc3_cnt_3d:.0f} middle accounts)"
        return "the receiver paid the sender earlier (money coming back)" if r.pair_rev_cnt_3d else "no money flowed back to the sender"
    if g == "time of day":
        return f"sent at {int(r.hour):02d}:00"
    return f"{r.payment_currency} to {r.receiving_currency}, {'cross-bank' if r.is_cross_bank else 'same bank'}"


def main():
    con = duckdb.connect(str(DB), read_only=True, config={"memory_limit": "700MB", "threads": 2})
    v = load(con, COLS, split="val")
    s = con.sql("""SELECT s.tx_id, s.score, t.day, CASE WHEN t.is_laundering = 0 THEN 'licit' ELSE COALESCE(p.pattern_type, 'UNLISTED') END pattern
                   FROM scores s JOIN transactions t USING (tx_id) LEFT JOIN tx_pattern p USING (tx_id)
                   WHERE s.split = 'val' AND s.model = 'lgbm' AND s.feature_set = 'ABC' ORDER BY s.tx_id""").df()
    assert (s.tx_id.to_numpy() == v.tx_id.to_numpy()).all()
    alert = top_k_per_day(s.score, s.day)
    a, X = s[alert].reset_index(drop=True), v.loc[alert, COLS].reset_index(drop=True)
    del v

    booster = lgb.Booster(model_file=str(ROOT / "data/models/lgbm_ABC.txt"))
    contrib = booster.predict(X, pred_contrib=True)
    raw = contrib.sum(axis=1)
    assert np.allclose(1 / (1 + np.exp(-raw)), a.score, atol=1e-6), "contributions don't reproduce the stored scores"
    groups = pd.Series([group(c) for c in COLS])
    assert len(groups) == 62 and groups.nunique() == 8
    g = pd.DataFrame(contrib[:, :-1], columns=COLS).T.groupby(groups.to_numpy()).sum().T

    rows = []
    for i, r in enumerate(X.itertuples()):
        top = g.iloc[i][g.iloc[i] > 0].sort_values(ascending=False).head(3)
        rows.append({"tx_id": a.tx_id[i], "day": a.day[i], "score": a.score[i], "pattern": a.pattern[i],
                     "groups": "; ".join(f"{k} {v:+.2f}" for k, v in top.items()), "reasons": "; ".join(say(k, r) for k in top.index)})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "phase4-val-alert-reasons.csv", index=False)
    print(f"{len(out)} val alerts at 500/day explained ({(out.pattern != 'licit').mean():.1%} laundering)")
    lead = out.groups.str.split(" [+-]", regex=True).str[0]
    print("\ntop reason group, laundering vs false alarms:")
    print(pd.crosstab(lead, out.pattern == "licit").rename(columns={False: "laundering", True: "false alarm"}).sort_values("laundering", ascending=False).to_string())
    assert out.reasons.str.len().gt(0).all() and len(out) == alert.sum()


if __name__ == "__main__":
    main()
