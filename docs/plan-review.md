# Plan review: "Follow the Money"

*Reviewed 2026-10-28. The project folder `D:\AML detection` is empty, so this reviews the plan only.*

## Verdict

The idea is good and the story is strong: pattern-level detection, a clear research question, explainable alerts. The weak spots are not the idea but the details the plan doesn't pin down yet: which dataset variant, what exactly is being predicted, how train/test are split in time, and how much of the "output" layer to build. Fix those up front and this is a very solid portfolio project. Leave them open and the headline result ("graph features help") can easily come out inflated by leakage.

## Strengths

- **Right framing.** Laundering is a network problem; the fan-in/fan-out/cycle/scatter-gather table makes that obvious to a non-technical reader.
- **Testable key question.** "Graph features vs per-transaction features" is a clean ablation: same model, same split, one feature group toggled.
- **Realistic angle.** Explainability and human review match how transaction monitoring actually works.
- **Public, well-known dataset** with a published benchmark (Altman et al., 2023, "Realistic Synthetic Financial Transactions for AML Models"), so your numbers can be sanity-checked against the literature.

## Gaps and risks

### 1. Dataset variant is unspecified
The Kaggle release has six files: HI/LI × Small/Medium/Large. HI = higher illicit ratio, LI = lower. Small is ~5–7M transactions; Large is ~180M+ and not laptop-friendly.
**Recommendation:** start on **HI-Small** (≈5M rows, ≈0.1% laundering). Optionally rerun the final pipeline on LI-Small as a robustness check ("does it hold when laundering is rarer?").

### 2. Label leakage from pattern types
Pattern types (FAN-IN, CYCLE, etc.) come from a separate `*_Patterns.txt` file, not a column. They are ground truth about *how* the laundering was generated.
- **Never** use them as features or to engineer features directly.
- **Do** use them for evaluation: recall broken down by pattern type is one of the most interesting outputs you can show ("graph features catch cycles; transaction features don't").
- Note that not every laundering transaction belongs to a listed pattern, so a breakdown needs an "unlisted" bucket.

### 3. Temporal leakage in graph features
This is the biggest risk. If you compute "number of senders", "cycle membership" or centrality on the **whole** graph, a training transaction's features see the future, and test-period edges leak into training. Graph features will look far better than they really are.
**Recommendation:**
- Split by time (e.g. first 60% of days train, next 20% validation, last 20% test), the same scheme the benchmark papers use.
- Compute every feature using only transactions **at or before** the transaction's timestamp (rolling windows: 1h / 24h / 7d), not the full graph.
- No random train/test split anywhere.

### 4. Transaction-level vs account-level target is undecided
Labels are per transaction (`Is Laundering`). The plan talks about flagging *accounts* in explainability and the dashboard.
**Recommendation:** model at **transaction level** (matches the labels and the benchmark), then **aggregate to account level** for alerts (e.g. max or sum of transaction scores per account per day). Define an account-level label explicitly ("account has ≥1 laundering transaction in the period") if you report account metrics. Also: identify accounts by **bank + account ID** together.

### 5. Metrics for extreme imbalance
"Precision/recall" is right but needs to be concrete at ~0.1% positives. Accuracy and ROC-AUC will both look great and mean little.
**Recommendation:**
- Primary: **PR-AUC** and **minority-class F1** (the benchmark's metric, so comparable).
- Business view: **precision and recall at a fixed alert budget** (e.g. top 100 alerts/day), which is how investigators actually work.
- Recall **by pattern type** (see 2).
- For "significantly better": bootstrap confidence intervals on the test set, or several seeds. Otherwise "significantly" is just a word.

### 6. Shortcut features in the baseline
In this synthetic data some per-transaction fields (notably payment format, e.g. ACH, and currency mismatches) are strongly associated with laundering. That makes the per-transaction baseline stronger than you might expect, which is good for honesty but worth inspecting in EDA so the comparison isn't a surprise.

### 7. The key question is already partly answered
The dataset paper and follow-up work (e.g. Multi-GNN, Egressy et al. 2024) already show that adding graph features to a GBM helps a lot on this data. That doesn't kill the project, but frame it as "reproduce and explain the gain" (which patterns, which features, what an investigator sees) rather than as a new discovery. Citing the benchmark makes you look more credible, not less.

### 8. Scope creep
SQL layer + two feature families + three models + SHAP + dashboard + interactive network view is a lot. The dashboard and network viewer can easily eat as much time as the modeling.
**Recommendation:** treat the dashboard as the last, thinnest layer. Build it only once the ablation result exists.

### 9. Practical/compute
- Cycle detection and centrality on 5M edges with NetworkX is slow. Use **bounded, time-windowed** features (2–3 hop cycles within N days, rolling in/out degree, pass-through ratio) computed in SQL, and skip global centrality unless it earns its place.
- Synthetic data: say so in the write-up, and don't claim real-world performance.

## Suggested scope

| Stage | Keep it to |
|---|---|
| Data | HI-Small CSV → **DuckDB** (real SQL, no server, fast on 5M rows) |
| EDA | Volumes, timing, currencies, payment formats, laundering rate over time |
| Split | Time-based 60/20/20 |
| Features | (a) per-transaction: amount, currency, format, hour, cross-bank; (b) behavioral per account, rolling windows; (c) graph: fan-in/out counts, pass-through ratio & speed, short cycles, all past-only |
| Models | Logistic regression baseline; LightGBM with (a)+(b) vs (a)+(b)+(c) |
| Eval | PR-AUC, minority F1, precision@alert budget, recall by pattern type, bootstrap CIs |
| Explain | SHAP + plain-English alert reasons ("received from 14 accounts in 3h, forwarded 95% within 1h") |
| Output | Streamlit app: alert list, metrics, and a 2-hop ego-graph of a flagged account (pyvis). Nothing more. |

Stretch only if time remains: LI-Small robustness run, a GNN comparison.

## Next steps

1. Download **HI-Small_Trans.csv** and **HI-Small_Patterns.txt** from Kaggle into `D:\AML detection\data\raw\`.
2. Load into DuckDB and do the EDA, including the laundering rate per day (check where the time split should fall).
3. Write down the target, split dates and metrics before building features, so the comparison is fixed in advance.
4. Build per-transaction + behavioral features → baseline results.
5. Add graph features → ablation → that's the headline answer.
6. Explainability, then dashboard.
