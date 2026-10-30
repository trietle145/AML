# Phase 4 research: published work on IBM AML HI-Small

Read-only literature and code review, 2026-10-31. Every claim links to a paper PDF or abstract page, repo source, or dataset page. I extracted PDF text with pdfminer. Where the extraction scrambled a table, the note says so. "Unverified" means I could not confirm it from a primary source.

## Summary

- **Published HI-Small numbers use a different test set from ours.** IBM's Multi-GNN code splits by whole days, about 60/20/20, and gives every day after the val cut to test ([data_loading.py](https://github.com/IBM/Multi-GNN/blob/main/data_loading.py)). FraudGT lists the HI-Small split as 64/19/17 ([FraudGT, Table 1](https://jshun.csail.mit.edu/FraudGT.pdf)), which matches our day cuts exactly ([decisions.md](decisions.md)). So their test is most likely our test plus our `tail` (Sep 11–18: 1,108 rows, 655 laundering, no UNLISTED). That is 655 easy positives added to our 956. This is my inference from the code and the 64/19/17 ratio. No paper says it outright.
- **The GNN papers are transductive within the split.** In Multi-GNN and FraudGT, the test graph holds every test-period edge, and neighbour sampling never filters by time. A Sep 9 transaction can therefore draw messages from Sep 10 and tail edges. The GFP authors say so themselves: GNN baselines are "effectively an offline solution with batch size ∞" ([GFP, §4](https://arxiv.org/abs/2402.08593)).
- **GFP + gradient boosting is the comparable reference.** It computes features in time order from past edges, with a 128-edge batch whose members see each other. HI-Small minority-class F1 is 62.9 (LightGBM) and 63.2 (XGBoost), against 21.3 and 19.8 on basic features alone ([GFP, Table 5](https://arxiv.org/abs/2402.08593)). In that setup, GBT plus features beats GIN, GIN+EU and PNA. This matches our result that the GNN ≈ the MLP.
- **Which GFP feature groups matter (HI-Small, LightGBM):** fan/degree +29.6 F1, multi-hop patterns (scatter-gather, cycles up to length 10, temporal cycles) +3.8, vertex statistics (amount and timestamp moments over one day, per endpoint and direction) +8.2 ([GFP, Table 5](https://arxiv.org/abs/2402.08593)). We have nothing comparable to scatter-gather counts, cycles of length 4–10, temporal-cycle bins or timestamp-moment features.
- **The weak patterns are weak in the published work too.** Multi-GNN per-pattern recall on HI-Small: CYCLE 40–62% and bipartite 35–58%, against 80–95% for fan and scatter/gather patterns. The authors blame two message-passing layers ([Egressy et al., App. I, Table 13](https://arxiv.org/pdf/2306.11586v3)). STACK and RANDOM are not reported separately.
- **Nobody detects UNLISTED.** Multi-GNN recall on "none" (no listed pattern) is 0.5–4.3% on HI-Small: "Lone illicit transactions are very difficult to identify" ([Egressy et al., §7.2, App. I](https://arxiv.org/pdf/2306.11586v3)). The generator makes these transactions as disguised integration (payroll, supplies) and labels funds transitively ([Altman et al., §3.4, App. A](https://arxiv.org/pdf/2306.16424)). Our ~63% recall ceiling is the expected outcome, not a bug.
- **What to try:** (1) for comparability, report F1 on test ∪ tail at threshold 0.5, after triet OKs reading test scores again; (2) a set-D trial of GFP-style features: scatter-gather counts binned by number of intermediates (6 h window), cycles of length 2–10 and temporal cycles (1 d window, binned by length), timestamp and amount moments per endpoint; (3) a biclique (K2,2) count aimed at BIPARTITE and STACK, which no library computes.

## 1. Papers by the dataset authors

### Split, graph and metric

| Paper | Split | Do val/test see future edges? | Metric |
|---|---|---|---|
| Altman et al. 2023, NeurIPS D&B ([arXiv 2306.16424](https://arxiv.org/pdf/2306.16424)) | "60-20-20 temporal train-validation-test split ... after ordering them by their timestamps" (§4) | GNN: "The test graph contains all the transactions, but only the test indices are used for evaluation" (§4). GBT: GFP processes transactions "in the increasing order of their timestamps ... prevents data leakage" (App. D) | Minority-class F1 (§4); PR curves mark the F1 "obtained using the prediction threshold of 0.5" (Fig. 9, 11) |
| Egressy et al. 2024, AAAI ([arXiv 2306.11586v3](https://arxiv.org/pdf/2306.11586v3)) | Same 60-20-20 temporal split and same three snapshots (App. E.1) | Yes, the same snapshot protocol. Neighbour sampling of 100 one-hop and 100 two-hop neighbours (App. F.4) | Minority-class F1, mean ± std over 5 seeds (§6, App. F.1) |
| Blanuša et al. 2024, ICAIF ([arXiv 2402.08593](https://arxiv.org/abs/2402.08593)) | 60% oldest transactions train, next 20% val, rest test (§3, "Data split") | GFP: train is processed before val, and test streams in batches after `fit` on past transactions (§3). Within a batch (128 or 2048), edges see each other. "Batch size of ∞ ... future transactions are also visible" | Minority-class F1, 5 runs (§4) |

**Multi-GNN code, re-verified** ([data_loading.py](https://github.com/IBM/Multi-GNN/blob/main/data_loading.py)):

```python
n_days = int(timestamps.max() / (3600 * 24) + 1)
split_per = [0.6, 0.2, 0.2]
split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
e_val = np.concatenate([tr_inds, val_inds])
val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,  edge_attr,  y,  timestamps
```

- **Test graph = all edges.** The val and test loaders are `LinkNeighborLoader`s with `edge_label_index=te_data.edge_index[:, te_inds]`, and `train_util.py` applies no time filter ([train_util.py](https://github.com/IBM/Multi-GNN/blob/main/train_util.py)). So the sampled neighbourhood of a test edge includes later test-period edges. Labels are not leaked, but future structure is. This confirms our earlier reading.
- **The last days go to test.** The split covers every day up to `timestamps.max()`, and nothing is dropped. The Kaggle page notes "there are some transactions after the specified date period, and that those transactions are all laundering" ([Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)).
- **Timestamp as a feature.** The timestamp is shifted to start at 0, kept as the first edge-attribute column and z-normalised ([data_loading.py](https://github.com/IBM/Multi-GNN/blob/main/data_loading.py), [format_kaggle_files.py](https://github.com/IBM/Multi-GNN/blob/main/format_kaggle_files.py)). That is an absolute-time feature, which our rules forbid. Node features are all ones (`np.ones(max_n_id)`), so there are no IDs.
- **Model selection** is by best val F1, and test F1 is logged at that epoch ([training.py](https://github.com/IBM/Multi-GNN/blob/main/training.py)).
- **FraudGT copies the protocol:** `e_test = torch.cat([train_inds, val_inds, test_inds])`, with no time filtering of neighbours ([aml_dataset.py](https://github.com/junhongmit/FraudGT/blob/main/fraudGT/datasets/aml_dataset.py)).

### Reported HI-Small numbers (minority-class F1, %)

All rows use the protocols above: the GNNs see future edges within the split, and GFP is past-only apart from within-batch visibility. None of these numbers is directly comparable to our test F1 of about 0.40 (no tail, strictly past).

| Model | HI-Small F1 | Source |
|---|---|---|
| LightGBM, basic features only | 21.30 ± 0.30 | [GFP, Table 4](https://arxiv.org/abs/2402.08593) |
| XGBoost, basic features only | 19.75 ± 0.89 | same |
| GFP + LightGBM (batch 128 / 2048) | 62.86 ± 0.25 / 60.52 ± 0.59 | same; 62.86 also in [Altman, Table 2](https://arxiv.org/pdf/2306.16424) |
| GFP + XGBoost (batch 128 / 2048) | 63.23 ± 0.17 / 64.77 ± 0.47 | same |
| GIN | 28.70 ± 1.13 | [Altman, Table 2](https://arxiv.org/pdf/2306.16424) |
| GIN + EU (edge updates) | 47.73 ± 7.86 | same |
| PNA | 56.77 ± 2.41 | same |
| GIN + ReverseMP / +Ports / +EgoIDs (= Multi-GIN), cumulative | 46.79 / 56.85 / 57.15 | [Egressy, Table 2](https://arxiv.org/pdf/2306.11586v3) |
| GIN + EgoIDs alone / GIN + Ports alone | 39.65 / 54.85 | same |
| Multi-GIN+EU / Multi-PNA / Multi-PNA+EU | 64.79 / 64.59 / 68.16 ± 2.65 | same |

- **Precision and recall** (Altman App. E): PNA has recall 53.15 and precision 58.48, GIN+EU recall 55.41 and precision 42.29 ([Altman, Tables 13–14](https://arxiv.org/pdf/2306.16424)).
- **Adaptations:** reverse MP plus port numbering do most of the work on Small HI (28.7 → 56.9). Ego IDs "do not make much difference here" ([Egressy, §7.2](https://arxiv.org/pdf/2306.11586v3)). Port numbers are computed in advance, with ports ordered by timestamp where available (§4.2).
- **Other setup details:** two GNN layers and hidden size 64 ([Altman, App. E](https://arxiv.org/pdf/2306.16424)); minority class weight tuned in (6, 8) ([Altman, Table 11](https://arxiv.org/pdf/2306.16424)). Our DEFAULTS sit in the same range.

## 2. Which features catch which patterns

### What GFP computes

These definitions come from [GFP §2.2–2.3](https://arxiv.org/abs/2402.08593) and the [Snap ML GraphFeaturePreprocessor docs](https://snapml.readthedocs.io/en/latest/graph_preprocessor.html).

- **Fan-in / fan-out:** the number of distinct incoming or outgoing neighbour accounts. **Degree-in / degree-out:** the number of incident edges. **Ratio:** fan/degree. All are computed for the source and the target account, within the time window.
- **Gather-scatter:** "detected implicitly if the fan-in and fan-out of a vertex v are at least two" (§2.2).
- **Scatter-gather:** for edge u→v, phase 1 treats v as an intermediate. For each out-neighbour w of v, it intersects N⁺(u) with N⁻(w) inside the window [t − δ, t], and reports a pattern if |I| ≥ 2. Phase 2 is the mirror case (Algorithm 1). The feature is the number of scatter-gather patterns the transaction belongs to, binned by number of intermediate vertices (Fig. 6).
- **Simple cycles:** hop-constrained, with `lc-cycle_len` defaulting to 10. **Temporal cycles:** "a simple cycle with edges ordered in time". Both are counted per transaction and binned by length; Fig. 6 shows bins 2, 3, …, ≥30.
- **Vertex statistics:** over a chosen column (amount, timestamp), for each endpoint and direction: fan, degree, ratio, average, sum, min, max, median, var, skew, kurtosis. The default `vertex_stats_feats` is fan, degree, ratio, average, sum, var, skew, kurtosis ([docs](https://snapml.readthedocs.io/en/latest/graph_preprocessor.html)).
- **Paper configuration:** a 6 h window for scatter-gather, 1 d for everything else, cycle length ≤ 10, vertex statistics on "Amount" and "Timestamp", batch 128 ([Altman, App. D](https://arxiv.org/pdf/2306.16424); [GFP §3](https://arxiv.org/abs/2402.08593)). IBM's public demo notebook uses `'scatter-gather_tw': 6*3600`, `'lc-cycle_len': 10` and 1-day windows elsewhere ([aml_model_training.ipynb](https://github.com/ambitus/aionz-st-anti-money-laundering/blob/main/zST-model-training-jupyter/aml_model_training.ipynb)).
- **SHAP example:** for one flagged HI-Small transaction, the two top features were "Temporal Cycle length 2" (a 2-hop temporal cycle, i.e. A→B then B→A) and the target account's received-amount sum ([GFP §4, Fig. 10](https://arxiv.org/abs/2402.08593)).
- **Egressy's LightGBM+GFs** enumerates degree-in/out, fan-in/out, directed and undirected cycles, and scatter-gather of different sizes, and tunes the time range as a hyperparameter. "We leave out bicliques, as these are not currently included in the library" ([Egressy, App. F.2, F.4](https://arxiv.org/pdf/2306.11586v3)).

### GFP feature-group ablation (HI-Small, minority-class F1)

Each row adds a group to the one above. Source: [GFP, Table 5](https://arxiv.org/abs/2402.08593). The feature computation is past-only plus within-batch visibility. The test set includes the tail (see §1).

| Features | LightGBM (batch 128) | XGBoost (batch 128) |
|---|---|---|
| basic | 21.30 | 19.75 |
| + fan-in / fan-out | 50.85 | 56.88 |
| + multi-hop (scatter-gather, simple and temporal cycles) | 54.66 | 58.60 |
| + vertex statistics | 62.86 | 63.23 |

### Per-pattern recall (Egressy, App. I, Table 13, HI-Small, %)

"B-C" is bipartite and "none" is no listed pattern. The paper has no STACK or RANDOM columns, and I could not verify where those rows went. The threshold is presumably 0.5, as in training (unverified). The protocol is the transductive one from §1.

| Model | Fan-in | Fan-out | Cycle | S-G | G-S | B-C | none |
|---|---|---|---|---|---|---|---|
| GIN | 40.65 | 64.89 | 44.44 | 51.46 | 47.24 | 34.85 | 4.31 |
| GIN + ReverseMP | 58.54 | 80.92 | 40.40 | 72.80 | 68.50 | 36.36 | 2.43 |
| + Ports | 67.48 | 91.60 | 48.48 | 82.85 | 84.51 | 51.52 | 1.08 |
| + EgoIDs (Multi-GIN) | 87.80 | 80.15 | 50.51 | 89.96 | 86.35 | 51.52 | 1.89 |
| Multi-GIN+EU | 88.62 | 95.42 | 57.58 | 88.28 | 94.49 | 57.58 | 1.35 |
| Multi-PNA | 85.37 | 93.13 | 61.62 | 93.72 | 92.65 | 54.55 | 0.54 |

The authors' explanation for cycle and bipartite: "our best-performing GNN models used only two message passing layers, making it impossible to detect longer cycles" (App. I). The same weak spots show up in our results.

### Pattern definitions and sizes

- **Definitions** ([Altman §3.2](https://arxiv.org/pdf/2306.16424)): fan-out/in means k ≥ 2 distinct counterparties. RANDOM "is similar to the cycle pattern ... except ... funds are not returned to their original account ... a random walk among accounts owned or controlled". BIPARTITE "moves funds from a set of input accounts to a set of output accounts". STACK "extends the bipartite pattern by adding an additional bipartite layer". The laundering entity controls every node in the pattern.
- **Sizes:** Altman Table 6 gives a histogram of nodes per pattern, but for **LI-Large**, not HI-Small. The text extraction scrambled the bin edges. As far as I can read them, the bins run up to "18 ∞" with zero counts in the last bin, so patterns have at most about 18 nodes. Cycles spread across the bins from small to 12–18 nodes ([Altman, App. A](https://arxiv.org/pdf/2306.16424)). The exact cycle-length distribution for HI-Small is unverified. It is cheap to measure from our own `pattern_tx` table: our decisions file already gives 5–14 transactions per attempt ([decisions.md](decisions.md)). Our time-ordered 3-cycle feature can only fire on the shortest cycles.
- **Theory:** Multi-GNN with ego IDs, ports and reverse MP can in principle detect directed cycles, scatter-gather and bicliques (Theorem 4.4). On synthetic graphs it reaches biclique F1 ≈ 94–97 and 6-cycle F1 ≈ 89–95 with six layers ([Egressy, Table 1, Table 8](https://arxiv.org/pdf/2306.11586v3)). That does not carry over to bipartite recall on AML with two layers (≈ 52–58%).

## 3. UNLISTED (laundering outside the 8 patterns)

- **Dataset page:** "We note that not all laundering in the data follows one of these 8 patterns." The page describes placement ("Sources like smuggling of illicit funds") and integration ("Spending the illicit funds") ([Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)).
- **How the generator makes them** ([Altman](https://arxiv.org/pdf/2306.16424)):
  - "AMLworld also makes some laundering transfers happen 'naturally', e.g. a criminal boss launders money by paying employees, buying supplies, etc." (§3.4)
  - Table 8 splits laundering into pattern transactions and "the integration transactions disguised as other activities (e.g. employee payroll or company supplies)" (App. A).
  - The labels are transitive: "If A pays B $100 in illicit funds, and B then pays $50 of those funds to C ... the $50 and the $25 are also [laundering]", including "co-mingling of funds from legitimate and non-legitimate sources" (§3.4).
  - Placement comes from 9 crime types, and "AMLworld tags illicit funds from their origin ... through all transactions used to launder those funds" (§3.3).
- **Shares:** in LI-Large, 81,143 of 100,604 laundering transactions (81%) are "Other" (Altman, Table 8; the numbers are extracted from a scrambled table but sum correctly). For Small HI, 38% of laundering is outside patterns, and for Small LI 71% ([Egressy, App. I](https://arxiv.org/pdf/2306.11586v3)). This matches our 38% UNLISTED.
- **Has anyone detected them?** Not in any primary source I found. Multi-GNN recall on "none" is 0.54–4.31% on HI-Small (table above). The authors: "Lone illicit transactions are very difficult to identify" ([Egressy §7.2](https://arxiv.org/pdf/2306.11586v3)). Our own train+val analysis found that 34.6% of UNLISTED rows touch an account with earlier laundering, against 9.6% for licit rows ([unlisted-exploration.md](../reports/unlisted-exploration.md)). That fits the taint mechanism above, but it rests on labels.

## 4. Other approaches on this dataset

Comparability legend: **T** = test graph includes future test-period edges (the Multi-GNN protocol), **+tail** = test includes the post-Sep-10 days, **P** = past-only features.

| Work | Model / features | Split / transductive? / metric | HI-Small result | Comparable to us? |
|---|---|---|---|---|
| FraudGT, ICAIF 2024 ([paper](https://jshun.csail.mit.edu/FraudGT.pdf), [code](https://github.com/junhongmit/FraudGT/blob/main/fraudGT/datasets/aml_dataset.py)) | Graph transformer with edge-gated MP and edge-attribute attention bias; + RMP/ports/ego variants | Temporal 64/19/17, T, +tail; F1 | FraudGT 69.68 ± 1.58; PE-FraudGT 76.41 ± 1.45; Multi-FraudGT 76.13 ± 0.95. The PDF table is scrambled: I matched columns to datasets via the GFP numbers, and the Multi-FraudGT column order is inferred. The MLP baseline scores 0.42, but it uses raw node/edge features only | No (T, +tail) |
| MEGA-GNN, TMLR 2026 ([arXiv 2412.00241](https://arxiv.org/abs/2412.00241)) | Neighbour-aware aggregation of parallel edges (multigraph), ego IDs, bi-directional MP | "Same temporal splits as Multi-GNN"; F1 | MEGA-PNA 74.01 ± 1.55, MEGA-GIN 70.83, Multi-FraudGT 75.81 ± 0.75 (Table 16) | No (T, +tail) |
| SALT-GNN, 2026 ([arXiv 2607.10131](https://arxiv.org/abs/2607.10131)) | Statistics-aware attention for dense neighbourhoods | "Strictly temporal splits" following Egressy (graph construction not stated); F1 at threshold 0.5; **PR-AUC** in App. F | PR-AUC overall: GIN 0.410, PNA 0.559, FraudGT 0.664, SALT-Trans 0.702; F1 0.716 (Table 10, §5) | Likely T (unverified); the only published HI-Small PR-AUCs I found |
| Adversarial Multi-GNN, 2026 ([arXiv 2610.31718](https://arxiv.org/abs/2610.31718)) | Multi-GNN-style two-layer model, structural attacks, adversarial training | Calls it a "temporal inductive split" following [3]; PR-AUC | Clean PR-AUC 69.4 (standard), 70.4 (adv. trained 20%) (Fig. 6) | Probably the Multi-GNN protocol (unverified) |
| ProtoCP (TGN backbone), KDD 2026 ([arXiv 2608.15768](https://arxiv.org/abs/2608.15768)) | TGN continuous-time memory GNN + conformal prediction | 55/25/20 temporal; "No future information is accessible during any stage" (App. E); F1 | Backbone accuracy 99.81, F1 75.78 (Table 9; supplementary, few details) | Closest in spirit (streaming, past-only), but the threshold, test range and tail handling are unverified |
| GFP + GBT ([arXiv 2402.08593](https://arxiv.org/abs/2402.08593)) | See §2 | 60/20/20 by transaction, P (+ within-batch), +tail; F1 | 62.9–64.8 | Closest published reference |
| BlazingAML, 2026 ([arXiv 2604.12241](https://arxiv.org/abs/2604.12241)) | Fast fuzzy pattern mining (fan, degree, cycle, scatter-gather counts) → XGBoost | First 80% train, last 20% test; "XGB is the baseline with source and destination account ID"; F1 | 46.4 (its own comparison lists FraudGT at 69.6) | No: uses account IDs |
| Amatriciana, 2025 ([arXiv 2506.00654](https://arxiv.org/abs/2506.00654)) | GraphSAGE-like temporal GNN, **account** (node) classification | Validation from "a label-based stratified subset of nodes", not by time; F1 | F1 0.76 (account level) | No: different task and split |
| Deprez et al., review + experiments ([arXiv 2405.19383](https://arxiv.org/abs/2405.19383)) | Transactions-as-nodes graph; network features, DeepWalk etc., GNNs | 500k-transaction subsample (1,356 laundering, 0.27%), 60-20-20 by time; AUC-PR | Mixed: "Only the egonet features improve performance" for supervised models, while DeepWalk-type representations beat the supervised methods on AUC-PR in another table | No: subsample, different graph |
| GPT-4o in-context learning ([arXiv 2507.14785](https://arxiv.org/abs/2507.14785)) | k-hop subgraph serialised into the prompt, 8 pattern + 4 benign examples | **Balanced** test, 1,000 + 1,000; accuracy/F1 | F1 65.1 ± 3.9, accuracy 63.7 | No: balanced |
| Kaggle "Anti Money Laundering Detection with GNN" ([Kaggle](https://www.kaggle.com/code/issacchanjj/anti-money-laundering-detection-with-gnn), [notebook source](https://github.com/issacchan26/AntiMoneyLaunderingDetectionWithGNN/blob/main/anti-money-laundering-detection-with-gnn.ipynb)) | GAT, account classification (an account is positive if it touched any laundering) | `RandomNodeSplit(num_val=0.1, num_test=0)`, random, no test set; accuracy | "accuracy: 0.97" on val | No: random split, accuracy, account labels |
| steveee27 ([notebook](https://github.com/steveee27/Money-Laundering-Detection/blob/main/Money_Laundering.ipynb)) | RF / XGBoost on HI-Large | `RandomUnderSampler` on the full data, then a random `train_test_split`, so the test set is balanced and random | Accuracy ~0.88–0.90 (README) | No: balanced, random (and HI-Large, not HI-Small) |

**Kaggle vote ranking:** unverified. Kaggle's listing pages render with JavaScript and the kernels API returns 401 without auth, so I could not rank notebooks by votes. The GNN notebook above is the one search surfaces.

## 5. Practical takeaways for us

1. **Make our headline comparable before claiming a gap.** IBM's test sets almost certainly include the tail: 655 positives that are 59% of their days' rows and contain no UNLISTED. They also report F1 at threshold 0.5, and their GNN neighbourhoods see future edges. Our test F1 of ~0.40 against their 63 is not like for like.
   - Cheapest check: compute F1 and PR-AUC on test ∪ tail from the existing scores. This reads test scores again, so ask triet first.
   - Rough arithmetic, illustrative only: if we keep our test TP/FP and catch about 90% of the tail with few extra FPs, F1 moves from ~0.40 to ~0.6.
2. **Feature ideas with published evidence (set D candidates), all computable strictly from the past within our ≤ 3 d rule:**
   - **Scatter-gather counts per transaction**, binned by number of intermediates (2, 3, ≥4…), 6 h window. This is GFP Algorithm 1 run on past edges only. It is part of the +3.8 F1 multi-hop group ([GFP Table 5](https://arxiv.org/abs/2402.08593)). BlazingAML ranks scatter-gather as the strongest of its pattern counts, but that test includes account IDs ([BlazingAML §8.1](https://arxiv.org/abs/2604.12241)).
   - **Cycle counts of length 2–10**, simple and temporal, 1 d window, binned by length. Aimed at CYCLE: our 3-cycles miss longer cycles, and in GFP's SHAP example the 2-hop temporal cycle ranks top ([GFP Fig. 10](https://arxiv.org/abs/2402.08593)). Bounded DFS from the receiver back to the sender over past edges in the window should be enough at HI-Small scale.
   - **Vertex statistics including timestamp moments:** var, skew and kurtosis of incident-edge timestamps and amounts, for sender and receiver, in and out, 1 d window. This group was worth +8.2 F1 on top of everything else ([GFP Table 5](https://arxiv.org/abs/2402.08593)). Check overlap with set B first.
   - **Fan/degree ratio** for both endpoints in both directions, if not already present.
3. **BIPARTITE / STACK:** no library feature exists (GFP omits bicliques), and Multi-GNN recall stays at 52–58%. One idea, untested, no published evidence: count K2,2 bicliques that contain edge u→v in the window. That is the number of pairs (u′, w) with u′→v, u→w and u′→w. This is a set intersection like scatter-gather. For STACK, apply the same count on both sides of v.
4. **RANDOM:** no published feature targets it. An idea only: the length of the time-respecting, amount-preserving forward chain ending at the sender (a multi-hop pass-through).
5. **GNN direction:** on HI-Small, the published gains come from reverse MP + ports (+28 F1 for GIN), in a protocol where neighbourhoods include future edges. Ego IDs add nothing on Small HI ([Egressy §7.2](https://arxiv.org/pdf/2306.11586v3)). Port numbers are a cheap precomputed edge feature (neighbour index ordered by time) if we revisit the GNN. In a past-only, feature-rich setting, GFP shows GBT beating the GNNs. That is consistent with our GNN ≈ MLP, so more GNN work is low priority.
6. **UNLISTED and delayed labels:** the generator labels by taint propagation from criminal entities through ordinary-looking payroll and supply payments ([Altman §3.4](https://arxiv.org/pdf/2306.16424)). So the only signal with a mechanism behind it is "this account recently handled known-illicit funds". In our data that is 34.6% of UNLISTED rows against 9.6% of licit rows ([unlisted-exploration.md](../reports/unlisted-exploration.md)). With an UNLISTED base rate of 0.04%, a flag that also fires on ~10% of licit rows would have very low precision.
   - I found no primary source that uses prior-label (prior-SAR) features on this dataset or reports detecting UNLISTED (unverified for the wider literature).
   - If tried, use a label delay of at least the realistic investigation lag (e.g. labels only from ≥ D days before the scored row), evaluate on val only, and expect a small effect. Otherwise accept the ~63% recall ceiling as the published state of the art.
