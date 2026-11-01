# Follow the Money: the look-ahead experiment

*This branch extends the main project (branch `main`: the full pipeline, data analysis, models and results) with one question: **why do published results on IBM's AML HI-Small dataset score so much higher than ours, and what would it take to match them?***

Our best real-time model reaches 0.398 test PR-AUC (F1 0.42). Published graph neural networks report about 0.65–0.76 minority-class F1. The answer, measured below: **published models see future transactions.** Give our own LightGBM the same view of the future and it matches IBM's Multi-GNN exactly.

## Two experiments

1. **IBM's Multi-GNN, run on the same data.** IBM's own code (their best Small-HI GIN setup with edge updates, reverse message passing, ego IDs and port numbers) was trained on Kaggle, and its per-transaction predictions were scored with this project's metrics on this project's test days. Their code and predictions are kept out of this repository; only the comparison is reported.
2. **Set C+: our graph features with look-ahead.** The same 20 graph features as set C (fan-in/out, pass-through, pair history, 3-cycles), but each window reaches **forward** as well as back, and 3-cycles may close in either time order. Everything else stays identical: set A, set B (strictly past), the LightGBM grid, the selection rule and the evaluation. The model is `lgbm/ABC_ahead`.

Set C+ deliberately breaks the project's core rule (features from strictly earlier transactions only). It exists to measure what seeing the future is worth, and is never a deployable or headline model. The exception is recorded in [`docs/decisions.md`](docs/decisions.md).

### What look-ahead changes in the features

| | Rows closing a 3-cycle: past only → with look-ahead | Mean distinct senders to the receiver in 3 days |
|---|---|---|
| Licit | 0.00% → 0.03% | 1.95 → 2.93 |
| **Laundering** | **1.4% → 2.9%** | **2.67 → 5.35** |

A laundering scheme takes about 3 days, so when its first payments happen, the rest hasn't happened yet. Strictly-past features can only fire on the payment that *completes* a structure; look-ahead lets every payment of the scheme see the whole thing.

## Results

**Test (Sep 9–10, scored once):**

| Model | Sees the future? | PR-AUC [95% CI] | P / R at 500 alerts/day | F1 |
|---|---|---|---|---|
| `lgbm/ABC`: LightGBM, graph features | no | 0.345 [0.313, 0.381] | 0.389 / 0.407 | 0.41 |
| `ens/ABC`: LightGBM + MLP (our best) | no | 0.398 [0.366, 0.432] | 0.395 / 0.413 | 0.42 |
| **`lgbm/ABC_ahead`**: same LightGBM, set C+ | **yes** | **0.528** [0.496, 0.560] | 0.489 / 0.512 | 0.57 |
| IBM Multi-GNN (their protocol) | yes | 0.525 [0.494, 0.557] | 0.517 / 0.541 | 0.58 |

- **Look-ahead is worth +0.183 PR-AUC** on test (`lgbm/ABC_ahead` − `lgbm/ABC`, paired bootstrap, significant under row-level and attempt-cluster resampling; +0.120 on val).
- **With the same view of the future, a plain LightGBM matches IBM's GNN:** IBM − `lgbm/ABC_ahead` = −0.003 [−0.026, +0.021] on test, and +0.015 [−0.008, +0.036] on val. Neither is significant.
- **IBM's published number reproduces:** on their test range (our test plus the tail, Sep 11–18) at their 0.5 cutoff, IBM's model reaches F1 **0.71**, against a published ~0.70. That test range is also easier: the tail adds 655 positives from listed schemes winding down, and it lifts our own best model from F1 0.42 to 0.59.

**Recall by scheme at 500 alerts a day (test):**

| Scheme | Rows | Ours, past only (`ens/ABC`) | **Ours, look-ahead** | IBM |
|---|---|---|---|---|
| STACK | 84 | 0.38 | **0.77** | 0.77 |
| BIPARTITE | 44 | 0.45 | **0.73** | 0.73 |
| RANDOM | 43 | 0.53 | **0.72** | 0.74 |
| CYCLE | 55 | 0.44 | **0.67** | 0.78 |
| FAN-OUT | 65 | 0.77 | 0.95 | 0.97 |
| GATHER-SCATTER | 127 | 0.79 | 0.88 | 0.90 |
| SCATTER-GATHER | 110 | 0.80 | 0.87 | 0.96 |
| FAN-IN | 57 | 0.86 | 0.84 | 0.95 |
| UNLISTED | 371 | 0.024 | 0.016 | 0.022 |

- **The gain is where the theory says it should be:** the slow multi-hop schemes (stack, bipartite, random, cycle), whose completing payments the past-only view can't see in time.
- **IBM's remaining edge is long cycles** (0.78 vs 0.67). Set C+ only counts 3-cycles, while a GNN with future edges can follow longer loops.
- **UNLISTED stays near zero with or without the future.** It's ordinary-looking spending of dirty money, not a graph shape.

## What it means

Under real-time rules, where a payment is judged only on what happened before it, about 0.40 PR-AUC is what this data allows, and a tuned LightGBM with hand-made graph features gets there. The higher published scores come from two things a bank doesn't have at decision time: future transactions in the graph, and an easier test period. Neither comes from the model architecture: our tree model with the same look-ahead scores the same as IBM's graph neural network.

## Reproduce

On top of the main pipeline (see the `main` branch README):

```bash
python src/features.py ahead                          # set C+ columns (~3 min)
python src/train.py --phase 4 lgbm ABC_ahead          # LightGBM grid (~7 min)
python src/evaluate.py --split val --phase 4 --tag ahead --models lgbm/ABC_ahead,lgbm/ABC --pairs lgbm/ABC_ahead:lgbm/ABC
```

Results are in `reports/phase4-ahead-*.csv`. The models and features are described in [`docs/model.md`](docs/model.md) and [`docs/features.md`](docs/features.md); the literature review is in [`docs/phase4-research.md`](docs/phase4-research.md).
