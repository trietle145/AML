# Phase 3 results: graph features and a GNN against the frozen baseline

*Rules: `docs/decisions.md`. Models: `docs/model.md`. Val numbers regenerate with `evaluate.py --split val --phase 3 --models lgbm/AB,lgbm/ABC,gnn/ABC,mlp/ABC,gnn_s0/ABC,gnn_s1/ABC,gnn_s2/ABC,mlp_s0/ABC,mlp_s1/ABC,mlp_s2/ABC --pairs lgbm/ABC:lgbm/AB,gnn_s0/ABC:mlp_s0/ABC,gnn_s1/ABC:mlp_s1/ABC,gnn_s2/ABC:mlp_s2/ABC,gnn_s0/ABC:lgbm/ABC,gnn_s1/ABC:lgbm/ABC,gnn_s2/ABC:lgbm/ABC,mlp_s0/ABC:lgbm/ABC,mlp_s1/ABC:lgbm/ABC,mlp_s2/ABC:lgbm/ABC` (CSVs: `reports/phase3-val-*.csv`). GNN run settings and histories: `reports/phase3-gnn-runs.csv`. Test and tail were scored once on 2026-10-30 after Checkpoint B (logs `logs/t6_test.log`, `logs/t6_tail.log`).*

## Val (Sep 7–8; 965,524 rows, 1,036 laundering)

Val was used for early stopping, grid choice, best-epoch choice and thresholds, so all val numbers are optimistic. Test is the unbiased one.

### Headline (`phase3-val-metrics.csv`)

| Model | PR-AUC [iid 95% CI] | Cluster CI | F1 | P@500/day | R@500/day |
|---|---|---|---|---|---|
| `lgbm/AB` (frozen Phase 2 baseline) | 0.393 [0.360, 0.426] | [0.330, 0.451] | 0.466 | 0.446 | 0.431 |
| `lgbm/ABC` (+ set C, hand-made graph features) | 0.444 [0.413, 0.474] | [0.381, 0.499] | 0.479 | 0.456 | 0.440 |
| `mlp/ABC` (T5 baseline settings) | 0.429 [0.398, 0.461] | [0.365, 0.490] | 0.457 | 0.435 | 0.420 |
| `gnn/ABC` (T5 baseline settings) | 0.462 [0.430, 0.492] | [0.400, 0.525] | 0.488 | 0.462 | 0.446 |
| `mlp_s0` / `s1` / `s2` (IBM-style settings) | 0.477 / 0.472 / 0.464 | | 0.486 / 0.487 / 0.488 | 0.474 / 0.466 / 0.459 | 0.458 / 0.450 / 0.443 |
| `gnn_s0` / `s1` / `s2` (IBM-style settings) | 0.483 / 0.463 / 0.463 | | 0.512 / 0.488 / 0.489 | 0.481 / 0.458 / 0.454 | 0.464 / 0.442 / 0.438 |
| **MLP, 3 seeds** | **0.471 ± 0.006** (mean ± sd) | | | | |
| **GNN, 3 seeds** | **0.470 ± 0.011** | | | | |

- **Seeds:** `gnn_s0` = `gnn_ibm` (unclipped). `gnn_s1`/`s2` use gradient clipping, `clip=1.0`. Clipped seed 0 (`gnn_clip_s0`) diverged at epoch 3 and is excluded. `mlp_s0` = `mlp_ibm` (unclipped); `mlp_s1`/`s2` use clipping too. No MLP run diverged.

### Paired differences (`phase3-val-paired.csv`; same rows, same resamples, 1,000×)

| Comparison | PR-AUC diff | iid 95% CI | Cluster 95% CI | Significant? |
|---|---|---|---|---|
| `lgbm/ABC` − `lgbm/AB` | +0.051 | [0.026, 0.075] | [0.025, 0.079] | **yes, both** |
| `gnn_s0` − `mlp_s0` | +0.007 | [−0.008, 0.019] | [−0.008, 0.022] | no |
| `gnn_s1` − `mlp_s1` | −0.009 | [−0.021, 0.004] | [−0.020, 0.005] | no |
| `gnn_s2` − `mlp_s2` | −0.001 | [−0.011, 0.009] | [−0.010, 0.009] | no |
| `gnn_s0` − `lgbm/ABC` | +0.039 | [0.014, 0.063] | [0.009, 0.069] | yes, both |
| `gnn_s1` − `lgbm/ABC` | +0.019 | [−0.004, 0.042] | [−0.009, 0.048] | no |
| `gnn_s2` − `lgbm/ABC` | +0.019 | [−0.004, 0.042] | [−0.010, 0.049] | no |
| `mlp_s0` − `lgbm/ABC` | +0.033 | [0.011, 0.053] | [0.008, 0.059] | yes, both |
| `mlp_s1` − `lgbm/ABC` | +0.028 | [0.005, 0.049] | [0.004, 0.053] | yes, both |
| `mlp_s2` − `lgbm/ABC` | +0.020 | [−0.005, 0.043] | [−0.008, 0.049] | no |

### What it shows

1. **Hand-made graph features help.** Set C adds +0.051 PR-AUC over `lgbm/AB`, significant under iid and attempt-cluster resampling.
2. **Message passing adds nothing measurable on top.** With identical inputs (A + B + C), head and training, the GNN and the MLP differ by −0.009 to +0.007 per seed, every CI includes 0, and the seed means are 0.470 vs 0.471. `gnn_ibm`'s 0.483 was a lucky seed.
3. **The neural nets edge out the trees on val, but not reliably.** The MLP and GNN lead `lgbm/ABC` by +0.02 to +0.04. That's significant for 3 of 6 seeds, and at the operating point it's tiny: at 500 alerts/day everything sits at ~0.45–0.48 precision and ~0.44–0.46 recall. The neural nets were also selected on val more heavily (best epoch out of up to 37), so part of the gap may be selection.
4. **Training settings matter more than architecture.** Moving from the T5 settings (lr 0.001, positive weight 100, one update per hour) to IBM-style ones (lr 0.006, weight 6, 8,192-row batches) lifted the MLP 0.429 → ~0.471 and the GNN 0.462 → ~0.470.
5. **The GNN is less stable at these settings.** 2 of 5 GNN runs diverged (`gnn_w72`, `gnn_clip_s0`); 0 of 4 MLP runs did. Gradient clipping didn't prevent it, because Adam rescales steps anyway; the learning rate is too high for the GNN.

### Recall by pattern at each model's val threshold (`phase3-val-recall-by-pattern.csv`)

| Pattern | n | `lgbm/AB` | `lgbm/ABC` | MLP seeds | GNN seeds |
|---|---|---|---|---|---|
| FAN-IN | 60 | 0.75 | 0.80 | 0.77–0.80 | 0.75–0.80 |
| FAN-OUT | 64 | 0.64 | 0.63 | **0.80–0.84** | **0.72–0.81** |
| GATHER-SCATTER | 96 | 0.80 | 0.81 | **0.88–0.93** | **0.90–0.93** |
| SCATTER-GATHER | 134 | 0.83 | 0.81 | 0.87–0.90 | 0.86–0.90 |
| CYCLE | 66 | 0.44 | 0.38 | 0.30–0.42 | 0.33–0.52 |
| BIPARTITE | 49 | **0.49** | 0.35 | 0.29–0.35 | 0.25–0.29 |
| STACK | 150 | **0.49** | 0.33 | 0.23–0.31 | 0.18–0.29 |
| RANDOM | 38 | 0.37 | 0.29 | 0.26–0.40 | 0.29–0.55 |
| **UNLISTED** | **379** | 0.008 | 0.005 | 0.003–0.018 | 0.008–0.011 |

- **The neural nets are better on fan-out and gather/scatter shapes** and worse on stack and bipartite, where `lgbm/AB` is still best. They trade one kind of scheme for another rather than catching more overall.
- **UNLISTED stays at ~0 for every model** (at most 7 of 379). This is the ~63% recall ceiling documented in Phase 4's `reports/unlisted-exploration.md` (branch `unlisted-diagnosis`).

### `lgbm/ABG` (GNN embeddings into LightGBM): no-go

T6's condition was "the best GNN beats `lgbm/ABC` on val". Taken alone it does (by +0.02 to +0.04), but the MLP does just as well without any graph embeddings, so the embeddings carry nothing the head's inputs don't already have. Stacking them into LightGBM would add the stacking-leak risk for no expected gain.

## Test (Sep 9–10; 862,792 rows, 956 laundering), scored once on 2026-10-30

Same seven models, 10 comparisons, val thresholds, 1,000 resamples. Command: the val command above with `--split test --final` and `lgbm/AB` left out of `--models` (it is only loaded as a pair partner; its Phase 2 test scores are reused).

### Headline (`phase3-test-metrics.csv`)

| Model | PR-AUC [iid 95% CI] | Cluster CI | F1 | P@500/day | R@500/day |
|---|---|---|---|---|---|
| `lgbm/AB` (frozen, Phase 2 test run) | 0.292 [0.261, 0.328] | [0.232, 0.351] | 0.351 | 0.337 | 0.353 |
| `lgbm/ABC` | **0.345** [0.313, 0.381] | [0.286, 0.400] | **0.414** | **0.389** | **0.407** |
| `mlp_s0` / `s1` / `s2` | 0.371 / 0.368 / 0.368 | | 0.359 / 0.357 / 0.388 | 0.377 / 0.382 / 0.369 | 0.394 / 0.400 / 0.386 |
| `gnn_s0` / `s1` / `s2` | 0.372 / 0.375 / 0.379 | | 0.408 / 0.399 / 0.394 | 0.386 / 0.385 / 0.377 | 0.404 / 0.403 / 0.394 |
| **MLP, 3 seeds** | **0.369 ± 0.002** | | 0.368 | 0.376 | 0.393 |
| **GNN, 3 seeds** | **0.375 ± 0.003** | | 0.400 | 0.383 | 0.400 |

### Paired differences (`phase3-test-paired.csv`)

| Comparison | PR-AUC diff | iid 95% CI | Cluster 95% CI | Significant? |
|---|---|---|---|---|
| `lgbm/ABC` − `lgbm/AB` | **+0.053** | [0.024, 0.080] | [0.022, 0.084] | **yes, both** |
| `gnn_s0` − `mlp_s0` | +0.002 | [−0.012, 0.016] | [−0.016, 0.018] | no |
| `gnn_s1` − `mlp_s1` | +0.006 | [−0.007, 0.020] | [−0.009, 0.020] | no |
| `gnn_s2` − `mlp_s2` | +0.011 | [0.001, 0.021] | [−0.001, 0.023] | iid only |
| `gnn_s0` − `lgbm/ABC` | +0.027 | [−0.001, 0.055] | [−0.003, 0.058] | no |
| `gnn_s1` − `lgbm/ABC` | +0.029 | [0.003, 0.055] | [0.001, 0.058] | yes, both |
| `gnn_s2` − `lgbm/ABC` | +0.034 | [0.009, 0.058] | [0.004, 0.065] | yes, both |
| `mlp_s0` − `lgbm/ABC` | +0.026 | [0.001, 0.051] | [−0.003, 0.054] | iid only |
| `mlp_s1` − `lgbm/ABC` | +0.023 | [−0.003, 0.049] | [−0.007, 0.051] | no |
| `mlp_s2` − `lgbm/ABC` | +0.023 | [−0.003, 0.049] | [−0.005, 0.053] | no |

### What it shows

1. **Headline: hand-made graph features beat the per-transaction + behavioural baseline.** `lgbm/ABC` − `lgbm/AB` is +0.053 PR-AUC on test (val +0.051), significant under iid and attempt-cluster resampling. At 500 alerts/day, precision goes 0.337 → 0.389 and recall 0.353 → 0.407.
2. **Message passing still adds nothing that clears the bar.** GNN − MLP is +0.002 / +0.006 / +0.011 per seed, never significant under both CIs (val: +0.007 / −0.009 / −0.001). The GNN is ahead on all three test seeds, by +0.006 on average, which is too small to matter.
3. **The neural nets rank better overall, but not at the operating point.** On PR-AUC they lead `lgbm/ABC` by +0.023 to +0.034, significant under both CIs for 2 of 6 seeds (both GNN). At 500 alerts/day, `lgbm/ABC` matches or beats every one of them (precision 0.389 vs 0.369–0.386, recall 0.407 vs 0.386–0.404), and has the best F1 at its val threshold. Their extra ranking quality sits away from the top 500 alerts a day.
4. **Val → test drop of ~0.1 for every model** (`lgbm/ABC` 0.444 → 0.345, neural nets ~0.47 → ~0.37), the same as Phase 2's `lgbm/AB` (0.393 → 0.292). The ordering holds, so this is a harder period, not overfitting of one model.
5. **UNLISTED stays near 0:** 0.8–3.2% recall (3–12 of 371). The GNN seeds catch slightly more (2.4–3.2%) than `lgbm/ABC` (1.3% [0.3%, 2.7%]), with overlapping CIs (`phase3-test-recall-by-pattern-ci.csv`).

### Recall by pattern at the val threshold (`phase3-test-recall-by-pattern.csv`)

| Pattern | n | `lgbm/ABC` | MLP seeds | GNN seeds |
|---|---|---|---|---|
| FAN-IN | 57 | **0.86** | 0.72–0.75 | 0.74–0.75 |
| FAN-OUT | 65 | 0.79 | 0.83–0.86 | 0.77–0.85 |
| GATHER-SCATTER | 127 | 0.74 | 0.78–0.84 | 0.73–0.82 |
| SCATTER-GATHER | 110 | 0.73 | 0.80–0.86 | 0.78–0.81 |
| CYCLE | 55 | 0.44 | 0.24–0.35 | 0.18–0.53 |
| BIPARTITE | 44 | 0.39 | 0.30–0.48 | 0.30–0.32 |
| STACK | 84 | **0.35** | 0.18–0.32 | 0.23–0.27 |
| RANDOM | 43 | 0.44 | 0.33–0.54 | 0.28–0.51 |
| UNLISTED | 371 | 0.013 | 0.008–0.019 | 0.024–0.032 |

As on val, the neural nets do better on fan-out and the gather/scatter shapes, and `lgbm/ABC` on fan-in and stack. Seed spread on the small patterns (cycle, random) is wide: 0.18–0.53 for the GNN on cycles.

## Tail (Sep 11–18; 1,108 rows, 59% laundering): not a headline

Every model scores 0.95–0.97 PR-AUC, and every row is an alert at 500/day (`phase3-tail-*.csv`). `lgbm/ABC` − `lgbm/AB` is +0.007 [−0.002, 0.017], n.s. `gnn_s0` is significantly below `mlp_s0` (−0.019) and `lgbm/ABC` (−0.018) there, the other seeds aren't. As in Phase 2, the tail can't separate models and stays out of the conclusions.

## Conclusion

- **Do past-only graph features beat per-transaction + behavioural features? Yes.** +0.053 test PR-AUC from the hand-made set C, significant on val and test under both CIs, and better precision and recall at 500 alerts/day.
- **Does a GNN add anything on top of hand-made graph features? No, nothing measurable.** The MLP with the same inputs matches it on val and test.
- **Model choice:** the neural nets rank a bit better overall, but `lgbm/ABC` is as good at the operating point, explainable with SHAP, and stable (2 of 5 GNN runs diverged). Proposed Phase 4 base: `lgbm/ABC` (pending triet's OK for `decisions.md`).
