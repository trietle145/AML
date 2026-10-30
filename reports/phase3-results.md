# Phase 3 results: graph features and a GNN against the frozen baseline

*Rules: `docs/decisions.md`. Plan: `docs/phase3-plan.md`. Model walkthrough: `docs/model.md`. Val numbers regenerate with the `evaluate.py --split val --phase 3 --models … --pairs …` command recorded in the plan's T6 Outcome (CSVs: `reports/phase3-val-*.csv`). GNN run settings and histories: `reports/phase3-gnn-runs.csv`. Test is not scored yet (Checkpoint B first).*

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

- **Seeds:** `gnn_s0` = `gnn_ibm` (unclipped). `gnn_s1`/`s2` use gradient clipping, `clip=1.0`. Clipped seed 0 (`gnn_clip_s0`) diverged at epoch 3 and is excluded. The MLP seeds are unclipped; they never diverged.

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

## Test

*Not scored yet. Checkpoint B (triet's review of the above) decides which models go to test, each scored once.*
