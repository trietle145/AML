# Phase 4 plan: catch the near-miss patterns, then explain the alerts (DRAFT, revised)

*First draft 2026-10-30; revised 2026-10-31 after Phase 3 closed on test and the literature review in `docs/phase4-research.md`. Task 1 done on branch `unlisted-diagnosis`; nothing else started. Needs triet's review before any task starts.*

## Goal

Lift recall at 500 alerts/day on the listed patterns the models nearly catch (STACK, BIPARTITE, CYCLE, RANDOM), with past-only multi-hop features that the published work shows help. State the UNLISTED ceiling with evidence instead of chasing it. Then make the chosen model usable by an investigator: account-level alerts and plain-English reasons. Phase 4 is done when at most 2 finalists are scored once on test against `lgbm/ABC`, with alert reasons. The IBM Multi-GNN reference run comes last.

## Where Phase 3 leaves us (test, scored once 2026-10-30)

| Model | Test PR-AUC | P / R @500/day | Note |
|---|---|---|---|
| `lgbm/AB` (frozen) | 0.292 | 0.337 / 0.353 | Phase 2 baseline |
| `lgbm/ABC` | 0.345 | **0.389 / 0.407** | set C +0.053, significant (iid and cluster) |
| MLP, 3 seeds | 0.369 ± 0.002 | 0.369–0.382 / 0.386–0.400 | same head as the GNN, no graph |
| GNN, 3 seeds | 0.375 ± 0.003 | 0.377–0.386 / 0.394–0.404 | GNN − MLP n.s. on val and test |

- **Neural nets lead on PR-AUC, not at the operating point.** Their lead lives deep in the ranking: on val at 5,000/day, recall is 0.78–0.80 for them vs 0.72 for `lgbm/ABC`; at 500/day they tie.
- **Three tiers at 500/day (val):** FAN-IN/OUT and GATHER/SCATTER-GATHER 77–95%; STACK, BIPARTITE (and CYCLE, RANDOM for LightGBM) 35–53%, but 69–82% at 2,000/day, so they sit **just below the cut**; UNLISTED 1–3%, reaching only 20% at 2,000/day.
- **The literature agrees** (`docs/phase4-research.md`): published GNNs are weak on CYCLE (40–62%) and BIPARTITE (35–58%) too, and nobody detects UNLISTED (0.5–4.3%). The closest comparable published setup (IBM's Graph Feature Preprocessor + gradient boosting, past-only) beats the GNNs, consistent with our GNN ≈ MLP. Its multi-hop features (scatter-gather, cycles up to length 10) added +3.8 F1 and per-account amount/time statistics +8.2.
- **Published numbers aren't comparable to ours:** their test likely includes our tail (655 easy positives, no UNLISTED), F1 is at threshold 0.5, and the GNN graphs see future edges.

## Decisions

- **Base model `lgbm/ABC`** (approved by triet 2026-10-31, `docs/decisions.md` § Phase 4 base model): as good at 500/day, explainable, stable (2 of 5 GNN runs diverged).
- **No more GNN work.** Message passing added nothing measurable; the MLP is kept as an ensemble partner only.
- **Same rules as Phases 2–3:** strictly-past features, windows ≤ 3d, no IDs or pattern data as features (pattern data may be used for *design analysis* on train, never as inputs), val for all tuning, test once per finalist, at most 2 finalists. Outputs use `reports/phase4-*`.
- **New features are set D** (they are about counterparties and flows, so C-like, never B). Model names `lgbm/ABCD` etc.; `lgbm/AB` and `lgbm/ABC` are never retrained.
- **No new dependencies without asking.** SHAP comes from LightGBM's built-in `pred_contrib=True`. Cycle search may need compiled loops: numba is approved if Task 3's timing slice shows plain Python is too slow.

## Dependency graph

```
T1 UNLISTED diagnosis (done)
T2 pattern anatomy (train) ─ T3 cycle timing slice (done; T4 cycles, T5 scatter-gather/bicliques dropped)
T6 set D: account amount/time statistics (independent)
        T6 ─ T7 lgbm/ABCD on val ─ Checkpoint A
Checkpoint A ─ T8 ensemble ─┐
              T9 account alerts ─┼─ T10 explanations ─ Checkpoint B ─ T11 test once ─ T12 IBM reference
```

## Tasks

### Task 1: UNLISTED diagnosis (S) — DONE 2026-10-30
**Description:** move the exploration into `src/` (`evaluate.py --analyze-unlisted`, train + val only) and write `reports/unlisted-exploration.md` with the profile, scheme links, repeat accounts and upper bound. Then answer the Task 2 go/no-go question: how many val UNLISTED rows have a sender or receiver whose earlier transactions (strictly earlier, ≤ 3d) the current models (`lgbm/ABC`, `gnn/ABC`) already scored high (≥ the model's val best-F1 threshold), compared with licit val rows (lift). Earlier rows in train carry in-sample scores (the stacking trap), so hits from val-period history are reported separately.
**Acceptance:**
- [x] `evaluate.py --analyze-unlisted` regenerates the profile, scheme-link, repeat-account and upper-bound tables and writes `reports/phase4-unlisted-*.csv`; it reads only train + val rows and asserts that.
- [x] Past-score table: share of val UNLISTED vs licit rows with an earlier high-scored transaction on either account, per model, val-only history vs any history, and the lift.
- [x] The report states the recall ceiling with its evidence and a go/no-go for Task 2.
**Verification:** the command runs end to end with its asserts; the upper-bound numbers match `logs/explore_unlisted.log` (0.0147 / 0.0016).
**Outcome:** `evaluate.py --analyze-unlisted` (~3 min, asserts train + val only) writes `reports/phase4-unlisted-*.csv`; report in `reports/unlisted-exploration.md`. Upper bound reproduces 0.0147 / 0.0016. Ceiling: listed share of val laundering is 63.4%; UNLISTED caught 2/379 (`lgbm/ABC`), 4/379 (`gnn/ABC`). **Past scores:** with val-only history, an UNLISTED row's account had an earlier alerted transaction (val best-F1 threshold, ≤ 3d) in 1/379 (`lgbm/ABC`) and 4/379 (`gnn/ABC`) rows, against 0.3% of licit rows. The 34.6% repeat-account signal exists in the labels, not in the scores. **Verdict: past-score account risk is no-go as an UNLISTED lever** (dropped from this plan; see "Dropped").
**Files:** `src/evaluate.py`, `reports/unlisted-exploration.md`

### Task 2: Pattern anatomy on train (XS) — DONE 2026-10-31
**Description:** measure, from `pattern_tx` on train attempts only, what each pattern type looks like: transactions and accounts per attempt, time span (first to last transaction), cycle length, number of layers and width for STACK and BIPARTITE, and hops for RANDOM. These numbers set the windows and length caps for Tasks 3–5 (GFP uses 6 h / 1 d and cycles ≤ 10; ours must be ≤ 3d). Design analysis only: nothing here becomes a feature.
**Acceptance:**
- [x] Table per pattern type: attempts, median and p90 of transactions, accounts, span in hours, and cycle length / layer count where defined
- [x] Written choice of window and max cycle length for Tasks 3–5, with the share of train CYCLE attempts it covers
**Verification:** counts per type sum to the train attempt total; script asserts it.
**Dependencies:** none
**Outcome:** `evaluate.py --analyze-patterns` (~10 s, log `logs/phase4_t2.log`) writes `reports/phase4-pattern-anatomy.csv`. Only **124 of 370 attempts lie entirely in train**; longer attempts straddle the cut and are skipped, so the spans below are biased short, and samples are small (11–25 attempts per type).
- **Attempts are slow.** Median span 65–85 h for CYCLE, FAN-IN, FAN-OUT, SCATTER-GATHER and STACK (BIPARTITE 29 h, RANDOM and GATHER-SCATTER ~11 h). Only 33–62% of those finish within 72 h, and 0–27% within 24 h. **Window choice: 3d everywhere** (the rule's cap), not GFP's 6 h / 1 d, which would miss most of each structure.
- **Cycle lengths (21 train cycles):** 2 → 10, 3 → 3, 4–7 → 5, 10 → 3. Lengths 2 and 3 are already in set C (`pair_rev_cnt_3d`, `cyc3_cnt_3d`), yet CYCLE recall at 500/day is only 0.44. **Max cycle length L = 10**; lengths 4–10 are new and cover the other 8 of 21 (38%).
- **Layers:** BIPARTITE is 1 hop (sources → sinks, up to 30 accounts); STACK is at most 2 hops (3 layers, up to 45 accounts); SCATTER-GATHER and GATHER-SCATTER 2 hops; RANDOM up to 9 hops (median 2). So bicliques on each side of the middle account (Task 5) cover STACK, and nothing longer than 2 hops is needed for it.
- **Pattern edges are in distinct minutes** (minutes ≈ transactions), so the strictly-earlier rule doesn't hide an attempt's earlier edges from its later ones.
- **Caveat for Tasks 4–5:** a past-only structural feature can only fire on the edge that **completes** a structure (the closing edge of a cycle, the gather edges after a scatter, the last edge of a biclique). The first edges of an attempt look like nothing yet. That caps how much per-pattern recall set D can add.
**Files:** `src/evaluate.py` (`--analyze-patterns`), `reports/phase4-pattern-anatomy.csv`

### Task 3: Cycle-search timing slice (S, highest risk) — DONE 2026-10-31, long cycles dropped
**Description:** time a search for time-ordered cycles closing on each edge u→v (simple paths v → … → u over strictly earlier edges in the 3d window, lengths up to 10), check it, and decide the implementation path. Throwaway measurement, not in the repo.
**Outcome:** pure Python with a backward-reachability prune runs at **0.1 ms/edge** (p99 0.5 ms, max 0.01 s) on 20,344 val edges: **~9 min for the full table, no numba needed**. Checks: length-3 counts equal set C's `cyc3_cnt_3d` and 2-cycles equal `pair_rev_cnt_3d > 0` on every sampled edge; a naive unpruned search agrees on all of them. **But the signal is tiny:** of the 66 val CYCLE-pattern rows, 8 close a 2- or 3-cycle (already in set C) and only **2** close a cycle of length 4–10. A past-only feature fires only on the closing edge, and many cycles take longer than 3d (Task 2). **Decision: drop long cycles** (old Task 4); the ceiling is ~2 of 1,036 val laundering rows.

### Task 4: dropped (long cycles, see Task 3)

### Task 5: Scatter-gather and bicliques — DROPPED 2026-10-31 after a ceiling check
**Outcome:** throwaway check on all val laundering rows and 20k licit rows (0.33 ms/edge, ~30 min for the full table). Over strictly earlier edges in 3d, the edge completes a **K2,2 biclique** or a **two-hop scatter-gather** for: BIPARTITE 0 of 49, STACK 4 / 3 of 150, CYCLE 0 of 66, RANDOM 1 of 38. Only SCATTER-GATHER fires (50 of 134, 37%), and `lgbm/ABC` already catches it at ~89% at 500/day. UNLISTED 6.6% / 0.3% vs licit 1.6% / 0.3%. **The near-miss patterns don't complete these structures within a past-only 3d view, so the features can't lift them.**

### Near-miss contributions (throwaway check, 2026-10-31)
Per-row TreeSHAP contributions of `lgbm/ABC` (`pred_contrib`) on val laundering rows + 20k licit, caught = inside the day's top 500. Missed STACK / BIPARTITE / CYCLE / RANDOM rows sit a median **1.1–1.9 log-odds below the cut** (UNLISTED 2.9). Within each pattern, caught rows have **larger amounts, and amounts far above the receiver's usual** (`in_amt_vs_mean_24h` median 9.5× caught vs 3.6× missed for STACK + BIPARTITE); fan-in/out push 45–60% of missed rows down, because each transfer in these schemes looks one-to-one. UNLISTED: nothing separates caught from missed. **Caveat:** correlated features split credit into large opposite values (e.g. `pair_min_since_last` −2 to −16 vs `pair_cnt_3d` +1 to +2, both "first payment on this pair"; max |contribution| 139), so contributions must be summed per feature group before use (Task 10). **Steers Task 6 towards amount-vs-own-usual features.**

### Task 6: Set D, account amount and timing statistics (S, leakage risk) — amount part DONE 2026-11-01
**Description:** priority from the near-miss check above: **how unusual this amount is for this sender and for this receiver** (amount vs the account's own median and spread over 24h / 3d, per direction; z-score or ratio), extending the existing `*_amt_vs_mean_24h`. Then GFP's "vertex statistics" group (+8.2 F1 in their ablation), restricted to what set B lacks: for sender and receiver, in and out, over 24h: variance, skew and kurtosis of amounts, and of **gaps between** incident transactions (never raw timestamps, which grow with absolute time). Plus the old draft's forwarding speed (minutes from an account's last receipt to this send) and structuring (sends just under $10k / $20k in 3d). Check overlap with set B first and drop duplicates.
**Acceptance:**
- [x] Brute-force check 0 mismatches; leaky variant fails; drift: distribution on test within 1% of val on accounts seen in both
- [x] List of B columns each new one overlaps with (correlation > 0.95 → dropped)
**Verification:** `features.py --check` passes.
**Dependencies:** none
**Files:** `src/features.py`
**Outcome:** built the amount part only: `SET_D` = 12 columns, per side (sender's past sends / receiver's past receipts) and window (24h, 3d): `amt_z` (z-score vs own mean and sd), `amt_vs_med`, `amt_vs_max`, strictly earlier rows (`d_query`, `features.py d` builds D alone, ~1 min). `features.py --check d`: 0 mismatched cells on 277 stress rows; leaky variant (frame ends at the current row) caught with 2,205 bad cells in all 12 columns. Drift: no test account above the train max. Overlap with B's `amt_vs_mean`: rank correlation 0.63–0.94 (highest `amt_vs_max`), none > 0.95, all kept. Medians laundering vs licit: `in_amt_vs_med_3d` 1.99 vs 1.00, `in_amt_vs_max_3d` 0.72 vs 0.32, `out_amt_vs_med_3d` 1.40 vs 1.00. **Skipped for now:** timing-gap moments, forwarding speed and structuring (no support in the near-miss check; add only if Task 7 shows the amount part helps).

### Task 7: `lgbm/ABCD` on val (S) — DONE 2026-11-01, no gain
**Description:** add `ABCD` (and, if Task 6 is large, `ABC+cycles` / `ABC+sg` ablations) to `FEATURE_SETS`; train with the Phase 2 grid; evaluate on val, paired against `lgbm/ABC`.
**Acceptance:**
- [x] Paired val PR-AUC difference vs `lgbm/ABC`, iid and cluster CIs
- [x] Recall by pattern **at 500/day** and at the val threshold, with CIs, for STACK, BIPARTITE, CYCLE, RANDOM, UNLISTED
- [x] `lgbm/AB` and `lgbm/ABC` rows in `scores` unchanged (max diff 0)
**Verification:** `train.py --phase 4 lgbm ABCD` then `evaluate.py --split val --phase 4 --models lgbm/ABC,lgbm/ABCD --pairs lgbm/ABCD:lgbm/ABC`.
**Dependencies:** Task 6
**Files:** `src/train.py`, `src/evaluate.py` (`--phase 4`), `reports/phase4-val-*.csv`
**Outcome:** grid (~7 min, `logs/train_lgbm_ABCD.log`) picked 63 leaves, lr 0.05, min_child 500, 179 trees. `lgbm/AB` scores equal the frozen copy (max diff 0); `lgbm/ABC` val PR-AUC reproduces 0.4439. Val (`logs/phase4_val_abcd.log`, `reports/phase4-val-*.csv`): `lgbm/ABCD` PR-AUC 0.4392 vs 0.4439, **paired −0.005 [−0.022, 0.012], cluster [−0.021, 0.013]: not significant**. P/R@500/day 0.455 / 0.439 vs 0.456 / 0.440 (same). F1 at the val threshold 0.493 vs 0.479. Recall by pattern at 500/day (new `phase4-val-recall-by-pattern-at-k.csv`): CYCLE 0.44 → 0.56 (+8 rows), STACK 0.53 → 0.48 (−7), RANDOM 0.45 → 0.37 (−3), BIPARTITE 0.49 → 0.47 (−1), UNLISTED 5 → 8 of 379; the rest within ±2 rows. **Set D reshuffles which near-miss rows reach the top 500 instead of adding any**: the amount-vs-usual signal was already in set B's `amt_vs_mean`. Code: `evaluate.py --phase 4` (default pair `lgbm/ABCD:lgbm/ABC`), recall by pattern at 500/day for every phase, earlier-phase test guard covers Phases 2–3; `metrics.top_k_per_day` (identical results to before).

### Checkpoint A (review with triet)
- [x] Did set D move the near-miss patterns at 500/day? If not, the write-up says the multi-hop gains in the literature don't carry over to a past-only ≤ 3d setting, and Phase 4 moves straight to Tasks 9–10.
- [x] UNLISTED reported as the ceiling with Task 1's and the literature's evidence

**Checkpoint A outcome (triet, 2026-11-01): no.** Set D didn't improve `lgbm/ABC` (Task 7); long cycles and bicliques/scatter-gather were dropped before building because they reach almost no near-miss rows (Tasks 3, 5). **Base stays `lgbm/ABC`; no more feature work.** Set D lives only on branch `feature-set-d` (archived, not merged); Phase 4 continues from `main` on branch `ensemble` with Task 8.

### Task 8: Ensemble (S) — DONE 2026-11-01, fails the keep rule
**Description:** rank-average the best LightGBM (`lgbm/ABCD` or `lgbm/ABC`) with the MLP. They're strong on different patterns (LightGBM on FAN-IN and STACK, the MLP on FAN-OUT and gather/scatter), so the blend may help at 500/day. Uses the existing `mlp_s0–s2` scores (mean over seeds, no seed picked on val); retraining the MLP on set D needs Colab and is out of scope unless the blend is promising.
**Acceptance:** ensemble kept only if its paired val gain over the best single model excludes 0 under both CIs, **and** P/R at 500/day doesn't drop.
**Dependencies:** Checkpoint A
**Files:** `src/evaluate.py` or `src/train.py` (writes `scores` as `ens/<set>`)
**Outcome (2026-11-01):** `train.py ens` writes `ens/ABC`: equal-weight average of each model's percentile among its own train scores (`lgbm/ABC` and the MLP mean over seeds 0–2; no weight tuned on val). Val (`logs/phase4_val_ens.log`): PR-AUC **0.483** vs `lgbm/ABC` 0.444 and MLP seeds 0.477 / 0.472 / 0.464; P/R@500/day **0.476 / 0.460** (best of all; `lgbm/ABC` 0.456 / 0.440, `mlp_s0` 0.474 / 0.458). Paired: vs `lgbm/ABC` **+0.039** [0.028, 0.050], cluster [0.028, 0.052], significant; vs the MLP seeds +0.007 / +0.011 / +0.019, **no CI excludes 0 under both**. **Fails the keep rule** (must beat the best single model): it is the MLP plus a little LightGBM. Shared weak spots unchanged at 500/day (STACK 0.51 vs 0.53, BIPARTITE 0.45 vs 0.49); gains where the models already differed (FAN-IN 0.93, CYCLE 0.53, GATHER-SCATTER 0.92). Held as a possible second test finalist at Checkpoint B (triet's call).

**Cycle-window check (throwaway, 2026-11-01):** val CYCLE rows closing a time-ordered cycle (≤ 10): 8 of 66 at 3d, **11 of 66 at 5d, 7d and unlimited** (licit 0.07% → 0.17%). Of the 24 CYCLE attempts touching val, **13 finish after val** (closing payment in test/tail), so val holds only their opening payments; of the 11 that close inside val, the features already catch 10. Train cycles are time-ordered (19/21) and the last payment closes the loop (20/21). **Per-pattern recall on val and test is structurally capped for slow patterns:** attempts take ~3 days, val and test are 2 days each, and a past-only feature can only fire on the completing payment. Not fixable without changing the locked split; goes in the write-up.

### Task 9: Account-level alerts (S) — DONE 2026-11-01
**Description:** roll transaction scores up to one alert per account per day (max score). Account label as in `decisions.md` (≥ 1 laundering transaction that day). Precision and recall at K accounts per day.
**Acceptance:** account-level table on val next to transaction-level; K chosen and written into `decisions.md` (triet approves).
**Dependencies:** Checkpoint A (uses the chosen model)
**Files:** `src/evaluate.py`, `docs/decisions.md`
**Outcome:** `evaluate.py --accounts --phase 4 --models …` (val only, ~1 min) writes `reports/phase4-val-accounts.csv`. Val: 450,244 account-days, 1,629 positive (~815 a day of ~225k active). Account-day PR-AUC: `lgbm/ABC` 0.432 [0.407, 0.457], `mlp_s0` 0.440, `ens/ABC` 0.459. **K = 800 accounts/day** (triet, in `decisions.md`): P / R `lgbm/ABC` 0.42 / 0.42, `mlp_s0` 0.43 / 0.42, `ens/ABC` 0.43 / 0.42, close to the transaction-level 0.456 / 0.440. The ensemble's lead shows only on short queues (500/day: P 0.62 vs 0.57).

### Task 10: Explanations (M) — DONE 2026-11-01
**Description:** per-alert feature contributions from LightGBM's `pred_contrib=True` (exact TreeSHAP, no new package); plain-English reasons from each alert's top 3 contributions (e.g. "received from 14 accounts in 3 days; forwarded 95% within 1 h; first payment to this receiver"); 10 val alerts checked by hand against the raw transactions.
**Acceptance:** reasons for every val alert at 500/day, built from contributions **summed per feature group** (correlated features such as `pair_min_since_last` and `pair_cnt_3d` split credit into large opposite values); contributions sum to the raw score (assert); the 10 hand-checked examples read correctly.
**Dependencies:** Checkpoint A
**Files:** `src/explain.py`, `reports/phase4-alert-examples.md`
**Outcome:** `src/explain.py` (~1 min, val only) explains all 1,000 val alerts of `lgbm/ABC` at 500/day → `reports/phase4-val-alert-reasons.csv`. TreeSHAP contributions reproduce the stored scores (assert); 62 features in **8 groups** (assert): features that encode the same fact share a group (account activity + pair history + receiver fan-in; amount + both sides' usual amounts), which cuts the median sum|group| / |net| from 2.23 per feature (1.55 with 13 groups) to 1.22. Top 3 positive groups → sentences from the alert's own values. **10 alerts (6 patterns incl. UNLISTED + 4 false alarms) checked against raw `transactions` rows: all facts match** (`reports/phase4-alert-examples.md`). Findings: ACH leads 278 of 456 laundering alerts and 251 of 544 false alarms; new-account facts ("account history") lead 116 / 156 and read identically for a STACK hop and a licit payment, which is where the false alarms come from. Limit: a few rows still have ±100 log-odds offsetting groups, so their reason order is less reliable.

### Checkpoint B (hard stop)
- [ ] triet reviews val results and picks at most 2 finalists for test

### Task 11: Test once and results (S)
**Description:** score the finalists once on test (`--phase 4 --final`), paired against `lgbm/ABC` (its Phase 3 test scores reused, never rescored) and the frozen `lgbm/AB`; write `reports/phase4-results.md`. The comparison with published numbers comes from Task 12, not from re-reading our test scores.
**Files:** `src/evaluate.py`, `reports/phase4-results.md`, `docs/decisions.md`

### Task 12 (end): IBM Multi-GNN reference run (M)
**Description:** run IBM's code unchanged on HI-Small in a separate Colab notebook (`--model gin --emlps --reverse_mp --ego --ports`), map their test predictions to our `tx_id`s, evaluate with `metrics.py` including recall by pattern, on their test range and on ours. Label it "IBM reference (transductive: test graph includes later edges; test includes tail)". Check the repo's licence before reusing any code.
**Acceptance:** their score next to ours on both ranges, with the protocol differences listed; their UNLISTED recall reported.
**Write-up context (triet, 2026-11-01):** after this run, put three of our rules into numbers next to the published results: the **2-day val/test windows** (slow attempts cut; 13 of 24 val CYCLE attempts finish in test), **strictly-past features** (this run's transductive-vs-ours gap), and **no account or bank IDs** (memorising repeat accounts; 34.6% of UNLISTED rows touch an account that laundered earlier).
**Files:** a separate notebook; `reports/phase4-results.md`

## Dropped (with reasons)

- **Published-protocol side table** (F1@0.5 on test ∪ tail): triet 2026-10-31. Task 12 compares with IBM properly, without re-reading our test scores.
- **Delayed-label features** ("account had confirmed laundering ≥ D days ago"): triet 2026-10-31. Uses labels as inputs, a realistic investigation delay is longer than our 10 days of data, and it would fire on ~10% of licit rows.

- **Phase 5 dashboard** (Streamlit alert list + pyvis ego-graph): out of scope for now, triet 2026-10-31. Phase 4 ends with its report; revisit only after Phase 4 closes.

- **Account-risk features from past model scores:** Task 1 showed no UNLISTED signal in past scores (1–4 of 379 rows).
- **More GNN work** (`gnn_w72` rerun, port numbering, deeper models): GNN ≈ MLP on val and test, and the published past-only reference (GFP + boosting) beats GNNs too. Port numbers (+28 F1 for GIN in the papers) were measured with future edges visible.
- **2-hop counts in SQL** (old set D): superseded by scatter-gather and bicliques (Task 5).

## Optional, if time allows

- 2 extra seeds of `lgbm/ABCD` to measure LightGBM's seed spread.
- Forward pass-through chain length for RANDOM (idea only, no published evidence).
- LI-Small robustness run (UNLISTED is 71% of its laundering).

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| A structure feature can only fire on the completing edge | High | throwaway ceiling check before building each one (Task 3 found ~2 val rows for long cycles, so they were dropped) |
| Set D leaks (same-minute edges, later edges) | High | brute-force `--check` with a leaky variant that must fail, as for B and C |
| Multi-hop gains in the papers don't transfer (their windows, batch visibility, tail in test) | Med | Checkpoint A states it either way; per-pattern recall at 500/day is the target, not overall F1 |
| Memory (bicliques and scatter-gather intersections) | Med | DuckDB caps as in Phase 3; one temp table per step; Colab fallback |
| Test credibility worn down | Med | ≤ 2 finalists, scored once, all logged |

## Open questions

None (resolved with triet 2026-10-31: base `lgbm/ABC`; side table and delayed labels dropped; numba allowed if Task 3 needs it).
