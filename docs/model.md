# The GNN model (`gnn/ABC`), explained and mapped to `src/gnn.py`

*Phase 3, T4–T5. Line numbers refer to [`src/gnn.py`](../src/gnn.py) as of 2026-10-30. The diagrams are Mermaid; GitHub and VS Code's Markdown preview render them.*

## 1. The data as a graph

- **Node** = an account (`bank:account`), 515,088 in total.
- **Edge** = one transaction u → v: directed, with its own time, amount and label. It's a **multigraph**: ten payments from u to v are ten edges. Self-transfers are self-loops.

```mermaid
graph LR
    w1((w1)) -->|$9k ACH| u((u))
    w2((w2)) -->|$12k ACH| u
    w3((w3)) -->|$8k ACH| u
    u ==>|"$28k ACH (to score)"| v((v))
    v -->|$27k ACH| x((x))
```

*Read as: u received from three accounts (fan-in), then sends almost all of it on to v (pass-through). The GNN is meant to recognise shapes like this without being told the rule.*

| What | Where |
|---|---|
| Export all edges, sorted by time, accounts and categories as small integers | `export()`, lines 27–52 → `data/gnn/edges.npz` |
| The head's extra inputs: each transaction's B + C columns (54), `NULL → 0`, then `log1p` | `export()`, lines 54–60 → `data/gnn/head.npy` |
| What the model may read from an edge (set A only) | `EDGE_NUM`, `EDGE_CAT`, lines 22–23 |

`tx_id`, `split` and `label` are exported too, but they are never in `EDGE_NUM` / `EDGE_CAT`, so the model can't see them.

## 2. One hour = one snapshot (the leakage rule)

Transactions are scored hour by hour. For hour `h`, the graph holds only the edges of the **24 hours before** `h`. The transactions *of* hour `h` are the targets, and they are never part of their own graph.

```mermaid
graph LR
    subgraph past["graph edges: ts in [h - 24h, h)"]
        P[(~200k–1.1M earlier transactions)]
    end
    subgraph now["targets: ts in [h, h + 1h)"]
        T[(~8k–45k transactions to score)]
    end
    past -->|"message passing builds account states"| now
    F["hour h+1 and later"]:::future
    classDef future fill:#eee,stroke:#bbb,color:#999
```

| What | Where |
|---|---|
| Find both slices with binary search on the time-sorted edges | `snapshot()`, lines 64–70 |
| **Assert** every graph edge is strictly before every target | line 69, checked on all 425 hours by `bench` (lines 130–136) |
| Keep only the accounts this hour touches, renumbered 0..k−1 | `batch()`, line 76 |
| Node start state = `[1, log1p(in-degree), log1p(out-degree)]` in the snapshot | `batch()`, line 80 |
| Sender and receiver of each target | `target_index`, line 84 |

## 3. The network

```mermaid
graph TD
    subgraph snapshot["snapshot (earlier edges)"]
        E1["edge set A<br/>(amount, hour, flags,<br/>format, currencies)"] --> EE1["edge_enc<br/>→ ea (32)"]
        X["node start state<br/>[1, log in-deg, log out-deg]"] --> NE["node_enc<br/>→ h0 (32)"]
    end
    NE --> L1in["conv_in[0]<br/>incoming edges"]
    NE --> L1out["conv_out[0]<br/>outgoing edges"]
    EE1 --> L1in
    EE1 --> L1out
    L1in --> M1["mix[0]: concat → Linear(64→32) → ReLU<br/>= h1 (1-hop view)"]
    L1out --> M1
    M1 --> L2in["conv_in[1]"]
    M1 --> L2out["conv_out[1]"]
    EE1 --> L2in
    EE1 --> L2out
    L2in --> M2["mix[1] → h2 (2-hop view)"]
    L2out --> M2
    subgraph target["each target transaction u → v"]
        TA["its set A"] --> EE2["edge_enc → 32"]
        TB["its B + C (54, standardized)"]
    end
    M2 -->|"h2[u] (32)"| H
    M2 -->|"h2[v] (32)"| H
    EE2 --> H
    TB --> H
    H["head: concat (150) → Linear(150→32) → ReLU → Linear(32→1)"] --> S["sigmoid → laundering probability"]
```

| Block | Parameters (hidden 32) | Where defined | Where used in `forward` |
|---|---|---|---|
| `edge_enc`: Embedding(8→4) format, Embedding(16→4) currencies, Linear(17→32) | 672 | `EdgeEncoder`, lines 89–97 | lines 114, 116 |
| `node_enc`: Linear(3→32) | 128 | line 106 | line 116 |
| `conv_in` ×2: GINEConv over incoming edges | 6,336 | line 108 | line 118 |
| `conv_out` ×2: GINEConv over **reversed** edges | 6,336 | line 109 | line 118 (`edge_index.flip(0)`) |
| `mix` ×2: Linear(64→32) | 4,160 | line 110 | line 118 |
| `head`: Linear(150→32), ReLU, Linear(32→1) | 4,865 | line 111 | lines 119–123 |
| **Total** | **22,497** | | |

All weights are **shared by every account and transaction**. There is no per-account vector anywhere, so the model learns *how to read* a neighbourhood, not which accounts are dirty.

## 4. The sum: what one GINEConv layer does

For an account `i`, with incoming edges `j → i` carrying edge vectors `e_ji` (from `edge_enc`):

```
message from j   =  ReLU( h_j  +  W · e_ji )                  (W: Linear(32→32) inside GINEConv)
sum              =  Σ  over all incoming edges j → i   of the messages
new state of i   =  MLP( h_i  +  sum )                         (MLP: Linear → ReLU → Linear, line 107)
```

```mermaid
graph LR
    w1["h_w1 + W·e(w1→u)"] -->|ReLU| S((Σ))
    w2["h_w2 + W·e(w2→u)"] -->|ReLU| S
    w3["h_w3 + W·e(w3→u)"] -->|ReLU| S
    hu["h_u (own state)"] --> P((+))
    S --> P
    P --> MLP["MLP"] --> out["conv_in output for u"]
```

- **Why a sum and not an average:** 40 incoming payments give a bigger sum than 2, so fan-in is visible. An average would hide how many there were. That's what makes GIN ("Graph Isomorphism Network") able to count neighbours.
- **Why two directions:** `conv_in` sums over edges *into* u (who paid u); `conv_out` runs the same on reversed edges, so it sums over edges *out of* u (whom u paid). `mix` combines both, so u's state knows "what came in" and "what went out", which is pass-through.
- **Why two layers:** after layer 1, `h1[u]` summarises u's own transactions. Layer 2 sums *neighbours' `h1`*, so `h2[u]` also reflects the neighbours' neighbourhoods (2 hops: "u received from accounts that had just received from many others").

## 5. Training (backward pass)

```mermaid
graph TD
    A["for each epoch"] --> B["for each of the 144 train hours, in time order"]
    B --> C["batch(): snapshot + targets of hour h"]
    C --> D["forward: model(b, head feats) → 19k logits"]
    D --> E["loss = BCE with positives ×100"]
    E --> F["loss.backward(): gradients for all 22,497 weights"]
    F --> G["Adam step (lr 0.001)"]
    G --> B
    B -->|"after the last hour"| V["score the 48 val hours (no gradients)<br/>→ val PR-AUC"]
    V --> K{"better than best?"}
    K -->|yes| SAVE["keep a copy of the weights"] --> A
    K -->|"no, 3 epochs in a row"| STOP["stop; reload best weights;<br/>score all 425 hours → scores table"]
```

| Step | Where |
|---|---|
| B + C head inputs standardized with **train-row** mean and std only | `train()`, line 179 |
| Model built; `mlp` = the same with `use_graph=False` (head gets 86 inputs: target edge + B + C) | line 182 |
| Loss: binary cross-entropy, positives weighted ×100 (`POS_WEIGHT`, line 19) | line 184 |
| One pass over hours: forward (line 193), then for training `zero_grad` / `backward` / `step` (lines 195–198) | `run()`, lines 187–201 |
| Epochs, val PR-AUC, early stopping (patience 3, max 30) | lines 205–215 |
| Reload the best epoch, score every split, check no NaNs and that val reproduces | lines 216–221 |
| Save weights to `data/models/{gnn,mlp}_ABC.pt` and scores to `scores` as `gnn/ABC` / `mlp/ABC` | lines 222–231 |

**What backward does:** the loss for one hour is differentiated back through the head, into `h2[u]` and `h2[v]`, back through layer 2's sums into the neighbours' `h1`, through layer 1, down to the encoders. Each weight gets nudged so that neighbourhoods like those of laundering transactions produce larger logits next time. Because the weights are shared, every hour's 19k transactions update the same 22,497 numbers.

## 6. How the comparison isolates the graph

| Model | Sees | Compared with | Answers |
|---|---|---|---|
| `lgbm/ABC` | A + B + C, trees | `lgbm/AB` | Do hand-made graph features help? |
| `mlp/ABC` | A + B + C, the same head with no graph | `lgbm/ABC` | Neural net vs trees on the same inputs |
| `gnn/ABC` | A + B + C **plus** `h2[u]`, `h2[v]` | `mlp/ABC`, `lgbm/ABC` | Does *learned* 2-hop graph context add anything? |

`gnn/ABC − mlp/ABC` is the effect of the message passing alone; everything else is identical.
