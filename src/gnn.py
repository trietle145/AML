import json
import os
import sys
import time

import duckdb
import numpy as np
import psutil
import torch
from torch import nn
from torch_geometric.nn import GINEConv

from features import SET_B, SET_C
from load import DB, EXPECTED_ROWS, ROOT
from metrics import pr_auc

EDGES = ROOT / "data/gnn/edges.npz"
HEAD = ROOT / "data/gnn/head.npy"
MODELS = ROOT / "data/models"
RUNS = ROOT / "data/gnn/runs"
# positives are ~1 in 1,000; weighting them 100x keeps their gradient visible without the noise of the full ratio
POS_WEIGHT = 100.0
SPLITS = ["train", "val", "test", "tail"]
# model inputs per edge (set A, encoded); tx_id, split and label are kept out of this list on purpose
EDGE_NUM = ["log_amount_usd", "hour", "is_cross_bank", "is_currency_mismatch", "is_self_transfer"]
EDGE_CAT = ["fmt", "pay_cur", "recv_cur"]


# all transactions sorted by time, accounts and categories as small ints; minute = minutes since Sep 1 00:00
def export():
    con = duckdb.connect(str(DB), read_only=True, config={"memory_limit": "2GB", "threads": 4})
    e = con.sql("""
        WITH accts AS (SELECT a, row_number() OVER (ORDER BY a) - 1 i
                       FROM (SELECT from_acct a FROM transactions UNION SELECT to_acct FROM transactions)),
             fmt AS (SELECT v, row_number() OVER (ORDER BY v) - 1 c FROM (SELECT DISTINCT payment_format v FROM transactions)),
             cur AS (SELECT v, row_number() OVER (ORDER BY v) - 1 c
                     FROM (SELECT payment_currency v FROM transactions UNION SELECT receiving_currency FROM transactions))
        SELECT t.tx_id, date_diff('minute', TIMESTAMP '2022-09-01', t.ts)::INT "minute", s.i::INT src, d.i::INT dst,
               f.log_amount_usd::FLOAT log_amount_usd, f.hour::FLOAT "hour", f.is_cross_bank::FLOAT is_cross_bank,
               f.is_currency_mismatch::FLOAT is_currency_mismatch, f.is_self_transfer::FLOAT is_self_transfer,
               fm.c::TINYINT fmt, pc.c::TINYINT pay_cur, rc.c::TINYINT recv_cur,
               list_position(['train', 'val', 'test', 'tail'], t.split)::TINYINT - 1 split, t.is_laundering::TINYINT label
        FROM transactions t JOIN features f USING (tx_id)
        JOIN accts s ON s.a = t.from_acct JOIN accts d ON d.a = t.to_acct
        JOIN fmt fm ON fm.v = t.payment_format JOIN cur pc ON pc.v = t.payment_currency JOIN cur rc ON rc.v = t.receiving_currency
        ORDER BY t.tx_id
    """).fetchnumpy()
    e = {k: np.asarray(v) for k, v in e.items()}
    n = len(e["tx_id"])
    assert n == EXPECTED_ROWS and (e["tx_id"] == np.arange(n)).all(), "tx_id must be 0..n-1"
    assert (np.diff(e["minute"]) >= 0).all(), "tx_id order must be time order"
    assert e["label"].sum() == 5177 and set(np.unique(e["split"])) == {0, 1, 2, 3}
    EDGES.parent.mkdir(exist_ok=True)
    np.savez(EDGES, **e)
    print(f"edges: {n:,} rows, {e['src'].max() + 1:,} accounts, {EDGES.stat().st_size / 2**20:.0f} MB -> {EDGES}")

    # the head's per-transaction B + C columns, tx_id order; all non-negative and heavy-tailed, so NULL -> 0 then log1p (as in lr)
    cols = SET_B + SET_C
    h = con.sql(f"SELECT {', '.join(f'COALESCE({c}, 0)::FLOAT' for c in cols)} FROM features ORDER BY tx_id").fetchnumpy()
    h = np.stack([np.asarray(v) for v in h.values()], 1)
    assert h.shape == (n, len(cols)) and h.min() >= 0, (h.shape, h.min())
    np.save(HEAD, np.log1p(h))
    print(f"head features: {h.shape[1]} columns (B + C) -> {HEAD}")


# edges usable for hour h: graph = [h - window, h), targets = [h, h + 1); leak=True wrongly includes hour h itself
def snapshot(minute, h, window_h=24, leak=False):
    lo = np.searchsorted(minute, (h - window_h) * 60)
    hi = np.searchsorted(minute, (h + leak) * 60)
    t_lo, t_hi = np.searchsorted(minute, [h * 60, (h + 1) * 60])
    # every graph edge strictly before every target: the structural leakage rule
    assert hi == lo or t_lo == t_hi or minute[hi - 1] < h * 60 <= minute[t_lo], f"hour {h}: graph sees target-hour edges"
    return slice(lo, hi), slice(t_lo, t_hi)


# one hour as a compact graph: only accounts touched by the snapshot or the targets, renumbered 0..k-1
def batch(e, h, window_h=24, device="cpu"):
    g, t = snapshot(e["minute"], h, window_h)
    nodes, inv = np.unique(np.concatenate([e["src"][g], e["dst"][g], e["src"][t], e["dst"][t]]), return_inverse=True)
    ng, nt = g.stop - g.start, t.stop - t.start
    src, dst = inv[:ng], inv[ng:2 * ng]
    k = len(nodes)
    deg = np.stack([np.ones(k), np.log1p(np.bincount(dst, minlength=k)), np.log1p(np.bincount(src, minlength=k))], 1)
    tt = lambda a: torch.as_tensor(a, device=device)
    edge = lambda s: (tt(np.stack([e[c][s] for c in EDGE_NUM], 1)), tt(np.stack([e[c][s].astype(np.int64) for c in EDGE_CAT], 1)))
    return {"rows": t, "x": tt(deg.astype(np.float32)), "edge_index": tt(np.stack([src, dst]).astype(np.int64)), "edge": edge(g),
            "target_index": tt(np.stack([inv[2 * ng:2 * ng + nt], inv[2 * ng + nt:]]).astype(np.int64)), "target": edge(t),
            "y": tt(e["label"][t].astype(np.float32))}


# set A of an edge -> vector; categories get tiny embeddings (formats and currencies, never accounts)
class EdgeEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fmt, self.cur = nn.Embedding(8, 4), nn.Embedding(16, 4)
        self.lin = nn.Linear(len(EDGE_NUM) + 12, hidden)

    def forward(self, num, cat):
        num = torch.cat([(num[:, :1] - 3) / 1.5, num[:, 1:2] / 23, num[:, 2:]], 1)
        return self.lin(torch.cat([num, self.fmt(cat[:, 0]), self.cur(cat[:, 1]), self.cur(cat[:, 2])], 1))


# GINe with reverse message passing: separate convs over incoming and outgoing edges; head on [h_u, h_v, target edge, head_feats]
class Net(nn.Module):
    def __init__(self, hidden=64, layers=2, head_dim=0, use_graph=True, dropout=0.0, head_dropout=0.0):
        super().__init__()
        self.use_graph = use_graph
        self.drop, self.drop_head = nn.Dropout(dropout), nn.Dropout(head_dropout)
        self.edge_enc = EdgeEncoder(hidden)
        self.node_enc = nn.Linear(3, hidden)
        mlp = lambda: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv_in = nn.ModuleList(GINEConv(mlp(), edge_dim=hidden) for _ in range(layers))
        self.conv_out = nn.ModuleList(GINEConv(mlp(), edge_dim=hidden) for _ in range(layers))
        self.mix = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear((3 if use_graph else 1) * hidden + head_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, b, head_feats=None):
        parts = [self.edge_enc(*b["target"])]
        if self.use_graph:
            h, ea = self.node_enc(b["x"]), self.edge_enc(*b["edge"])
            for ci, co, mx in zip(self.conv_in, self.conv_out, self.mix):
                h = self.drop(torch.relu(mx(torch.cat([ci(h, b["edge_index"], ea), co(h, b["edge_index"].flip(0), ea)], 1))))
            u, v = b["target_index"]
            parts = [h[u], h[v]] + parts
        if head_feats is not None:
            parts.append(head_feats)
        # dropout before the last layer, applied here so saved weights keep the head's layer names
        z = self.drop_head(self.head[1](self.head[0](torch.cat(parts, 1))))
        return self.head[2](z).squeeze(1)


# leakage asserts on every hour, a leaky snapshot that must fail, then forward + backward timing on CPU and GPU
def bench(window_h=24, hidden=64, head_dim=54):
    e = dict(np.load(EDGES))
    hours = np.arange(e["minute"].max() // 60 + 1)
    for h in hours:
        snapshot(e["minute"], h, window_h)
    try:
        snapshot(e["minute"], 100, window_h, leak=True)
        raise SystemExit("leaky snapshot was not caught")
    except AssertionError:
        print(f"leakage assert holds on all {len(hours)} hourly snapshots; leaky snapshot caught")
    train_hours = np.unique(e["minute"][e["split"] == 0] // 60)
    # the 5 slices: midnight batch hour, warm-up day, normal hours, last train hour
    pick = [24, 30, 75, 110, int(train_hours.max())]
    sizes = [(lambda g, t: (g.stop - g.start, t.stop - t.start))(*snapshot(e["minute"], h, window_h)) for h in pick]
    print("hours", pick, "-> (graph edges, targets):", sizes)
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        torch.manual_seed(0)
        model = Net(hidden, head_dim=head_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100.0, device=device))
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        secs = []
        for h in pick:
            t0 = time.time()
            b = batch(e, h, window_h, device)
            # placeholder for the target's B + C columns (54 floats), loaded for real in T5
            out = model(b, torch.zeros(len(b["y"]), head_dim, device=device))
            loss = loss_fn(out, b["y"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            secs.append(time.time() - t0)
        per = float(np.mean(secs))
        mem = f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if device == "cuda" else \
              f"peak process RAM {psutil.Process().memory_info().peak_wset / 2**30:.2f} GB"
        print(f"{device}: {per:.2f} s per snapshot ({', '.join(f'{s:.2f}' for s in secs)}), "
              f"epoch over {len(train_hours)} train hours ~ {per * len(train_hours) / 60:.1f} min, {mem}")


# defaults follow IBM Multi-GNN's GIN settings (model_settings.json): lr ~0.006, class weight ~6, dropout 0.01 / 0.1,
# batches of 8,192 transactions, up to 100 epochs. The baseline gnn/ABC used lr 0.001, pos_weight 100, whole hours, hidden 32
# clip: gradient-norm cap per update (0 = off). Without it, runs at these settings blew up mid-training
# (gnn_w72 epoch 10, gnn_best_s1 epoch 4: train loss jumps, then a dead model at val ~0.001)
DEFAULTS = {"name": "gnn", "seed": 0, "hidden": 64, "window_h": 24, "lr": 0.006, "pos_weight": 6.0, "dropout": 0.01,
            "head_dropout": 0.1, "batch_size": 8192, "max_epochs": 100, "patience": 10, "max_hours": 0, "clip": 1.0}


# train hours (shuffled each epoch), early stopping on val PR-AUC, then score every transaction into data/gnn/runs/<tag>.*
def train(tag, **cfg):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    # CPU by default: on the 2 GB MX550 the GNN spills to shared RAM and runs 3.6x slower than CPU; Colab sets GNN_DEVICE=cuda
    device = os.environ.get("GNN_DEVICE", "cpu")
    e = dict(np.load(EDGES))
    head = np.load(HEAD)
    tr = e["split"] == 0
    head = (head - head[tr].mean(0)) / (head[tr].std(0) + 1e-6)  # standardized on train rows only
    hour = e["minute"] // 60
    # max_hours > 0 keeps only the first hours of each split: a quick smoke run, not a real result
    hours = {s: np.unique(hour[e["split"] == i])[:cfg["max_hours"] or None] for i, s in enumerate(SPLITS)}
    model = Net(cfg["hidden"], head_dim=head.shape[1], use_graph=cfg["name"] == "gnn",
                dropout=cfg["dropout"], head_dropout=cfg["head_dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["pos_weight"], device=device))
    score = np.full(len(hour), np.nan, dtype=np.float32)

    def run(hs, learn):
        model.train(learn)
        total, steps = 0.0, 0
        for h in (rng.permutation(hs) if learn else hs):
            b = batch(e, h, cfg["window_h"], device)
            n = len(b["y"])
            # several updates per hour: the graph is rebuilt per chunk because the weights change in between
            for idx in np.array_split(rng.permutation(n) if learn else np.arange(n), max(1, -(-n // cfg["batch_size"]))):
                i = torch.as_tensor(idx, device=device)
                sub = {**b, "target": (b["target"][0][i], b["target"][1][i]), "target_index": b["target_index"][:, i]}
                with torch.set_grad_enabled(learn):
                    out = model(sub, torch.as_tensor(head[b["rows"].start + idx], device=device))
                if learn:
                    loss = loss_fn(out, b["y"][i])
                    opt.zero_grad()
                    loss.backward()
                    if cfg["clip"]:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                    opt.step()
                    total, steps = total + loss.item(), steps + 1
                score[b["rows"].start + idx] = torch.sigmoid(out).detach().cpu().numpy()
        return total / max(steps, 1)

    va = e["split"] == 1
    best, best_state, waited, t0, history = -1.0, None, 0, time.time(), []
    for epoch in range(cfg["max_epochs"]):
        loss = run(hours["train"], True)
        run(hours["val"], False)
        seen = va & ~np.isnan(score)
        ap = pr_auc(e["label"][seen], score[seen])
        history.append({"epoch": epoch + 1, "train_loss": loss, "val_pr_auc": ap, "seconds": time.time() - t0})
        print(f"  {tag} epoch {epoch + 1}: train loss {loss:.4f}, val PR-AUC {ap:.4f}, {time.time() - t0:.0f} s", flush=True)
        if ap > best:
            best, best_state, waited = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break
        # a diverged run doesn't come back: stop instead of spending the remaining patience on a dead model
        if loss > 10 * min(r["train_loss"] for r in history):
            print(f"  {tag}: train loss jumped to {loss:.4f}, diverged; keeping epoch {int(np.argmax([r['val_pr_auc'] for r in history])) + 1}", flush=True)
            break
    model.load_state_dict(best_state)
    for s in SPLITS:
        run(hours[s], False)
    assert cfg["max_hours"] or not np.isnan(score).any(), "every transaction must get a score"
    # GPU scatter sums aren't bit-exact run to run, so rescoring may move PR-AUC in the 4th decimal at most
    seen = va & ~np.isnan(score)
    assert abs(pr_auc(e["label"][seen], score[seen]) - best) < 1e-3, "rescoring val must reproduce the best epoch"
    mi = psutil.Process().memory_info()
    summary = {"tag": tag, **cfg, "best_val_pr_auc": best, "best_epoch": int(np.argmax([r["val_pr_auc"] for r in history])) + 1,
               "epochs_run": len(history), "seconds": time.time() - t0, "device": device,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
               "peak_ram_gb": getattr(mi, "peak_wset", mi.rss) / 2**30, "history": history}
    RUNS.mkdir(parents=True, exist_ok=True)
    np.save(RUNS / f"{tag}.npy", score)
    torch.save(best_state, RUNS / f"{tag}.pt")
    (RUNS / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"{tag}: best val PR-AUC {best:.4f} at epoch {summary['best_epoch']}, {summary['seconds']:.0f} s on {device} -> {RUNS}")


# one row per finished run -> reports/phase3-gnn-runs.csv
def runs():
    import pandas as pd

    rows = [{k: v for k, v in json.loads(p.read_text()).items() if k != "history"} for p in sorted(RUNS.glob("*.json"))]
    df = pd.DataFrame(rows).sort_values("best_val_pr_auc", ascending=False)
    df.to_csv(ROOT / "reports/phase3-gnn-runs.csv", index=False)
    print(df[["tag", "name", "hidden", "window_h", "lr", "pos_weight", "seed", "best_val_pr_auc", "best_epoch", "epochs_run"]].to_string(index=False))


# a finished run's scores into the local `scores` table as <model>/ABC (e.g. gnn.py load gnn_t1 gnn2)
def load(tag, model):
    import pandas as pd
    from train import save_scores

    e = dict(np.load(EDGES))
    score = np.load(RUNS / f"{tag}.npy")
    assert len(score) == len(e["tx_id"]) and not np.isnan(score).any()
    con = duckdb.connect(str(DB), config={"memory_limit": "1GB", "threads": 2})
    save_scores(con, model, "ABC", pd.DataFrame({"tx_id": e["tx_id"], "split": np.array(SPLITS)[e["split"]]}), score)


if __name__ == "__main__":
    # gnn.py export | bench [window_h] [hidden] | train <tag> [key=value ...] | runs | load <tag> <model>
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "train":
        num = lambda v: float(v) if "." in v or "e" in v else int(v)
        train(args[0], **{k: (v if k == "name" else num(v)) for k, v in (a.split("=") for a in args[1:])})
    elif cmd in ("runs", "load"):
        {"runs": runs, "load": load}[cmd](*args)
    else:
        {"export": export, "bench": bench}[cmd](*map(int, args))
