import sys
import time

import duckdb
import numpy as np
import psutil
import torch
from torch import nn
from torch_geometric.nn import GINEConv

from load import DB, EXPECTED_ROWS, ROOT

EDGES = ROOT / "data/gnn/edges.npz"
SPLITS = ["train", "val", "test", "tail"]
# model inputs per edge (set A, encoded); tx_id, split and label are kept out of this list on purpose
EDGE_NUM = ["log_amount_usd", "hour", "is_cross_bank", "is_currency_mismatch", "is_self_transfer"]
EDGE_CAT = ["fmt", "pay_cur", "recv_cur"]


# all transactions sorted by time, accounts and categories as small ints; minute = minutes since Sep 1 00:00
def export():
    con = duckdb.connect(str(DB), read_only=True, config={"memory_limit": "700MB", "threads": 2})
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
    return {"x": tt(deg.astype(np.float32)), "edge_index": tt(np.stack([src, dst]).astype(np.int64)), "edge": edge(g),
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
    def __init__(self, hidden=64, layers=2, head_dim=0, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
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
                h = torch.relu(mx(torch.cat([ci(h, b["edge_index"], ea), co(h, b["edge_index"].flip(0), ea)], 1)))
            u, v = b["target_index"]
            parts = [h[u], h[v]] + parts
        if head_feats is not None:
            parts.append(head_feats)
        return self.head(torch.cat(parts, 1)).squeeze(1)


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


if __name__ == "__main__":
    {"export": export, "bench": bench}[sys.argv[1]]()
