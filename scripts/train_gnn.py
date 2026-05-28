"""Train the GNN conflict selector on collected GNN data (X, agent_pairs, idx).

Run in mlenv:
  D:/software/anaconda3/envs/mlenv/python.exe -m scripts.train_gnn
"""
from __future__ import annotations

import argparse
import copy
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from mapf.strategies.gnn import ConflictGNN, build_edge_index
from mapf.strategies.features import N_EXT_FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories/strong_gnn.pkl")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="models/selector_gnn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.data, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} GNN records | device={device}")

    # Standardize features.
    allX = np.concatenate([X for X, _, _ in records], axis=0)
    mean, std = allX.mean(0), allX.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    F_dim = allX.shape[1]
    print(f"Feature dim {F_dim} (expected {N_EXT_FEATURES})")

    def prep(rec):
        X, pairs, y = rec
        Xn = (X - mean) / std
        edges = build_edge_index(pairs)
        return (torch.tensor(Xn, dtype=torch.float32, device=device),
                torch.tensor(edges, dtype=torch.int64, device=device),
                int(y))

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(records))
    n_val = int(len(records) * args.val_frac)
    val = [prep(records[i]) for i in sorted(perm[:n_val].tolist())]
    train = [prep(records[i]) for i in perm[n_val:]]
    print(f"train={len(train)} val={len(val)}")

    model = ConflictGNN(F_dim, hidden=args.hidden, n_layers=args.layers,
                        dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    def top1(data):
        model.eval()
        c = 0
        with torch.no_grad():
            for X, e, y in data:
                if int(model(X, e).argmax()) == y:
                    c += 1
        return c / max(1, len(data))

    t0 = time.perf_counter()
    best_val, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train)
        tot = 0.0
        for X, e, y in train:
            opt.zero_grad()
            s = model(X, e).unsqueeze(0)
            loss = F.cross_entropy(s, torch.tensor([y], device=device))
            loss.backward()
            opt.step()
            tot += loss.item()
        v = top1(val)
        if v > best_val:
            best_val = v
            best_state = copy.deepcopy(model.state_dict())
            # Persist best on the fly so an interrupted run still leaves a usable
            # checkpoint behind.
            torch.save({
                "state_dict": best_state,
                "arch": {"n_features": F_dim, "hidden": args.hidden,
                         "n_layers": args.layers, "dropout": args.dropout},
                "mean": mean, "std": std,
            }, args.out)
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            print(f"  epoch {ep+1:3d} loss={tot/len(train):.4f} "
                  f"train_top1={top1(train):.3f} val_top1={v:.3f} "
                  f"best={best_val:.3f} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    torch.save({
        "state_dict": best_state,
        "arch": {"n_features": F_dim, "hidden": args.hidden,
                 "n_layers": args.layers, "dropout": args.dropout},
        "mean": mean, "std": std,
    }, args.out)
    print(f"\nSaved best-val GNN -> {args.out} | best val_top1={best_val:.3f}")


if __name__ == "__main__":
    main()
