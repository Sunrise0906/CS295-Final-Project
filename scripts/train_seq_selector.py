"""Train the sequence-model conflict selector by imitation of the oracle.

Run IN THE mlenv CONDA ENV (torch + CUDA):
  D:/software/anaconda3/envs/mlenv/python.exe -m scripts.train_seq_selector
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from mapf.strategies.features import N_FEATURES
from mapf.strategies.sequence import SeqConflictNet


def fit_standardizer(records):
    allX = [X for _, X, _ in records]
    allH = [np.stack(h) for h, _, _ in records if len(h) > 0]
    stacked = np.concatenate(allX + allH, axis=0) if allH else np.concatenate(allX, 0)
    mean = stacked.mean(0)
    std = stacked.std(0)
    return mean, np.where(std < 1e-8, 1.0, std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories/seq_train.pkl")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="models/selector_seq.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.data, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} nodes | device={device}")

    mean, std = fit_standardizer(records)

    def prep(rec):
        h, X, y = rec
        Hs = (np.stack(h) - mean) / std if len(h) else np.zeros((0, N_FEATURES))
        Xs = (X - mean) / std
        return (torch.tensor(Hs, dtype=torch.float32, device=device),
                torch.tensor(Xs, dtype=torch.float32, device=device),
                int(y))

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(records))
    n_val = int(len(records) * args.val_frac)
    val_idx, tr_idx = set(perm[:n_val].tolist()), perm[n_val:].tolist()
    train = [prep(records[i]) for i in tr_idx]
    val = [prep(records[i]) for i in sorted(val_idx)]

    model = SeqConflictNet(N_FEATURES, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def top1(data):
        model.eval()
        c = 0
        with torch.no_grad():
            for H, X, y in data:
                if int(model(H, X).argmax()) == y:
                    c += 1
        return c / max(1, len(data))

    import copy
    t0 = time.perf_counter()
    best_val, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train)
        tot = 0.0
        for H, X, y in train:
            opt.zero_grad()
            s = model(H, X).unsqueeze(0)
            loss = F.cross_entropy(s, torch.tensor([y], device=device))
            loss.backward()
            opt.step()
            tot += loss.item()
        v = top1(val)
        if v > best_val:                       # early stopping on val top-1
            best_val = v
            best_state = copy.deepcopy(model.state_dict())
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            print(f"  epoch {ep+1:3d} loss={tot/len(train):.4f} "
                  f"train_top1={top1(train):.3f} val_top1={v:.3f} "
                  f"best={best_val:.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    torch.save({
        "state_dict": best_state,              # save BEST-val model, not final
        "arch": {"n_features": N_FEATURES, "hidden": args.hidden},
        "mean": mean, "std": std,
    }, args.out)
    print(f"Saved best-val model -> {args.out} | best val_top1={best_val:.3f}")


if __name__ == "__main__":
    main()
