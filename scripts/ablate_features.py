"""Feature ablation for the linear conflict selector: train on feature subsets
and report held-out oracle-imitation top-1, to see which feature groups drive the
ability to beat cardinal. Cheap (training only, no search). Run:
  python -m scripts.ablate_features
"""
from __future__ import annotations

import numpy as np

from mapf.strategies.features import FEATURE_NAMES, N_FEATURES

# Feature groups by index.
CARD = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]      # MDD widths, singletons, cardinality
COSTDIST = [17, 18, 19, 20]                                # path costs + remaining distance
TIMING = [2, 3]                                            # timestep features
DEGREE = [21, 22, 23]                                      # conflict degree / count
TYPE = [0, 1]

SUBSETS = {
    "all (24)": list(range(N_FEATURES)),
    "cardinality only": CARD,
    "no cardinality": [i for i in range(N_FEATURES) if i not in CARD],
    "cost+dist+timing": COSTDIST + TIMING,
    "cardinality+cost+dist": CARD + COSTDIST,
}


def load_groups(path):
    d = np.load(path)
    feats, groups, labels = d["feats"], d["groups"], d["labels"]
    out, off = [], 0
    for g, y in zip(groups, labels):
        out.append((feats[off:off + g], int(y)))
        off += g
    return out


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def train_eval(data_tr, data_va, idx, epochs=150, lr=0.02, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    F = len(idx)
    w = np.zeros(F); b = 0.0
    mw = np.zeros(F); vw = np.zeros(F); mb = vb = 0.0
    for ep in range(epochs):
        gw = np.zeros(F); gb = 0.0
        for X, y in data_tr:
            Xs = X[:, idx]
            p = softmax(Xs @ w + b)
            dp = p.copy(); dp[y] -= 1
            gw += Xs.T @ dp; gb += dp.sum()
        gw = gw / len(data_tr) + l2 * w; gb /= len(data_tr)
        t = ep + 1
        mw = 0.9*mw+0.1*gw; vw = 0.999*vw+0.001*gw**2
        mb = 0.9*mb+0.1*gb; vb = 0.999*vb+0.001*gb**2
        w -= lr*(mw/(1-0.9**t))/(np.sqrt(vw/(1-0.999**t))+1e-8)
        b -= lr*(mb/(1-0.9**t))/(np.sqrt(vb/(1-0.999**t))+1e-8)
    correct = sum(1 for X, y in data_va if int(np.argmax(X[:, idx] @ w + b)) == y)
    return correct / len(data_va)


def main():
    data = load_groups("data/trajectories/strong_train.npz")
    # standardize once on full features
    allX = np.concatenate([X for X, _ in data], 0)
    mean, std = allX.mean(0), allX.std(0); std = np.where(std < 1e-8, 1.0, std)
    data = [((X - mean) / std, y) for X, y in data]
    rng = np.random.default_rng(0); rng.shuffle(data)
    n_va = len(data) // 5
    va, tr = data[:n_va], data[n_va:]
    print(f"{len(tr)} train / {len(va)} val groups\n")
    print(f"{'feature subset':>24} | held-out oracle top-1")
    for name, idx in SUBSETS.items():
        acc = train_eval(tr, va, idx)
        print(f"{name:>24} | {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
