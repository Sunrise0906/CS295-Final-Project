"""Train the learned conflict selectors on collected oracle data.

We treat conflict selection at a node as a listwise classification: a softmax
over the conflicts' scores, with cross-entropy against the oracle's choice. This
is the imitation-learning objective. Linear and 1-hidden-layer MLP rankers are
both supported; training is pure numpy + Adam.

Run:  python -m scripts.train_selector            # linear
      python -m scripts.train_selector --model mlp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mapf.strategies.features import N_FEATURES, FEATURE_NAMES


def load_groups(path: str):
    d = np.load(path)
    feats, groups, labels = d["feats"], d["groups"], d["labels"]
    out, off = [], 0
    for g, y in zip(groups, labels):
        out.append((feats[off:off + g], int(y)))
        off += g
    return out, feats


def standardize_fit(feats: np.ndarray):
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def train_linear(data, F, epochs, lr, l2, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, F)
    b = 0.0
    m_w = np.zeros(F); v_w = np.zeros(F); m_b = v_b = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for ep in range(epochs):
        rng.shuffle(data)
        total_loss = 0.0
        gw = np.zeros(F); gb = 0.0
        for X, y in data:
            s = X @ w + b
            p = softmax(s)
            total_loss += -np.log(p[y] + 1e-12)
            dp = p.copy(); dp[y] -= 1.0          # dL/ds
            gw += X.T @ dp
            gb += dp.sum()
        gw = gw / len(data) + l2 * w
        gb = gb / len(data)
        step += 1
        # Adam
        m_w = b1 * m_w + (1 - b1) * gw; v_w = b2 * v_w + (1 - b2) * gw**2
        m_b = b1 * m_b + (1 - b1) * gb; v_b = b2 * v_b + (1 - b2) * gb**2
        mhat = m_w / (1 - b1**step); vhat = v_w / (1 - b2**step)
        w -= lr * mhat / (np.sqrt(vhat) + eps)
        mhb = m_b / (1 - b1**step); vhb = v_b / (1 - b2**step)
        b -= lr * mhb / (np.sqrt(vhb) + eps)
        if (ep + 1) % max(1, epochs // 10) == 0:
            acc = top1_accuracy(data, lambda X: X @ w + b)
            print(f"  epoch {ep+1:4d}  loss={total_loss/len(data):.4f}  top1={acc:.3f}",
                  flush=True)
    return w, b


def train_mlp(data, F, hidden, epochs, lr, l2, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.1, (F, hidden)); b1v = np.zeros(hidden)
    W2 = rng.normal(0, 0.1, (hidden, 1)); b2v = np.zeros(1)
    params = [W1, b1v, W2, b2v]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    bb1, bb2, eps = 0.9, 0.999, 1e-8
    step = 0
    for ep in range(epochs):
        rng.shuffle(data)
        total_loss = 0.0
        grads = [np.zeros_like(p) for p in params]
        for X, y in data:
            z1 = X @ W1 + b1v; h = np.tanh(z1)
            s = (h @ W2 + b2v).ravel()
            p = softmax(s)
            total_loss += -np.log(p[y] + 1e-12)
            ds = p.copy(); ds[y] -= 1.0
            grads[2] += h.T @ ds[:, None]
            grads[3] += ds.sum()
            dh = ds[:, None] @ W2.T
            dz1 = dh * (1 - h**2)
            grads[0] += X.T @ dz1
            grads[1] += dz1.sum(axis=0)
        step += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g / len(data) + l2 * p
            m[i] = bb1 * m[i] + (1 - bb1) * g
            v[i] = bb2 * v[i] + (1 - bb2) * g**2
            mhat = m[i] / (1 - bb1**step); vhat = v[i] / (1 - bb2**step)
            p -= lr * mhat / (np.sqrt(vhat) + eps)
        if (ep + 1) % max(1, epochs // 10) == 0:
            acc = top1_accuracy(data, lambda X: (np.tanh(X @ W1 + b1v) @ W2 + b2v).ravel())
            print(f"  epoch {ep+1:4d}  loss={total_loss/len(data):.4f}  top1={acc:.3f}",
                  flush=True)
    return params


def top1_accuracy(data, score_fn) -> float:
    correct = sum(1 for X, y in data if int(np.argmax(score_fn(X))) == y)
    return correct / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories/train.npz")
    ap.add_argument("--model", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data, feats = load_groups(args.data)
    print(f"Loaded {len(data)} ranking groups, {feats.shape[0]} conflicts, "
          f"{N_FEATURES} features")
    mean, std = standardize_fit(feats)
    data = [((X - mean) / std, y) for X, y in data]

    # Majority/cardinal-style sanity baselines.
    print(f"  baseline top1 (always idx 0): "
          f"{sum(1 for _, y in data if y == 0)/len(data):.3f}")

    out = args.out or f"models/selector_{args.model}.npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if args.model == "linear":
        w, b = train_linear(data, N_FEATURES, args.epochs, args.lr, args.l2)
        np.savez(out, w=w, b=b, mean=mean, std=std)
        order = np.argsort(-np.abs(w))
        print("\nTop weighted features:")
        for i in order[:8]:
            print(f"    {FEATURE_NAMES[i]:>16}: {w[i]:+.3f}")
    else:
        W1, b1v, W2, b2v = train_mlp(data, N_FEATURES, args.hidden,
                                     args.epochs, args.lr, args.l2)
        np.savez(out, W1=W1, b1=b1v, W2=W2, b2=b2v, mean=mean, std=std)
    print(f"\nSaved model -> {out}")


if __name__ == "__main__":
    main()
