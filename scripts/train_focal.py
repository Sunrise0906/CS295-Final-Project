"""Train the learned focal-ordering model (logistic regression, class-weighted).

Predicts P(node on solution path) from cheap node features. Pure numpy + Adam.

Run:  python -m scripts.train_focal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mapf.strategies.focal import FOCAL_FEATURE_NAMES, N_FOCAL_FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories/focal_train.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--out", default="models/selector_focal.npz")
    args = ap.parse_args()

    d = np.load(args.data)
    X, y = d["feats"], d["labels"].astype(np.float64)
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    Xs = (X - mean) / std
    n, F = Xs.shape
    pos = y.mean()
    print(f"{n} nodes, {F} features, {100*pos:.1f}% on-path")
    # class weights (balance positives)
    w_pos = 0.5 / max(pos, 1e-6)
    w_neg = 0.5 / max(1 - pos, 1e-6)
    sample_w = np.where(y == 1, w_pos, w_neg)

    rng = np.random.default_rng(0)
    w = np.zeros(F)
    b = 0.0
    m_w = np.zeros(F); v_w = np.zeros(F); m_b = v_b = 0.0
    bb1, bb2, eps = 0.9, 0.999, 1e-8
    for ep in range(args.epochs):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) * sample_w
        gw = Xs.T @ g / n + args.l2 * w
        gb = g.mean()
        t = ep + 1
        m_w = bb1*m_w + (1-bb1)*gw; v_w = bb2*v_w + (1-bb2)*gw**2
        m_b = bb1*m_b + (1-bb1)*gb; v_b = bb2*v_b + (1-bb2)*gb**2
        w -= args.lr * (m_w/(1-bb1**t)) / (np.sqrt(v_w/(1-bb2**t)) + eps)
        b -= args.lr * (m_b/(1-bb1**t)) / (np.sqrt(v_b/(1-bb2**t)) + eps)
        if (ep + 1) % max(1, args.epochs // 8) == 0:
            pred = (p >= 0.5).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred==1)&(y==0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            prec = tp / max(1, tp+fp); rec = tp / max(1, tp+fn)
            print(f"  epoch {ep+1:4d}  prec={prec:.3f} rec={rec:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, w=w, b=b, mean=mean, std=std)
    order = np.argsort(-np.abs(w))
    print("\nTop focal features:")
    for i in order[:6]:
        print(f"    {FOCAL_FEATURE_NAMES[i]:>20}: {w[i]:+.3f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
