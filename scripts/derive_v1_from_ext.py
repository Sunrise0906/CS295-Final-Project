"""Derive a v1-feature dataset from an ext-feature dataset by slicing the first
N_FEATURES columns. Lets us train a v1-feature baseline on the same instances
the ext models are trained on, for fair comparison.

Usage:
  python -m scripts.derive_v1_from_ext --in data/trajectories/strong_train_ext.npz \\
      --out data/trajectories/strong_train_v1_from_ext.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from mapf.strategies.features import N_FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/trajectories/strong_train_ext.npz")
    ap.add_argument("--out", default="data/trajectories/strong_train_v1_from_ext.npz")
    args = ap.parse_args()

    d = np.load(args.inp)
    feats = d["feats"][:, :N_FEATURES]
    np.savez_compressed(args.out, feats=feats, groups=d["groups"], labels=d["labels"])
    print(f"Sliced {feats.shape[1]} features (from {d['feats'].shape[1]}); "
          f"saved {feats.shape[0]} conflict rows -> {args.out}")


if __name__ == "__main__":
    main()
