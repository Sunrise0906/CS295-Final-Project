"""Collect imitation data from the STRONG (subtree-minimizing) oracle.

One expensive pass produces BOTH datasets:
  - memoryless ranking data  -> data/trajectories/strong_train.npz
  - sequence-model data      -> data/trajectories/strong_seq.pkl

Runs on the pure-numpy core. Run: python -m scripts.collect_strong
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from mapf import random_instance, CBS
from mapf.strategies.oracle import StrongOracleSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="6,7,8,9,10")
    ap.add_argument("--density", default="0.0,0.1,0.15")
    ap.add_argument("--n-instances", type=int, default=250)
    ap.add_argument("--start-seed", type=int, default=20000)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--node-limit", type=int, default=200)
    ap.add_argument("--subtree-node-limit", type=int, default=400)
    ap.add_argument("--subtree-time-limit", type=float, default=1.0)
    ap.add_argument("--out-npz", default="data/trajectories/strong_train.npz")
    ap.add_argument("--out-pkl", default="data/trajectories/strong_seq.pkl")
    ap.add_argument("--out-gnn", default=None,
                    help="If set, also save (X, agent_pairs, idx) tuples for GNN.")
    ap.add_argument("--features", choices=["v1", "ext"], default="v1")
    ap.add_argument("--rollout-linear", default=None,
                    help="Path to a linear .npz model to use as the oracle's "
                         "rollout policy instead of cardinal (policy iteration).")
    args = ap.parse_args()

    if args.features == "ext":
        from mapf.strategies.features import extract_node_features_ext as feat_fn
    else:
        feat_fn = None  # default v1 in StrongOracleSelector

    if args.rollout_linear:
        from mapf.strategies.learned import LearnedLinearSelector
        rollout_model = LearnedLinearSelector.load(args.rollout_linear)
        print(f"Using {args.rollout_linear} as oracle rollout policy "
              f"(dim {rollout_model.w.shape[0]})")
    else:
        rollout_model = None

    agent_choices = [int(x) for x in args.agents.split(",")]
    dens_choices = [float(x) for x in args.density.split(",")]

    feats, groups, labels = [], [], []
    seq_records = []
    gnn_records = []
    t0 = time.perf_counter()
    for k in range(args.n_instances):
        seed = args.start_seed + k
        na = agent_choices[k % len(agent_choices)]
        dn = dens_choices[k % len(dens_choices)]
        inst = random_instance(args.size, args.size, na, dn, seed=seed)
        if inst is None:
            continue
        log, seq_log = [], []
        gnn_log = [] if args.out_gnn else None
        sel = StrongOracleSelector(
            log=log, seq_log=seq_log, gnn_log=gnn_log,
            subtree_node_limit=args.subtree_node_limit,
            subtree_time_limit=args.subtree_time_limit,
            feature_fn=feat_fn,
            rollout_selector=rollout_model)
        CBS(inst, sel, time_limit=args.time_limit,
            node_limit=args.node_limit, track_history=True).solve()
        for X, idx in log:
            if X.shape[0] >= 2:
                feats.append(X)
                groups.append(X.shape[0])
                labels.append(idx)
        for hist, X, idx in seq_log:
            if X.shape[0] >= 2:
                seq_records.append((hist, X, idx))
        if gnn_log is not None:
            for X, pairs, idx in gnn_log:
                if X.shape[0] >= 2:
                    gnn_records.append((X, pairs, idx))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{args.n_instances} | {len(groups)} nodes | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        feats=np.concatenate(feats, axis=0),
        groups=np.asarray(groups, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
    )
    with open(args.out_pkl, "wb") as f:
        pickle.dump(seq_records, f)
    print(f"\nSaved {len(groups)} ranking groups -> {args.out_npz}")
    print(f"Saved {len(seq_records)} sequence records -> {args.out_pkl}")
    if args.out_gnn:
        with open(args.out_gnn, "wb") as f:
            pickle.dump(gnn_records, f)
        print(f"Saved {len(gnn_records)} GNN records -> {args.out_gnn}")


if __name__ == "__main__":
    main()
