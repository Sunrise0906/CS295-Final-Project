"""Collect imitation-learning data by running oracle-guided CBS.

For each training instance we run CBS guided by the proxy oracle and log, at
every expanded node, the conflict feature matrix and the oracle's chosen index.
Saved as a ranking dataset: stacked features + per-node group sizes + labels.

Run:  python -m scripts.collect_data
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mapf import random_instance, CBS
from mapf.strategies.oracle import OracleSelector


def collect(args) -> None:
    feats: list[np.ndarray] = []
    groups: list[int] = []
    labels: list[int] = []

    rng_seeds = range(args.start_seed, args.start_seed + args.n_instances)
    agent_choices = [int(x) for x in args.agents.split(",")]
    dens_choices = [float(x) for x in args.density.split(",")]

    t0 = time.perf_counter()
    n_used = 0
    for k, seed in enumerate(rng_seeds):
        na = agent_choices[k % len(agent_choices)]
        dn = dens_choices[k % len(dens_choices)]
        inst = random_instance(args.size, args.size, na, dn, seed=seed)
        if inst is None:
            continue
        log: list = []
        solver = CBS(inst, OracleSelector(log=log),
                     time_limit=args.time_limit, node_limit=args.node_limit)
        res = solver.solve()
        # Keep nodes with >=2 conflicts (a real choice was made).
        for X, idx in log:
            if X.shape[0] >= 2:
                feats.append(X)
                groups.append(X.shape[0])
                labels.append(idx)
        n_used += 1
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{args.n_instances} instances | "
                  f"{len(groups)} nodes | {time.perf_counter()-t0:.1f}s",
                  flush=True)

    if not feats:
        print("No data collected -- try harder instances.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        feats=np.concatenate(feats, axis=0),
        groups=np.asarray(groups, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
    )
    print(f"\nSaved {len(groups)} ranking groups "
          f"({np.concatenate(feats).shape[0]} conflicts) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="6,7,8,9")
    ap.add_argument("--density", default="0.0,0.1,0.15")
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--start-seed", type=int, default=10000)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--node-limit", type=int, default=500)
    ap.add_argument("--out", default="data/trajectories/train.npz")
    collect(ap.parse_args())
