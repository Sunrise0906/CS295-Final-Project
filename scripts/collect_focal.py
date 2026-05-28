"""Collect focal-ordering training data for ECBS.

Run ECBS (fewest-conflicts focal, cardinal conflict selection) with node logging.
For each solved instance, recover the solution node's ancestors (the on-path set)
and label every generated node 1 if on-path else 0. Save (features, labels).

Run:  python -m scripts.collect_focal
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mapf import random_instance, make_selector
from mapf.ecbs import ECBS
from mapf.strategies.focal import focal_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="8,10,12,14")
    ap.add_argument("--density", default="0.1,0.15,0.2")
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--n-instances", type=int, default=400)
    ap.add_argument("--start-seed", type=int, default=30000)
    ap.add_argument("--time-limit", type=float, default=8.0)
    ap.add_argument("--node-limit", type=int, default=4000)
    ap.add_argument("--out", default="data/trajectories/focal_train.npz")
    args = ap.parse_args()

    agent_choices = [int(x) for x in args.agents.split(",")]
    dens_choices = [float(x) for x in args.density.split(",")]
    feats, labels = [], []
    t0 = time.perf_counter()
    solved = 0
    for k in range(args.n_instances):
        seed = args.start_seed + k
        na = agent_choices[k % len(agent_choices)]
        dn = dens_choices[k % len(dens_choices)]
        inst = random_instance(args.size, args.size, na, dn, seed=seed)
        if inst is None:
            continue
        node_log = []
        solver = ECBS(inst, make_selector("cardinal"), w=args.w,
                      time_limit=args.time_limit, node_limit=args.node_limit,
                      node_log=node_log)
        r = solver.solve()
        if not r.success:
            continue
        solved += 1
        # On-path set: ancestors of the solution node.
        on_path = set()
        n = solver.solution_node
        while n is not None:
            on_path.add(id(n))
            n = n.parent
        for node in node_log:
            feats.append(focal_features(node, inst.n_agents))
            labels.append(1 if id(node) in on_path else 0)
        if (k + 1) % 50 == 0:
            pos = sum(labels)
            print(f"  {k+1}/{args.n_instances} | solved {solved} | "
                  f"{len(labels)} nodes ({pos} on-path) | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, feats=np.stack(feats),
                        labels=np.asarray(labels, dtype=np.int64))
    print(f"\nSaved {len(labels)} nodes ({sum(labels)} on-path, "
          f"{100*sum(labels)/len(labels):.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
