"""Collect sequence-model training data: for each expanded node, the
root-to-node history of resolved-conflict features, the node's candidate
conflict features, and the oracle's chosen index.

Runs on the pure-numpy core (no torch). Output is a pickle of records
``(history: list[np.ndarray(F)], X: np.ndarray(n,F), label: int)``.

Run:  python -m scripts.collect_seq_data
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from mapf import random_instance, CBS
from mapf.strategies.oracle import OracleSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="6,7,8,9")
    ap.add_argument("--density", default="0.0,0.1,0.15")
    ap.add_argument("--n-instances", type=int, default=160)
    ap.add_argument("--start-seed", type=int, default=10000)
    ap.add_argument("--time-limit", type=float, default=8.0)
    ap.add_argument("--node-limit", type=int, default=300)
    ap.add_argument("--out", default="data/trajectories/seq_train.pkl")
    args = ap.parse_args()

    agent_choices = [int(x) for x in args.agents.split(",")]
    dens_choices = [float(x) for x in args.density.split(",")]
    records = []
    t0 = time.perf_counter()
    for k in range(args.n_instances):
        seed = args.start_seed + k
        na = agent_choices[k % len(agent_choices)]
        dn = dens_choices[k % len(dens_choices)]
        inst = random_instance(args.size, args.size, na, dn, seed=seed)
        if inst is None:
            continue
        seq_log = []
        sel = OracleSelector(seq_log=seq_log)
        CBS(inst, sel, time_limit=args.time_limit,
            node_limit=args.node_limit, track_history=True).solve()
        for hist, X, idx in seq_log:
            if X.shape[0] >= 2:
                records.append((hist, X, idx))
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{args.n_instances} | {len(records)} nodes | "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(records, f)
    avg_hist = sum(len(h) for h, _, _ in records) / max(1, len(records))
    print(f"\nSaved {len(records)} records (avg history len {avg_hist:.1f}) -> {out}")


if __name__ == "__main__":
    main()
