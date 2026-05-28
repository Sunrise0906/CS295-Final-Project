"""Per-instance sweep of learned selectors vs cardinal in optimal CBS.

Writes one row per (density, agents, seed) with each method's expansions and
runtime, so we can report per-instance ratio statistics (median, geomean,
win-rate). Run: python -m scripts.sweep_learned
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mapf import random_instance, CBS, make_selector, validate
from mapf.strategies.learned import LearnedLinearSelector, LearnedMLPSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", default="models/selector_linear.npz")
    ap.add_argument("--mlp", default="models/selector_mlp.npz")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="6,8,10,12,14")
    ap.add_argument("--density", default="0.0,0.1,0.2")
    ap.add_argument("--seeds", type=int, default=80)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--node-limit", type=int, default=8000)
    ap.add_argument("--out", default="results/optimal_learned.csv")
    args = ap.parse_args()

    lin = LearnedLinearSelector.load(args.linear)
    mlp = LearnedMLPSelector.load(args.mlp)
    methods = [("cardinal", lambda: make_selector("cardinal")),
               ("linear", lambda: lin),
               ("mlp", lambda: mlp)]
    agent_list = [int(x) for x in args.agents.split(",")]
    dens_list = [float(x) for x in args.density.split(",")]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["density", "agents", "seed", "method", "success",
                     "expansions", "cost", "runtime"])
        for dn in dens_list:
            for na in agent_list:
                for s in range(args.seeds):
                    inst = random_instance(args.size, args.size, na, dn, seed=s)
                    if inst is None:
                        continue
                    for name, fac in methods:
                        r = CBS(inst, fac(), time_limit=args.time_limit,
                                node_limit=args.node_limit).solve()
                        if r.success:
                            v, _ = validate(inst, r.paths)
                            assert v
                        wr.writerow([dn, na, s, name, int(r.success),
                                     r.expansions, r.cost, round(r.runtime, 5)])
                    f.flush()
                print(f"  density {dn} agents {na} done", flush=True)
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
