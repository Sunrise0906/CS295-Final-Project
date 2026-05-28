"""Evaluate focal node-ordering strategies in ECBS (all use cardinal conflict
selection). Reports success rate and, on commonly-solved instances, mean AND
median expansions (median is robust to the high variance of learned ordering).

Methods: fewest-conflicts (standard), learned-focal (pure model),
blended-focal (fewest-conflicts primary + learned soft adjustment).

Run:  python -m scripts.eval_focal
"""
from __future__ import annotations

import argparse
import statistics as st

from mapf import random_instance, make_selector, validate
from mapf.ecbs import ECBS, FewestConflicts
from mapf.strategies.focal import LearnedFocalSelector, BlendedFocalSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal-model", default="models/selector_focal.npz")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="8,10,12,14")
    ap.add_argument("--density", default="0.15")
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--time-limit", type=float, default=8.0)
    ap.add_argument("--node-limit", type=int, default=5000)
    args = ap.parse_args()

    dens = float(args.density)
    agent_list = [int(x) for x in args.agents.split(",")]
    names = ["fewest", "learned", "blended"]
    print(f"ECBS w={args.w} density={dens} lam={args.lam}  (mean/median expansions; "
          f"ratio vs fewest on common-solved)")
    print(f"{'agents':>6} | {'succ(f/l/b)':>14} | {'common':>6} | "
          f"{'fewest':>12} {'learned':>16} {'blended':>16}")
    for na in agent_list:
        def make(name):
            if name == "fewest":
                return FewestConflicts()
            if name == "learned":
                return LearnedFocalSelector.load(args.focal_model, na)
            return BlendedFocalSelector.load(args.focal_model, na, lam=args.lam)

        exps = {m: [] for m in names}
        succ = {m: 0 for m in names}
        total = 0
        common_rows = {m: [] for m in names}
        per_seed = {}
        for s in range(args.seeds):
            inst = random_instance(args.size, args.size, na, dens, seed=s)
            if inst is None:
                continue
            total += 1
            rows = {}
            for m in names:
                r = ECBS(inst, make_selector("cardinal"), node_selector=make(m),
                         w=args.w, time_limit=args.time_limit,
                         node_limit=args.node_limit).solve()
                if r.success:
                    succ[m] += 1
                    v, msg = validate(inst, r.paths); assert v, msg
                    rows[m] = r.expansions
            if len(rows) == len(names):
                for m in names:
                    common_rows[m].append(rows[m])
        common = len(common_rows["fewest"])
        sr = "/".join(f"{succ[m]/total:.2f}" for m in names) if total else "-"
        if common:
            fmean = st.mean(common_rows["fewest"])
            cells = {}
            for m in names:
                mn = st.mean(common_rows[m]); md = st.median(common_rows[m])
                cells[m] = f"{mn:.0f}/{md:.0f}({mn/fmean:.2f})"
            print(f"{na:>6} | {sr:>14} | {common:>6} | "
                  f"{cells['fewest']:>12} {cells['learned']:>16} {cells['blended']:>16}",
                  flush=True)
        else:
            print(f"{na:>6} | {sr:>14} | {common:>6} | (no common)")


if __name__ == "__main__":
    main()
