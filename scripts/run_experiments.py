"""Sweep conflict-selection strategies over a random-grid benchmark.

Writes one CSV row per (strategy, instance). Aggregation/plots are done
separately in ``scripts.plot``.

Run:  python -m scripts.run_experiments
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from mapf import random_instance, CBS, make_selector, validate

FIELDS = ["strategy", "height", "width", "density", "n_agents", "seed",
          "success", "cost", "expansions", "generated", "runtime", "reason"]


def build_selectors(names: list[str], model_dir: str) -> dict:
    sels = {}
    for name in names:
        if name in ("learned-linear", "learned-mlp"):
            from mapf.strategies.learned import LearnedLinearSelector, LearnedMLPSelector
            path = os.path.join(model_dir, f"selector_{name.split('-')[1]}.npz")
            if not os.path.exists(path):
                print(f"  [skip] {name}: model not found at {path}")
                continue
            cls = LearnedLinearSelector if name == "learned-linear" else LearnedMLPSelector
            sels[name] = ("static", cls.load(path))
        elif name == "oracle":
            sels[name] = ("factory", lambda s=name: _oracle())
        else:
            sels[name] = ("factory", (lambda nm=name: make_selector(nm)))
    return sels


def _oracle():
    from mapf.strategies.oracle import OracleSelector
    return OracleSelector()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="4,6,8,10,12,14")
    ap.add_argument("--density", default="0.0,0.1,0.2")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--node-limit", type=int, default=5000)
    ap.add_argument("--strategies",
                    default="first,random,earliest,most-conflicts,cardinal,learned-linear")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    names = args.strategies.split(",")
    sels = build_selectors(names, args.model_dir)
    agent_list = [int(x) for x in args.agents.split(",")]
    dens_list = [float(x) for x in args.density.split(",")]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_rows = 0
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for dn in dens_list:
            for na in agent_list:
                for s in range(args.start_seed, args.start_seed + args.seeds):
                    inst = random_instance(args.size, args.size, na, dn, seed=s)
                    if inst is None:
                        continue
                    for name, (kind, obj) in sels.items():
                        sel = obj if kind == "static" else obj()
                        res = CBS(inst, sel, time_limit=args.time_limit,
                                  node_limit=args.node_limit).solve()
                        if res.success:
                            ok, msg = validate(inst, res.paths)
                            if not ok:
                                raise RuntimeError(f"INVALID {name} d{dn} n{na} s{s}: {msg}")
                        wr.writerow({
                            "strategy": name, "height": args.size, "width": args.size,
                            "density": dn, "n_agents": na, "seed": s,
                            "success": int(res.success), "cost": res.cost,
                            "expansions": res.expansions, "generated": res.generated,
                            "runtime": round(res.runtime, 4), "reason": res.reason,
                        })
                        n_rows += 1
                    f.flush()
                print(f"  density {dn} agents {na} done "
                      f"({time.perf_counter()-t0:.1f}s, {n_rows} rows)", flush=True)
    print(f"\nWrote {n_rows} rows -> {args.out}")


if __name__ == "__main__":
    main()
