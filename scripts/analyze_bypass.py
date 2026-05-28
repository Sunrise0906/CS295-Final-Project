"""Analyze the bypass sweep CSV with per-instance ratio statistics."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/optimal_bypass.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    cell = defaultdict(dict)
    for r in rows:
        cell[(float(r["density"]), int(r["agents"]), int(r["seed"]))][r["method"]] = (
            int(r["success"]), int(r["expansions"]),
            float(r.get("runtime", 0.0)))
    densities = sorted({float(r["density"]) for r in rows})
    agents = sorted({int(r["agents"]) for r in rows})
    methods = ["cardinal+bp", "linear", "linear+bp"]

    def report(metric, label):
        idx = 1 if metric == "exp" else 2
        print(f"\nPer-instance {label} ratio vs cardinal "
              f"(median | geomean | win-rate | n)")
        for d in densities:
            print(f"== density {d} ==")
            for n in agents:
                line = f"  agents {n:>2}: "
                for m in methods:
                    out = []
                    for k, v in cell.items():
                        if k[0] != d or k[1] != n:
                            continue
                        if "cardinal" not in v or m not in v:
                            continue
                        sc, ec, rc = v["cardinal"]
                        sm, em, rm = v[m]
                        if not (sc and sm):
                            continue
                        out.append((rm if metric == "rt" else em) /
                                   max(1e-9, (rc if metric == "rt" else ec)))
                    out = np.array(out) if out else np.array([])
                    if len(out):
                        gm = math.exp(np.mean(np.log(np.maximum(out, 1e-6))))
                        line += (f"{m}={np.median(out):.2f}|{gm:.2f}|"
                                 f"{100*np.mean(out < 1):.0f}%|{len(out)}  ")
                    else:
                        line += f"{m}=-  "
                print(line)

    report("exp", "expansion")
    report("rt", "runtime")


if __name__ == "__main__":
    main()
