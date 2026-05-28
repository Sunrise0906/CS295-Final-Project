"""Analyze results/optimal_learned.csv with per-instance ratio statistics
(robust to heavy-tailed variance) and make figures.

Run:  python -m scripts.analyze_learned
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/optimal_learned.csv")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = list(csv.DictReader(open(args.csv)))
    has_runtime = rows and "runtime" in rows[0]
    # cell[(density, agents, seed)][method] = (success, expansions, runtime)
    cell = defaultdict(dict)
    for r in rows:
        cell[(float(r["density"]), int(r["agents"]), int(r["seed"]))][r["method"]] = (
            int(r["success"]), int(r["expansions"]),
            float(r["runtime"]) if has_runtime else 0.0)
    densities = sorted({float(r["density"]) for r in rows})
    agents = sorted({int(r["agents"]) for r in rows})
    learned = ["linear", "mlp"]

    def success_rate(d, n, m):
        vals = [v[m][0] for k, v in cell.items() if k[0] == d and k[1] == n and m in v]
        return np.mean(vals) if vals else float("nan")

    def ratios(d, n, m, metric=1):
        out = []
        for k, v in cell.items():
            if k[0] == d and k[1] == n and "cardinal" in v and m in v:
                if v["cardinal"][0] and v[m][0] and v["cardinal"][metric] > 0:
                    out.append(v[m][metric] / v["cardinal"][metric])
        return np.array(out)

    def report(metric, label):
        print(f"\nPer-instance {label} ratio vs cardinal (median | geomean | win-rate | n)")
        for d in densities:
            print(f"== density {d} ==")
            for n in agents:
                line = f"  agents {n:>2}: "
                for m in learned:
                    r = ratios(d, n, m, metric)
                    if len(r):
                        gm = math.exp(np.mean(np.log(np.maximum(r, 1e-6))))
                        line += (f"{m}={np.median(r):.2f}|{gm:.2f}|"
                                 f"{100*np.mean(r < 1):.0f}%|{len(r)}  ")
                    else:
                        line += f"{m}=-  "
                print(line)

    report(1, "expansion")
    if has_runtime:
        report(2, "runtime")

    # Figure: median ratio vs agents per density.
    fig, axes = plt.subplots(1, len(densities), figsize=(5*len(densities), 4), squeeze=False)
    for j, d in enumerate(densities):
        ax = axes[0][j]
        ax.axhline(1.0, color="#1f77b4", ls="--", label="cardinal (=1)")
        for m, col in [("linear", "#2ca02c"), ("mlp", "#d62728")]:
            ys = [np.median(ratios(d, n, m)) if len(ratios(d, n, m)) else np.nan
                  for n in agents]
            ax.plot(agents, ys, marker="o", color=col, label=f"learned-{m}")
        ax.set_title(f"median expansions ratio (density {d:g})")
        ax.set_xlabel("agents"); ax.set_ylabel("expansions / cardinal")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/fig_learned_ratio.png", dpi=130)
    plt.close(fig)
    print(f"\nsaved {args.outdir}/fig_learned_ratio.png")


if __name__ == "__main__":
    main()
