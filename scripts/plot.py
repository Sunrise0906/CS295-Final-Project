"""Aggregate results.csv into report figures and a summary table.

Fairness: per (density, #agents) cell, expansion/runtime means are computed over
the *common* set of instances solved by all plotted strategies. Success rate is
over all instances. No pandas dependency (csv + numpy + matplotlib).

Run:  python -m scripts.plot
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Display order, labels, and colors.
STYLE = {
    "first":          ("first (naive)",      "#9e9e9e"),
    "random":         ("random",             "#bdbd6b"),
    "earliest":       ("earliest-time",      "#c97b7b"),
    "most-conflicts": ("most-conflicts",     "#7b9ec9"),
    "cardinal":       ("cardinal (ICBS)",    "#1f77b4"),
    "learned-linear": ("learned-linear",     "#2ca02c"),
    "learned-mlp":    ("learned-MLP",        "#d62728"),
    "oracle":         ("oracle (lookahead)", "#000000"),
}


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load(args.results)
    # RQ1/RQ3 figures show the hand-crafted strategies only; the learned-vs-
    # cardinal comparison is reported separately (fig_learned_ratio).
    hardcoded = ["first", "random", "earliest", "most-conflicts", "cardinal"]
    strategies = [s for s in hardcoded if any(r["strategy"] == s for r in rows)]
    densities = sorted({float(r["density"]) for r in rows})
    agents = sorted({int(r["n_agents"]) for r in rows})

    # idx[(strategy, density, n_agents, seed)] = row
    idx = {(r["strategy"], float(r["density"]), int(r["n_agents"]), int(r["seed"])): r
           for r in rows}
    seeds_by_cell = defaultdict(set)
    for r in rows:
        seeds_by_cell[(float(r["density"]), int(r["n_agents"]))].add(int(r["seed"]))

    def success_rate(s, d, n):
        vals = [int(idx[(s, d, n, sd)]["success"])
                for sd in seeds_by_cell[(d, n)] if (s, d, n, sd) in idx]
        return np.mean(vals) if vals else np.nan

    def common_mean(metric, s, d, n):
        # seeds solved by ALL strategies in this cell
        good = []
        for sd in seeds_by_cell[(d, n)]:
            if all((st, d, n, sd) in idx and int(idx[(st, d, n, sd)]["success"]) == 1
                   for st in strategies):
                good.append(sd)
        vals = [float(idx[(s, d, n, sd)][metric]) for sd in good
                if (s, d, n, sd) in idx]
        return (np.mean(vals) if vals else np.nan), len(good)

    # --- Figure 1: success rate vs #agents ---
    _grid_plot(args.outdir, "fig_success.png", densities, agents, strategies,
               lambda s, d, n: success_rate(s, d, n),
               ylabel="success rate", ylog=False, title_prefix="Success rate")

    # --- Figure 2: expansions vs #agents (common-solved, log) ---
    _grid_plot(args.outdir, "fig_expansions.png", densities, agents, strategies,
               lambda s, d, n: common_mean("expansions", s, d, n)[0],
               ylabel="mean high-level expansions", ylog=True,
               title_prefix="Search effort")

    # --- Figure 3: runtime vs #agents (common-solved, log) ---
    _grid_plot(args.outdir, "fig_runtime.png", densities, agents, strategies,
               lambda s, d, n: common_mean("runtime", s, d, n)[0],
               ylabel="mean runtime (s)", ylog=True, title_prefix="Runtime")

    # --- Summary table ---
    out_csv = os.path.join(args.outdir, "..", "summary.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["density", "n_agents", "strategy", "success_rate",
                     "mean_expansions_common", "mean_runtime_common", "n_common"])
        for d in densities:
            for n in agents:
                for s in strategies:
                    me, nc = common_mean("expansions", s, d, n)
                    mr, _ = common_mean("runtime", s, d, n)
                    wr.writerow([d, n, s, f"{success_rate(s,d,n):.3f}",
                                 f"{me:.1f}" if np.isfinite(me) else "",
                                 f"{mr:.4f}" if np.isfinite(mr) else "", nc])
    print(f"Wrote figures to {args.outdir} and summary -> {out_csv}")


def _grid_plot(outdir, fname, densities, agents, strategies, fn, ylabel, ylog, title_prefix):
    nd = len(densities)
    fig, axes = plt.subplots(1, nd, figsize=(5 * nd, 4), squeeze=False)
    for j, d in enumerate(densities):
        ax = axes[0][j]
        for s in strategies:
            ys = [fn(s, d, n) for n in agents]
            label, color = STYLE[s]
            ax.plot(agents, ys, marker="o", ms=4, label=label, color=color)
        ax.set_title(f"{title_prefix}  (obstacle density {d:g})")
        ax.set_xlabel("number of agents")
        ax.set_ylabel(ylabel)
        if ylog:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


if __name__ == "__main__":
    main()
