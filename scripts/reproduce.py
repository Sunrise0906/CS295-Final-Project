"""One-command reproduction of the core results (optimal-CBS track).

Runs the full pipeline in order with the numpy-only core (Python 3.14 ok):
  smoke test -> RQ1/RQ3 sweep + figures -> strong-oracle data -> train linear+MLP
  -> rigorous learned-vs-cardinal sweep -> per-instance analysis + figure.

The ECBS-focal and sequence-model (torch / mlenv) tracks are optional and listed
in the README; they are not run here to keep this dependency-light.

Usage:  python -m scripts.reproduce          # full (slow, ~30-40 min)
        python -m scripts.reproduce --quick   # smaller sweeps for a fast check
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

PY = sys.executable


def run(desc, args):
    print(f"\n{'='*70}\n>>> {desc}\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([PY, "-m"] + args)
    if r.returncode != 0:
        print(f"!! step failed: {desc}")
        sys.exit(r.returncode)
    print(f"--- done in {time.perf_counter()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smaller sweeps/seeds for a fast sanity reproduction")
    a = ap.parse_args()
    seeds = "12" if a.quick else "20"
    big_seeds = "20" if a.quick else "80"
    n_inst = "80" if a.quick else "250"

    run("Correctness smoke tests", ["scripts.smoke_test"])
    run("RQ1/RQ3 hand-crafted strategy sweep",
        ["scripts.run_experiments", "--seeds", seeds,
         "--strategies", "first,random,earliest,most-conflicts,cardinal"])
    run("RQ1/RQ3 figures", ["scripts.plot"])
    run("Strong-oracle imitation data",
        ["scripts.collect_strong", "--n-instances", n_inst])
    run("Train linear ranker",
        ["scripts.train_selector", "--data", "data/trajectories/strong_train.npz",
         "--model", "linear", "--out", "models/selector_linear.npz"])
    run("Train MLP ranker",
        ["scripts.train_selector", "--data", "data/trajectories/strong_train.npz",
         "--model", "mlp", "--out", "models/selector_mlp.npz"])
    run("Rigorous learned-vs-cardinal sweep",
        ["scripts.sweep_learned", "--seeds", big_seeds])
    run("Per-instance analysis + RQ2 figure", ["scripts.analyze_learned"])
    run("Feature ablation", ["scripts.ablate_features"])
    print("\nAll core results reproduced. Figures in results/figures/, "
          "models in models/, CSVs in results/.")


if __name__ == "__main__":
    main()
