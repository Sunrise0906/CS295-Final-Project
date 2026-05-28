"""Evaluate the GNN selector against cardinal and the (ext-feature) linear/MLP
rankers on held-out instances. Per-instance ratios (median, geomean, win-rate).
Run in mlenv (needs torch):
  D:/software/anaconda3/envs/mlenv/python.exe -m scripts.eval_gnn
"""
from __future__ import annotations

import argparse
import math
import statistics as st

import numpy as np

from mapf import random_instance, CBS, make_selector, validate
from mapf.strategies.learned import LearnedLinearSelector, LearnedMLPSelector
from mapf.strategies.gnn import GNNSelector, EnsembleSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear-v1", default="models/selector_linear.npz")
    ap.add_argument("--linear", default="models/selector_linear_ext.npz")
    ap.add_argument("--mlp", default="models/selector_mlp_ext.npz")
    ap.add_argument("--gnn", default="models/selector_gnn.pt")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="8,10,12,14")
    ap.add_argument("--density", default="0.1,0.2")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--node-limit", type=int, default=8000)
    args = ap.parse_args()

    import torch
    # GNN inference on tiny conflict graphs is faster on CPU than GPU
    # (CPU<->GPU transfer dominates), so force CPU regardless of CUDA.
    device = "cpu"
    lin_v1 = LearnedLinearSelector.load(args.linear_v1)
    lin = LearnedLinearSelector.load(args.linear)
    mlp = LearnedMLPSelector.load(args.mlp)
    gnn = GNNSelector.load(args.gnn, device=device)

    ensemble = EnsembleSelector([lin, gnn])
    methods = [("cardinal", lambda: make_selector("cardinal")),
               ("linear-v1", lambda: lin_v1),
               ("linear-ext", lambda: lin),
               ("mlp-ext", lambda: mlp),
               ("gnn", lambda: gnn),
               ("ensemble", lambda: ensemble)]

    agent_list = [int(x) for x in args.agents.split(",")]
    dens_list = [float(x) for x in args.density.split(",")]

    print(f"device={device}, ratios = method-expansions / cardinal-expansions")
    print(f"{'density':>7} {'agents':>6} {'common':>6} | " +
          " ".join(f"{m:>16}" for m, _ in methods[1:]))
    for dn in dens_list:
        for na in agent_list:
            rows = {m: [] for m, _ in methods}
            common = 0
            for s in range(args.seeds):
                inst = random_instance(args.size, args.size, na, dn, seed=s)
                if inst is None:
                    continue
                ok, e = True, {}
                for name, fac in methods:
                    r = CBS(inst, fac(), time_limit=args.time_limit,
                            node_limit=args.node_limit).solve()
                    if not r.success:
                        ok = False
                        break
                    v, msg = validate(inst, r.paths); assert v, msg
                    e[name] = r.expansions
                if ok:
                    common += 1
                    for m in e:
                        rows[m].append(e[m])
            if common == 0:
                print(f"{dn:>7} {na:>6} {0:>6} | (none common)")
                continue
            cells = []
            card = rows["cardinal"]
            for m, _ in methods[1:]:
                rs = [rows[m][i] / max(1, card[i]) for i in range(common)]
                rs_arr = np.array(rs)
                gm = math.exp(np.mean(np.log(np.maximum(rs_arr, 1e-6))))
                cells.append(f"med{np.median(rs_arr):.2f}/gm{gm:.2f}/"
                             f"w{100*np.mean(rs_arr<1):.0f}%")
            print(f"{dn:>7} {na:>6} {common:>6} | " +
                  " ".join(f"{c:>22}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
