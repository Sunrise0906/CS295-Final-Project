"""Compare cardinal, learned-linear, learned-mlp (and optionally the strong
oracle as an upper bound) on held-out instances. Reports mean expansions over
instances solved by all methods, and the ratio vs cardinal.
"""
from __future__ import annotations

import argparse
import statistics as st

from mapf import random_instance, CBS, make_selector, validate
from mapf.strategies.learned import LearnedLinearSelector, LearnedMLPSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", default="models/selector_linear.npz")
    ap.add_argument("--mlp", default="models/selector_mlp.npz")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="8,9,10,11,12")
    ap.add_argument("--density", default="0.1")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--node-limit", type=int, default=5000)
    ap.add_argument("--with-oracle", action="store_true")
    args = ap.parse_args()

    lin = LearnedLinearSelector.load(args.linear)
    mlp = LearnedMLPSelector.load(args.mlp)

    methods = [("cardinal", lambda: make_selector("cardinal")),
               ("learned-linear", lambda: lin),
               ("learned-mlp", lambda: mlp)]
    if args.with_oracle:
        from mapf.strategies.oracle import StrongOracleSelector
        methods.append(("strong-oracle", lambda: StrongOracleSelector()))

    dens = float(args.density)
    agent_list = [int(x) for x in args.agents.split(",")]
    print(f"density={dens}  seeds={args.seeds}  (mean expansions; ratio vs cardinal)")
    print(f"{'agents':>6} {'common':>6} | " +
          " ".join(f"{m:>14}" for m, _ in methods))
    for na in agent_list:
        exps = {m: [] for m, _ in methods}
        common = 0
        for s in range(args.seeds):
            inst = random_instance(args.size, args.size, na, dens, seed=s)
            if inst is None:
                continue
            rows, ok = {}, True
            for name, fac in methods:
                r = CBS(inst, fac(), time_limit=args.time_limit,
                        node_limit=args.node_limit).solve()
                if not r.success:
                    ok = False
                    break
                v, msg = validate(inst, r.paths)
                assert v, msg
                rows[name] = r.expansions
            if ok:
                common += 1
                for m in rows:
                    exps[m].append(rows[m])
        card_mean = st.mean(exps["cardinal"]) if exps["cardinal"] else float("nan")
        cells = []
        for m, _ in methods:
            mean = st.mean(exps[m]) if exps[m] else float("nan")
            ratio = mean / card_mean if card_mean == card_mean else float("nan")
            cells.append(f"{mean:7.1f}({ratio:.2f})")
        print(f"{na:>6} {common:>6} | " + " ".join(f"{c:>14}" for c in cells),
              flush=True)


if __name__ == "__main__":
    main()
