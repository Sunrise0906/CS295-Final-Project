"""Evaluate the sequence-model selector against cardinal and the MLP ranker on
held-out instances. Run IN mlenv (needs torch):

  D:/software/anaconda3/envs/mlenv/python.exe -m scripts.eval_seq
"""
from __future__ import annotations

import argparse
import statistics as st

from mapf import random_instance, CBS, make_selector, validate
from mapf.strategies.learned import LearnedMLPSelector
from mapf.strategies.sequence import SeqSelector


def run(inst, selector, track_history, tl, nl):
    res = CBS(inst, selector, time_limit=tl, node_limit=nl,
              track_history=track_history).solve()
    if res.success:
        ok, msg = validate(inst, res.paths)
        assert ok, msg
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-model", default="models/selector_seq.pt")
    ap.add_argument("--mlp-model", default="models/selector_mlp.npz")
    ap.add_argument("--linear-model", default="models/selector_linear.npz")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--agents", default="8,10,12")
    ap.add_argument("--density", default="0.1")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--time-limit", type=float, default=8.0)
    ap.add_argument("--node-limit", type=int, default=4000)
    args = ap.parse_args()

    import torch
    from mapf.strategies.learned import LearnedLinearSelector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = SeqSelector.load(args.seq_model, device=device)
    mlp = LearnedMLPSelector.load(args.mlp_model)
    lin = LearnedLinearSelector.load(args.linear_model)

    methods = [
        ("cardinal", lambda: make_selector("cardinal"), False),
        ("learned-linear", lambda: lin, False),
        ("learned-mlp", lambda: mlp, False),
        ("sequence", lambda: seq, True),
    ]
    agent_list = [int(x) for x in args.agents.split(",")]
    dens = float(args.density)

    print(f"device={device}  density={dens}")
    print(f"{'agents':>6} {'common':>7} | mean expansions (solved-by-all)")
    for na in agent_list:
        exps = {m: [] for m, _, _ in methods}
        common = 0
        for s in range(args.seeds):
            inst = random_instance(args.size, args.size, na, dens, seed=s)
            if inst is None:
                continue
            rows, ok = {}, True
            for name, factory, th in methods:
                r = run(inst, factory(), th, args.time_limit, args.node_limit)
                if not r.success:
                    ok = False
                    break
                rows[name] = r.expansions
            if ok:
                common += 1
                for m in rows:
                    exps[m].append(rows[m])
        means = " ".join(f"{m}={round(st.mean(exps[m]),1) if exps[m] else '-'}"
                         for m, _, _ in methods)
        print(f"{na:>6} {common:>4d}    | {means}", flush=True)


if __name__ == "__main__":
    main()
