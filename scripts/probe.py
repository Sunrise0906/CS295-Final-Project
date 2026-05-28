"""Quick probe: do strategies diverge as instances get harder?"""
import statistics as st
from mapf import random_instance, CBS, make_selector

STRATS = ["first", "random", "earliest", "cardinal"]
print(f"{'agents':>6} {'dens':>5} {'solved':>7} | mean expansions (solved-by-all)")
for na, dn in [(8, 0.1), (10, 0.1), (12, 0.1), (10, 0.15)]:
    exps = {s: [] for s in STRATS}
    solved = 0
    n_inst = 15
    for seed in range(n_inst):
        inst = random_instance(8, 8, na, dn, seed=2000 + seed)
        if inst is None:
            continue
        row, ok = {}, True
        for s in STRATS:
            r = CBS(inst, make_selector(s), time_limit=3.0, node_limit=3000).solve()
            if not r.success:
                ok = False
                break
            row[s] = r.expansions
        if ok:
            solved += 1
            for s in STRATS:
                exps[s].append(row[s])
    means = " ".join(f"{s}={round(st.mean(exps[s]),1) if exps[s] else '-'}" for s in STRATS)
    print(f"{na:>6} {dn:>5} {solved:>4}/{n_inst} | {means}", flush=True)
