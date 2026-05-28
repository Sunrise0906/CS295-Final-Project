"""Compare the strong subtree oracle to cardinal on main-tree expansions."""
import statistics as st
from mapf import random_instance, CBS, make_selector
from mapf.strategies.oracle import StrongOracleSelector, OracleSelector

CONFIGS = [(8, 0.1), (10, 0.1), (12, 0.1), (10, 0.15), (12, 0.15)]
print(f"{'agents':>6} {'dens':>5} {'n':>3} | "
      f"{'cardinal':>9} {'weak-orac':>9} {'STRONG':>9} | strong/card")
for na, dn in CONFIGS:
    card, weak, strong = [], [], []
    n = 0
    for seed in range(8):
        inst = random_instance(8, 8, na, dn, seed=5000 + seed)
        if inst is None:
            continue
        rc = CBS(inst, make_selector("cardinal"), time_limit=10, node_limit=4000).solve()
        rw = CBS(inst, OracleSelector(), time_limit=20, node_limit=4000).solve()
        # adequate subtree budget so the oracle's rollouts actually solve
        rs = CBS(inst, StrongOracleSelector(subtree_node_limit=2000,
                 subtree_time_limit=3.0), time_limit=120, node_limit=4000).solve()
        if rc.success and rw.success and rs.success:
            card.append(rc.expansions)
            weak.append(rw.expansions)
            strong.append(rs.expansions)
            n += 1
    if n:
        mc, mw, ms = st.mean(card), st.mean(weak), st.mean(strong)
        print(f"{na:>6} {dn:>5} {n:>3} | {mc:>9.1f} {mw:>9.1f} {ms:>9.1f} | "
              f"{ms/mc:>6.2f}", flush=True)
