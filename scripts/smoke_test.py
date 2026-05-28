"""Correctness smoke tests for the CBS core. Run: python -m scripts.smoke_test"""
from __future__ import annotations

import sys

from mapf import (
    Agent, GridMap, MAPFInstance, CBS, random_instance, validate, make_selector,
)
from mapf.conflicts import find_all_conflicts


def test_low_level_and_validate():
    # 3x3 open grid, single agent corner-to-corner.
    grid = GridMap(3, 3)
    inst = MAPFInstance(grid, [Agent(0, (0, 0), (2, 2))])
    res = CBS(inst, make_selector("first")).solve()
    assert res.success and res.cost == 4, res
    ok, msg = validate(inst, res.paths)
    assert ok, msg
    print("  [ok] low-level + single agent, cost=4")


def test_cardinal_conflict():
    # 3x3 grid: a horizontal and a vertical agent both forced through centre
    # (1,1) at t=1 -> that vertex conflict must be CARDINAL.
    grid = GridMap(3, 3)
    inst = MAPFInstance(grid, [
        Agent(0, (1, 0), (1, 2)),   # horizontal
        Agent(1, (0, 1), (2, 1)),   # vertical
    ])
    solver = CBS(inst, make_selector("cardinal"))
    root_paths = [solver.plan_agent(0, []), solver.plan_agent(1, [])]
    from mapf.cbs import CBSNode
    from mapf.core import sum_of_costs
    node = CBSNode([], root_paths, sum_of_costs(root_paths))
    node.conflicts = find_all_conflicts(root_paths)
    solver.classify(node)
    center = [c for c in node.conflicts if c.loc1 == (1, 1) and c.time == 1]
    assert center and center[0].cardinality == "cardinal", node.conflicts
    print(f"  [ok] centre conflict classified cardinal: {center[0]}")

    res = solver.solve()
    assert res.success, res
    ok, msg = validate(inst, res.paths)
    assert ok, msg
    print(f"  [ok] solved, cost={res.cost}, expansions={res.expansions}")


def test_optimality_invariant():
    # Every strategy must return the SAME optimal cost on the same instance;
    # only expansions/runtime may differ.
    strategies = ["first", "random", "earliest", "most-conflicts", "cardinal"]
    n_checked = 0
    for seed in range(40):
        inst = random_instance(8, 8, n_agents=5, obstacle_density=0.1, seed=seed)
        if inst is None:
            continue
        costs = {}
        exps = {}
        ok_all = True
        for s in strategies:
            res = CBS(inst, make_selector(s), time_limit=10.0).solve()
            if not res.success:
                ok_all = False
                break
            v, msg = validate(inst, res.paths)
            assert v, f"seed {seed} strat {s}: {msg}"
            costs[s] = res.cost
            exps[s] = res.expansions
        if not ok_all:
            continue
        assert len(set(costs.values())) == 1, f"seed {seed}: costs differ {costs}"
        n_checked += 1
        if n_checked <= 3:
            print(f"  [ok] seed {seed}: cost={list(costs.values())[0]}, "
                  f"expansions={exps}")
    print(f"  [ok] optimality invariant held on {n_checked} solved instances")
    assert n_checked >= 10


if __name__ == "__main__":
    print("test_low_level_and_validate"); test_low_level_and_validate()
    print("test_cardinal_conflict"); test_cardinal_conflict()
    print("test_optimality_invariant"); test_optimality_invariant()
    print("\nALL SMOKE TESTS PASSED")
    sys.exit(0)
