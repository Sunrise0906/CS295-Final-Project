"""Multi-valued Decision Diagrams (MDDs) and conflict cardinality.

An agent's MDD for cost ``c`` is the union of all constraint-respecting paths of
exactly cost ``c`` from start to goal. The *width* of the MDD at level ``t`` is
the number of cells the agent could occupy at time ``t`` on some optimal path.

Conflict cardinality (Boyarski et al., ICBS 2015):
  * cardinal      -- resolving it increases cost on *both* child nodes,
  * semi-cardinal -- it increases cost on exactly one child,
  * non-cardinal  -- it increases cost on neither.
A vertex conflict at (v, t) is cardinal iff both agents have MDD width 1 at
level t (both are *forced* through v). Prioritising cardinal conflicts is the
classic hand-crafted heuristic that makes CBS practical.
"""
from __future__ import annotations

from typing import Optional

from .core import Cell, Conflict, GridMap
from .low_level import ConstraintTable


class MDD:
    def __init__(self, levels: list[set[Cell]], goal: Cell, cost: int):
        self.levels = levels  # levels[t] = set of cells at time t
        self.goal = goal
        self.cost = cost

    def width(self, t: int) -> int:
        """Number of cells reachable at time ``t``. Past arrival the agent is
        pinned to its goal, so width is 1."""
        if t >= self.cost:
            return 1
        if t < 0 or t >= len(self.levels):
            return 1
        return len(self.levels[t])


def build_mdd(
    grid: GridMap,
    start: Cell,
    goal: Cell,
    h: dict[Cell, int],
    table: ConstraintTable,
    cost: int,
) -> Optional[MDD]:
    """Build the constraint-respecting MDD of exact cost ``cost``."""
    if cost < 0 or start not in h or h[start] > cost:
        return None
    if table.vertex_blocked(start, 0):
        return None

    # Forward pass with reachability pruning (goal must stay reachable in time).
    fwd = [set() for _ in range(cost + 1)]
    fwd[0].add(start)
    for t in range(cost):
        nt = t + 1
        for v in fwd[t]:
            for w in grid.neighbors(v) + [v]:
                if table.vertex_blocked(w, nt):
                    continue
                if w != v and table.edge_blocked(v, w, nt):
                    continue
                if w in h and h[w] <= cost - nt:
                    fwd[nt].add(w)
    if goal not in fwd[cost]:
        return None

    # Backward prune: keep a node only if it links to a kept successor.
    keep = [set() for _ in range(cost + 1)]
    keep[cost].add(goal)
    for t in range(cost - 1, -1, -1):
        nt = t + 1
        for v in fwd[t]:
            for w in grid.neighbors(v) + [v]:
                if w not in keep[nt]:
                    continue
                if table.vertex_blocked(w, nt):
                    continue
                if w != v and table.edge_blocked(v, w, nt):
                    continue
                keep[t].add(v)
                break
    return MDD(keep, goal, cost)


def classify(conflict: Conflict, mdd_i: Optional[MDD], mdd_j: Optional[MDD]) -> str:
    """Return 'cardinal' | 'semi' | 'non' for a conflict given both MDDs.

    If an MDD is unavailable, fall back to 'non' (conservative)."""
    t = conflict.time
    if mdd_i is None or mdd_j is None:
        return "non"
    if conflict.kind == "vertex":
        forced_i = mdd_i.width(t) == 1
        forced_j = mdd_j.width(t) == 1
    else:  # edge: forced through the swap means width 1 at both endpoints
        forced_i = mdd_i.width(t - 1) == 1 and mdd_i.width(t) == 1
        forced_j = mdd_j.width(t - 1) == 1 and mdd_j.width(t) == 1
    if forced_i and forced_j:
        return "cardinal"
    if forced_i or forced_j:
        return "semi"
    return "non"
