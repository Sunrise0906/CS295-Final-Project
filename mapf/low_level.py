"""Low-level single-agent planner: space-time A* respecting CBS constraints."""
from __future__ import annotations

import heapq
from collections import deque
from typing import Optional

from .core import Cell, GridMap, VertexConstraint, EdgeConstraint


def backward_bfs(grid: GridMap, goal: Cell) -> dict[Cell, int]:
    """Shortest grid distance from every reachable cell to ``goal`` (the
    admissible, obstacle-aware low-level heuristic)."""
    dist = {goal: 0}
    q = deque([goal])
    while q:
        cur = q.popleft()
        for nb in grid.neighbors(cur):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


class ConstraintTable:
    """Per-agent view of a node's constraints, for fast lookup in A*."""

    def __init__(self, agent: int, constraints: list):
        self.vertex: set[tuple[Cell, int]] = set()
        self.edge: set[tuple[Cell, Cell, int]] = set()
        # Latest timestep at which the agent's goal is blocked, so the planner
        # knows it cannot "settle" before then.
        self.max_time = 0
        for c in constraints:
            if c.agent != agent:
                continue
            if isinstance(c, VertexConstraint):
                self.vertex.add((c.cell, c.time))
                self.max_time = max(self.max_time, c.time)
            else:  # EdgeConstraint
                self.edge.add((c.frm, c.to, c.time))
                self.max_time = max(self.max_time, c.time)

    def vertex_blocked(self, cell: Cell, t: int) -> bool:
        return (cell, t) in self.vertex

    def edge_blocked(self, frm: Cell, to: Cell, t: int) -> bool:
        return (frm, to, t) in self.edge

    def goal_blocked_after(self, goal: Cell) -> int:
        """Latest time the goal cell itself is vertex-constrained."""
        latest = 0
        for (cell, t) in self.vertex:
            if cell == goal:
                latest = max(latest, t)
        return latest


def astar(
    grid: GridMap,
    start: Cell,
    goal: Cell,
    h: dict[Cell, int],
    table: ConstraintTable,
    time_horizon: Optional[int] = None,
) -> Optional[list[Cell]]:
    """Min-cost (sum-of-costs) space-time path from ``start`` to ``goal``.

    Returns ``path[t] = cell`` (trailing waits trimmed to arrival), or None if
    no constraint-respecting path exists within the time horizon.
    """
    if start not in h:  # goal unreachable on the static grid
        return None

    goal_release = table.goal_blocked_after(goal)
    if time_horizon is None:
        # Generous but finite bound: enough room to detour around every
        # constraint. Safe for the small/medium instances we study.
        time_horizon = grid.n_free + table.max_time + 1

    # State = (cell, t). g = t. f = t + h[cell].
    start_state = (start, 0)
    open_heap = [(h[start], 0, start, 0)]  # (f, g, cell, t)
    came_from: dict[tuple[Cell, int], tuple[Cell, int]] = {}
    best_g: dict[tuple[Cell, int], int] = {start_state: 0}

    while open_heap:
        f, g, cell, t = heapq.heappop(open_heap)
        if g > best_g.get((cell, t), g):
            continue

        # Goal test: at goal, and past any constraint that would force it to move.
        if cell == goal and t >= goal_release:
            return _reconstruct(came_from, (cell, t), start)

        if t >= time_horizon:
            continue

        # Successors: 4 moves + wait.
        for nb in grid.neighbors(cell) + [cell]:
            nt = t + 1
            if table.vertex_blocked(nb, nt):
                continue
            if nb != cell and table.edge_blocked(cell, nb, nt):
                continue
            if nb not in h:  # cannot reach goal from here
                continue
            ns = (nb, nt)
            ng = nt
            if ng < best_g.get(ns, 1 << 30):
                best_g[ns] = ng
                came_from[ns] = (cell, t)
                heapq.heappush(open_heap, (ng + h[nb], ng, nb, nt))

    return None


def _reconstruct(came_from, end_state, start) -> list[Cell]:
    path = []
    s = end_state
    while s in came_from:
        path.append(s[0])
        s = came_from[s]
    path.append(start)
    path.reverse()
    return path
