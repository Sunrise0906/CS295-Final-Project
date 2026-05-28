"""Core data structures for Multi-Agent Path Finding (MAPF).

Cells are (row, col) tuples. Time is an integer index; a *path* is a list of
cells where ``path[t]`` is the agent's cell at timestep ``t``. The cost model is
sum-of-costs (SOC): the cost of one agent's path is the timestep at which it
arrives at its goal and stays there, i.e. ``len(path) - 1`` with any trailing
wait-at-goal steps trimmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

Cell = tuple[int, int]

# 4-connected grid moves (no diagonal). Waiting is handled separately.
MOVES: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


class GridMap:
    """A 4-connected grid with blocked/free cells."""

    def __init__(self, height: int, width: int, obstacles: Optional[set[Cell]] = None):
        self.height = height
        self.width = width
        self.passable = np.ones((height, width), dtype=bool)
        if obstacles:
            for (r, c) in obstacles:
                self.passable[r, c] = False

    def is_free(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width and bool(self.passable[r, c])

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Free 4-connected neighbours (excludes the wait/self move)."""
        r, c = cell
        out = []
        for dr, dc in MOVES:
            nb = (r + dr, c + dc)
            if self.is_free(nb):
                out.append(nb)
        return out

    @property
    def n_free(self) -> int:
        return int(self.passable.sum())


@dataclass
class Agent:
    id: int
    start: Cell
    goal: Cell


@dataclass
class MAPFInstance:
    grid: GridMap
    agents: list[Agent]

    @property
    def n_agents(self) -> int:
        return len(self.agents)


# --- Constraints (low-level) -------------------------------------------------

@dataclass(frozen=True)
class VertexConstraint:
    """Agent may not occupy ``cell`` at ``time``."""
    agent: int
    cell: Cell
    time: int


@dataclass(frozen=True)
class EdgeConstraint:
    """Agent may not traverse ``frm -> to`` arriving at ``time``."""
    agent: int
    frm: Cell
    to: Cell
    time: int


Constraint = "VertexConstraint | EdgeConstraint"


# --- Conflicts (high-level) --------------------------------------------------

@dataclass
class Conflict:
    """A conflict between two agents in a joint solution.

    ``vertex``: both agents occupy ``loc1`` at ``time`` (``loc2`` is None).
    ``edge``:   ``a1`` moves loc1->loc2 while ``a2`` moves loc2->loc1, arriving
                at ``time`` (a head-on swap).
    """
    a1: int
    a2: int
    kind: str  # 'vertex' | 'edge'
    loc1: Cell
    loc2: Optional[Cell]
    time: int
    # Filled in by classification (None until computed).
    cardinality: Optional[str] = field(default=None)  # 'cardinal'|'semi'|'non'

    def constraints(self) -> tuple[Constraint, Constraint]:
        """The two child constraints this conflict splits into."""
        if self.kind == "vertex":
            return (
                VertexConstraint(self.a1, self.loc1, self.time),
                VertexConstraint(self.a2, self.loc1, self.time),
            )
        else:
            return (
                EdgeConstraint(self.a1, self.loc1, self.loc2, self.time),
                EdgeConstraint(self.a2, self.loc2, self.loc1, self.time),
            )

    def __repr__(self) -> str:
        if self.kind == "vertex":
            return f"V(a{self.a1},a{self.a2}@{self.loc1},t{self.time},{self.cardinality})"
        return f"E(a{self.a1},a{self.a2}:{self.loc1}<->{self.loc2},t{self.time},{self.cardinality})"


def path_cost(path: list[Cell]) -> int:
    """SOC cost of a single path: timestep of final (non-trailing-wait) move."""
    if not path:
        return 0
    last = len(path) - 1
    goal = path[-1]
    while last > 0 and path[last - 1] == goal:
        last -= 1
    return last


def sum_of_costs(paths: list[list[Cell]]) -> int:
    return sum(path_cost(p) for p in paths)


def pos_at(path: list[Cell], t: int) -> Cell:
    """Position of an agent at time ``t`` (stays at goal after arrival)."""
    if t < len(path):
        return path[t]
    return path[-1]
