"""High-level Conflict-Based Search with a pluggable conflict selector.

The solver is optimal for sum-of-costs (best-first on node cost), so the
conflict-selection strategy changes only the search effort (node expansions /
runtime), not the returned solution cost. Strategies receive the current node
and the solver itself, so they can query MDDs / cardinality on demand.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional

from .core import (
    Agent,
    Cell,
    Conflict,
    Constraint,
    MAPFInstance,
    path_cost,
    sum_of_costs,
)
from .conflicts import find_all_conflicts
from .low_level import ConstraintTable, astar, backward_bfs
from .mdd import MDD, build_mdd, classify


@dataclass
class CBSNode:
    constraints: list
    paths: list[list[Cell]]
    cost: int
    conflicts: list[Conflict] = field(default_factory=list)
    classified: bool = False
    # Sequence of feature vectors of the conflicts resolved along the
    # root-to-this-node path (only populated when CBS.track_history is set).
    history: list = field(default_factory=list)
    # Parent node (set by ECBS when node_log is enabled, for on-path labelling).
    parent: object = field(default=None, repr=False, compare=False)

    def agent_constraints(self, agent: int) -> tuple:
        return tuple(c for c in self.constraints if c.agent == agent)


@dataclass
class CBSResult:
    success: bool
    cost: Optional[int]
    paths: Optional[list[list[Cell]]]
    expansions: int          # high-level nodes expanded (popped & processed)
    generated: int           # high-level nodes generated (pushed)
    runtime: float           # wall-clock seconds
    reason: str = ""         # 'solved' | 'timeout' | 'node_limit' | 'infeasible'


class CBS:
    def __init__(
        self,
        instance: MAPFInstance,
        selector,
        time_limit: float = 30.0,
        node_limit: Optional[int] = None,
        track_history: bool = False,
        init_constraints: Optional[list] = None,
        bypass: bool = False,
    ):
        self.instance = instance
        self.grid = instance.grid
        self.selector = selector
        self.time_limit = time_limit
        self.node_limit = node_limit
        # When set, child nodes inherit the parent's history extended by the
        # feature vector of the conflict selected here (for sequence models).
        self.track_history = track_history
        # Extra constraints to seed the root with (used for subtree rollouts
        # that estimate "how hard is it to finish from here").
        self.init_constraints = list(init_constraints) if init_constraints else []
        # If True, before branching on a conflict try to replace one agent's
        # path with a same-cost path that avoids the conflicting move (ICBS
        # Bypass). If found, update the node in-place instead of branching.
        self.bypass = bypass

        # Static low-level heuristics: backward BFS distance-to-goal per agent.
        self.heuristics = [backward_bfs(self.grid, a.goal) for a in instance.agents]

        self.expansions = 0
        self.generated = 0
        self._mdd_cache: dict = {}
        self._t0 = 0.0

    # --- low level ---------------------------------------------------------
    def plan_agent(self, agent: int, constraints: list) -> Optional[list[Cell]]:
        a = self.instance.agents[agent]
        table = ConstraintTable(agent, constraints)
        return astar(self.grid, a.start, a.goal, self.heuristics[agent], table)

    # --- MDD / classification (used by strategies) -------------------------
    def mdd_for(self, node: CBSNode, agent: int) -> Optional[MDD]:
        cost = path_cost(node.paths[agent])
        key = (agent, cost, node.agent_constraints(agent))
        if key in self._mdd_cache:
            return self._mdd_cache[key]
        a = self.instance.agents[agent]
        table = ConstraintTable(agent, node.constraints)
        mdd = build_mdd(self.grid, a.start, a.goal, self.heuristics[agent], table, cost)
        self._mdd_cache[key] = mdd
        return mdd

    def classify(self, node: CBSNode) -> None:
        """Annotate every conflict in the node with its cardinality."""
        if node.classified:
            return
        for c in node.conflicts:
            mi = self.mdd_for(node, c.a1)
            mj = self.mdd_for(node, c.a2)
            c.cardinality = classify(c, mi, mj)
        node.classified = True

    # --- high level --------------------------------------------------------
    def solve(self) -> CBSResult:
        self._t0 = time.perf_counter()

        root_paths = []
        for i in range(self.instance.n_agents):
            p = self.plan_agent(i, self.init_constraints)
            if p is None:
                return CBSResult(False, None, None, 0, 0,
                                 time.perf_counter() - self._t0, "infeasible")
            root_paths.append(p)

        root = CBSNode(constraints=list(self.init_constraints), paths=root_paths,
                       cost=sum_of_costs(root_paths))
        root.conflicts = find_all_conflicts(root_paths)
        return self._run(root)

    def solve_seeded(self, start: CBSNode) -> CBSResult:
        """Run the high-level search starting from an existing node (inheriting
        its paths/conflicts). Used to measure the true subtree size under a node
        without replanning every agent from scratch."""
        self._t0 = time.perf_counter()
        return self._run_open([start])

    def solve_from_open(self, nodes: list) -> CBSResult:
        """Run best-first search seeded with several open nodes at once. Used to
        measure the cost of committing to a particular first branching choice:
        the subtree is then explored as a whole (best-first, early-stop), which
        is the correct objective for optimal CBS."""
        self._t0 = time.perf_counter()
        return self._run_open([n for n in nodes if n is not None])

    def _run(self, root: CBSNode) -> CBSResult:
        return self._run_open([root])

    def _run_open(self, seeds: list) -> CBSResult:
        counter = 0
        open_list = []
        for n in seeds:
            heapq.heappush(open_list, (n.cost, len(n.conflicts), counter, n))
            counter += 1
        self.generated = len(open_list)
        if not open_list:
            return self._fail("infeasible")

        while open_list:
            if time.perf_counter() - self._t0 > self.time_limit:
                return self._fail("timeout")
            if self.node_limit is not None and self.expansions >= self.node_limit:
                return self._fail("node_limit")

            _, _, _, node = heapq.heappop(open_list)
            self.expansions += 1

            if not node.conflicts:
                return CBSResult(
                    True, node.cost, node.paths, self.expansions, self.generated,
                    time.perf_counter() - self._t0, "solved",
                )

            conflict = self.selector.select(node, self)
            chosen_feat = getattr(self.selector, "chosen_feature", None) \
                if self.track_history else None

            # Bypass: try to replace one agent's path with a same-cost path
            # that avoids the conflicting move. If found, update the node in
            # place (no branching), push back, continue.
            if self.bypass and self._try_bypass(node, conflict):
                # Recompute conflicts and re-push (cost unchanged, so f-value
                # unchanged; ranks identically in the open list).
                counter += 1
                heapq.heappush(
                    open_list, (node.cost, len(node.conflicts), counter, node))
                continue

            for constraint in conflict.constraints():
                child = self._branch(node, constraint, chosen_feat)
                if child is None:
                    continue
                counter += 1
                heapq.heappush(
                    open_list, (child.cost, len(child.conflicts), counter, child)
                )
                self.generated += 1

        return self._fail("infeasible")

    def _try_bypass(self, node: CBSNode, conflict) -> bool:
        """Attempt an ICBS-style bypass: replace one of the two agents' paths
        with a same-cost path that avoids the conflicting move. If successful,
        ``node.paths`` and ``node.conflicts`` are updated in place and we
        return True; otherwise False."""
        from .core import path_cost, sum_of_costs
        # The two child constraints already isolate the conflicting move.
        for con in conflict.constraints():
            agent = con.agent
            old_cost = path_cost(node.paths[agent])
            new_path = self.plan_agent(agent, node.constraints + [con])
            if new_path is None:
                continue
            if path_cost(new_path) > old_cost:
                continue
            # Same-cost alternative exists. Replace the path in place.
            new_paths = list(node.paths)
            new_paths[agent] = new_path
            new_conflicts = find_all_conflicts(new_paths)
            # Only accept the bypass if it reduces the number of conflicts (the
            # standard ICBS criterion). Otherwise it is no progress.
            if len(new_conflicts) < len(node.conflicts):
                node.paths = new_paths
                node.conflicts = new_conflicts
                node.cost = sum_of_costs(new_paths)
                node.classified = False
                return True
        return False

    def _branch(self, node: CBSNode, constraint: Constraint,
                chosen_feat=None) -> Optional[CBSNode]:
        agent = constraint.agent
        new_constraints = node.constraints + [constraint]
        new_path = self.plan_agent(agent, new_constraints)
        if new_path is None:
            return None
        new_paths = list(node.paths)
        new_paths[agent] = new_path
        child = CBSNode(new_constraints, new_paths, sum_of_costs(new_paths))
        child.conflicts = find_all_conflicts(new_paths)
        if self.track_history and chosen_feat is not None:
            child.history = node.history + [chosen_feat]
        return child

    def _fail(self, reason: str) -> CBSResult:
        return CBSResult(False, None, None, self.expansions, self.generated,
                         time.perf_counter() - self._t0, reason)
