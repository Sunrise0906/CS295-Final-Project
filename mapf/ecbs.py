"""Bounded-suboptimal CBS via high-level focal search (ECBS).

With an optimal low level, a node's cost is a valid lower bound on any solution
under its constraints. The high level keeps OPEN ordered by cost and, at each
step, expands from the FOCAL set (nodes with cost <= w * min_cost) the node
chosen by a pluggable node selector. Solutions then have cost <= w * C*. The
node-ordering choice within FOCAL does not affect the suboptimality bound.
"""
from __future__ import annotations

import time
from typing import Optional

from .core import Conflict, MAPFInstance, sum_of_costs
from .conflicts import find_all_conflicts
from .cbs import CBS, CBSNode, CBSResult


class FewestConflicts:
    """Standard ECBS focal heuristic: expand the focal node with the fewest
    conflicts (tie-break by cost)."""
    name = "fewest-conflicts"

    def priority(self, node: CBSNode, solver) -> tuple:
        return (len(node.conflicts), node.cost)


class ECBS(CBS):
    def __init__(self, instance: MAPFInstance, conflict_selector, node_selector=None,
                 w: float = 1.5, time_limit: float = 30.0,
                 node_limit: Optional[int] = None, node_log: Optional[list] = None):
        super().__init__(instance, conflict_selector, time_limit, node_limit)
        self.w = w
        self.node_selector = node_selector or FewestConflicts()
        # If a list is given, every generated node is appended (with parent set)
        # so the solution path can be recovered for on-path labelling.
        self.node_log = node_log

    def solve(self) -> CBSResult:
        self._t0 = time.perf_counter()
        root_paths = []
        for i in range(self.instance.n_agents):
            p = self.plan_agent(i, [])
            if p is None:
                return self._fail("infeasible")
            root_paths.append(p)
        root = CBSNode([], root_paths, sum_of_costs(root_paths))
        root.conflicts = find_all_conflicts(root_paths)

        open_list = [root]          # all unexpanded nodes
        self.generated = 1
        if self.node_log is not None:
            self.node_log.append(root)

        while open_list:
            if time.perf_counter() - self._t0 > self.time_limit:
                return self._fail("timeout")
            if self.node_limit is not None and self.expansions >= self.node_limit:
                return self._fail("node_limit")

            lb_min = min(n.cost for n in open_list)
            thresh = self.w * lb_min
            focal = [n for n in open_list if n.cost <= thresh]
            node = min(focal, key=lambda n: self.node_selector.priority(n, self))
            open_list.remove(node)
            self.expansions += 1

            if not node.conflicts:
                self.solution_node = node
                return CBSResult(True, node.cost, node.paths, self.expansions,
                                 self.generated, time.perf_counter() - self._t0,
                                 "solved")

            conflict = self.conflict_select(node)
            for constraint in conflict.constraints():
                child = self._branch(node, constraint)
                if child is not None:
                    child.parent = node
                    open_list.append(child)
                    self.generated += 1
                    if self.node_log is not None:
                        self.node_log.append(child)

        return self._fail("infeasible")

    def conflict_select(self, node: CBSNode) -> Conflict:
        return self.selector.select(node, self)
