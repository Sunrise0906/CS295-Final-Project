"""One-step look-ahead proxy oracle for conflict selection, with data logging.

The optimal oracle (the conflict yielding the smallest CT subtree) is too
expensive to run online. This proxy oracle branches once per candidate and
scores the resulting children lexicographically by:

    (1) #children pruned as infeasible  (higher is better)
    (2) minimum cost increase           (higher is better; cardinal-like)
    (3) total residual conflicts        (lower is better)
"""
from __future__ import annotations

from .features import extract_node_features


def oracle_choice_index(node, solver) -> int:
    """Index into ``node.conflicts`` of the oracle's preferred conflict."""
    base = node.cost
    best_idx, best_key = 0, None
    for idx, c in enumerate(node.conflicts):
        children = [solver._branch(node, con) for con in c.constraints()]
        pruned = sum(1 for ch in children if ch is None)
        alive = [ch for ch in children if ch is not None]
        if alive:
            dcost = min(ch.cost - base for ch in alive)
            resid = sum(len(ch.conflicts) for ch in alive)
        else:
            dcost, resid = 10**9, 0
        key = (pruned, dcost, -resid)
        if best_key is None or key > best_key:
            best_key, best_idx = key, idx
    return best_idx


class OracleSelector:
    """Selector that follows the proxy oracle and (optionally) logs training
    tuples for each expanded node.

    ``log``     -- if given, append ``(X, chosen_idx)`` (memoryless ranker data).
    ``seq_log`` -- if given, append ``(history, X, chosen_idx)`` where ``history``
                   is the node's root-to-node feature sequence (sequence-model
                   data); requires the solver to run with ``track_history=True``.
    After each call ``self.chosen_feature`` holds the chosen conflict's feature
    row, which CBS uses to extend child histories.
    """
    name = "oracle"

    def __init__(self, log: list | None = None, seq_log: list | None = None):
        self.log = log
        self.seq_log = seq_log
        self.chosen_feature = None

    def select(self, node, solver):
        if self.log is not None or self.seq_log is not None:
            X, conflicts = extract_node_features(node, solver)
            idx = oracle_choice_index(node, solver)
            self.chosen_feature = X[idx]
            if self.log is not None:
                self.log.append((X, idx))
            if self.seq_log is not None:
                self.seq_log.append((list(node.history), X, idx))
            return conflicts[idx]
        idx = oracle_choice_index(node, solver)
        # Still expose the chosen feature when history tracking is on.
        if solver.track_history:
            X, conflicts = extract_node_features(node, solver)
            self.chosen_feature = X[idx]
            return conflicts[idx]
        return node.conflicts[idx]


# --- Strong oracle: one-step policy improvement over cardinal ----------------

def _cost_of_choosing(solver, node, conflict, node_limit, time_limit) -> int:
    """Q(node, conflict): expansions to solve node's subtree *as a whole* (best-
    first, early-stop) when we branch on `conflict` first and roll out with
    cardinal afterwards. This respects best-first's early termination and is the
    correct objective for optimal CBS (summing isolated subtrees is not)."""
    from ..cbs import CBS
    from .hardcoded import CardinalSelector
    children = [solver._branch(node, con) for con in conflict.constraints()]
    children = [ch for ch in children if ch is not None]
    if not children:
        return 0  # both branches infeasible (node already unsolvable)
    sub = CBS(solver.instance, CardinalSelector(), time_limit=time_limit,
              node_limit=node_limit)
    r = sub.solve_from_open(children)
    return r.expansions if r.success else node_limit + 1


def strong_oracle_choice_index(node, solver, subtree_node_limit=400,
                               subtree_time_limit=1.0) -> int:
    """One-step policy improvement over cardinal: pick the conflict whose
    subtree (explored as a whole with cardinal afterwards) is cheapest. Since
    cardinal's own choice is among the candidates, this is provably <= cardinal
    on each node, hence <= cardinal overall."""
    best_idx, best_cost = 0, None
    for idx, c in enumerate(node.conflicts):
        cost = _cost_of_choosing(solver, node, c, subtree_node_limit, subtree_time_limit)
        if best_cost is None or cost < best_cost:
            best_cost, best_idx = cost, idx
    return best_idx


class StrongOracleSelector:
    """Selector following the subtree-minimizing (policy-improvement) oracle,
    with optional logging of memoryless / sequence training tuples."""
    name = "strong-oracle"

    def __init__(self, log=None, seq_log=None, subtree_node_limit=1000,
                 subtree_time_limit=2.0):
        self.log = log
        self.seq_log = seq_log
        self.subtree_node_limit = subtree_node_limit
        self.subtree_time_limit = subtree_time_limit
        self.chosen_feature = None

    def select(self, node, solver):
        idx = strong_oracle_choice_index(
            node, solver, self.subtree_node_limit, self.subtree_time_limit)
        if self.log is not None or self.seq_log is not None or solver.track_history:
            X, conflicts = extract_node_features(node, solver)
            self.chosen_feature = X[idx]
            if self.log is not None:
                self.log.append((X, idx))
            if self.seq_log is not None:
                self.seq_log.append((list(node.history), X, idx))
            return conflicts[idx]
        return node.conflicts[idx]
