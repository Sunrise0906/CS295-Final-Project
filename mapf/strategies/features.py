"""Per-conflict feature extraction for the learned selectors.

Features for one conflict at a CBS node: conflict type, timing, MDD widths and
cardinality of both agents, path costs, remaining distances, and conflict
degree. The same fixed-length vector feeds the linear ranker, the MLP, and the
sequence model.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..core import Conflict, pos_at

FEATURE_NAMES = [
    "is_vertex", "is_edge",
    "t_norm", "t_frac_makespan",
    "w_i_tm1", "w_i_t", "w_i_tp1",
    "w_j_tm1", "w_j_t", "w_j_tp1",
    "singleton_i", "singleton_j",
    "min_w_t", "prod_w_t_norm",
    "card_cardinal", "card_semi", "card_non",
    "cost_i_norm", "cost_j_norm",
    "h_i_norm", "h_j_norm",
    "deg_i_norm", "deg_j_norm",
    "n_conflicts_norm",
]
N_FEATURES = len(FEATURE_NAMES)

_CARD_ONEHOT = {
    "cardinal": (1.0, 0.0, 0.0),
    "semi": (0.0, 1.0, 0.0),
    "non": (0.0, 0.0, 1.0),
}


def extract_node_features(node, solver) -> tuple[np.ndarray, list[Conflict]]:
    """Return an ``(n_conflicts, N_FEATURES)`` matrix and the matching conflict
    list for one CBS node. Builds/uses cached MDDs and cardinality."""
    solver.classify(node)  # fills cardinality, populates MDD cache
    conflicts = node.conflicts
    makespan = max((len(p) for p in node.paths), default=1)
    n_pairs = max(1, len(conflicts))

    deg = Counter()
    for c in conflicts:
        deg[c.a1] += 1
        deg[c.a2] += 1

    rows = []
    for c in conflicts:
        mi = solver.mdd_for(node, c.a1)
        mj = solver.mdd_for(node, c.a2)
        t = c.time
        wi = (mi.width(t - 1), mi.width(t), mi.width(t + 1)) if mi else (1, 1, 1)
        wj = (mj.width(t - 1), mj.width(t), mj.width(t + 1)) if mj else (1, 1, 1)
        ci = _CARD_ONEHOT.get(c.cardinality, (0.0, 0.0, 1.0))

        from ..core import path_cost
        cost_i = path_cost(node.paths[c.a1])
        cost_j = path_cost(node.paths[c.a2])
        # remaining grid distance from the conflict cell to each goal
        loc = c.loc1
        h_i = solver.heuristics[c.a1].get(loc, makespan)
        h_j = solver.heuristics[c.a2].get(loc, makespan)

        ms = float(max(1, makespan))
        rows.append([
            1.0 if c.kind == "vertex" else 0.0,
            1.0 if c.kind == "edge" else 0.0,
            t / ms,
            t / ms,
            wi[0], wi[1], wi[2],
            wj[0], wj[1], wj[2],
            1.0 if wi[1] == 1 else 0.0,
            1.0 if wj[1] == 1 else 0.0,
            float(min(wi[1], wj[1])),
            (wi[1] * wj[1]) / 100.0,
            ci[0], ci[1], ci[2],
            cost_i / ms, cost_j / ms,
            h_i / ms, h_j / ms,
            deg[c.a1] / n_pairs, deg[c.a2] / n_pairs,
            len(conflicts) / 50.0,
        ])
    return np.asarray(rows, dtype=np.float64), conflicts


# --- Extended features (v2): adds MDD-structure and conflict-graph features --

EXT_FEATURE_NAMES = FEATURE_NAMES + [
    "mdd_overlap_t",
    "mdd_overlap_global_norm",
    "cg_deg_i",
    "cg_deg_j",
    "cg_deg_max",
    "mdd_width_avg_i",
    "mdd_width_avg_j",
    "path_overlap_norm",
    "n_singleton_levels_norm",
]
N_EXT_FEATURES = len(EXT_FEATURE_NAMES)


def _mdd_level(m, t):
    """Cells that agent could occupy at time ``t`` according to MDD ``m``.
    For ``t`` past arrival the agent is pinned to the goal."""
    if m is None:
        return None
    if t < 0:
        return set()
    if t >= m.cost:
        return {m.goal}
    if t >= len(m.levels):
        return {m.goal}
    return m.levels[t]


def _mdd_overlap_at(mi, mj, t):
    si, sj = _mdd_level(mi, t), _mdd_level(mj, t)
    if si is None or sj is None:
        return 0
    return len(si & sj)


def _mdd_overlap_global(mi, mj):
    if mi is None or mj is None:
        return 0
    total = 0
    for t in range(max(mi.cost, mj.cost) + 1):
        total += _mdd_overlap_at(mi, mj, t)
    return total


def _mdd_width_avg(m):
    if m is None or not m.levels:
        return 1.0
    return sum(len(lvl) for lvl in m.levels) / len(m.levels)


def _n_singleton_levels(m):
    if m is None or not m.levels:
        return 0
    return sum(1 for lvl in m.levels if len(lvl) == 1)


def extract_node_features_ext(node, solver):
    """Extended features: base 24-dim vector plus MDD structure features
    (cross-agent MDD overlap, average widths, singleton level counts, path
    overlap) and conflict-graph degree features."""
    X, conflicts = extract_node_features(node, solver)
    n_agents = solver.instance.n_agents

    # Conflict-graph: for each agent, the set of other agents it conflicts with.
    neighbors: dict = {}
    for c in conflicts:
        neighbors.setdefault(c.a1, set()).add(c.a2)
        neighbors.setdefault(c.a2, set()).add(c.a1)

    extras = []
    makespan = max((len(p) for p in node.paths), default=1)
    ms = float(max(1, makespan))
    for c in conflicts:
        mi = solver.mdd_for(node, c.a1)
        mj = solver.mdd_for(node, c.a2)
        t = c.time
        ov_t = _mdd_overlap_at(mi, mj, t)
        ov_g = _mdd_overlap_global(mi, mj)
        cg_i = len(neighbors.get(c.a1, set()))
        cg_j = len(neighbors.get(c.a2, set()))
        wa_i = _mdd_width_avg(mi)
        wa_j = _mdd_width_avg(mj)
        # Path overlap: cells visited by both agents' current paths (ignoring time).
        pi = set(node.paths[c.a1])
        pj = set(node.paths[c.a2])
        po = len(pi & pj)
        ns = _n_singleton_levels(mi) + _n_singleton_levels(mj)
        extras.append([
            ov_t,
            ov_g / ms,
            cg_i / max(1.0, n_agents - 1),
            cg_j / max(1.0, n_agents - 1),
            max(cg_i, cg_j) / max(1.0, n_agents - 1),
            wa_i,
            wa_j,
            po / ms,
            ns / max(1.0, 2 * ms),
        ])
    return np.hstack([X, np.asarray(extras, dtype=np.float64)]), conflicts
