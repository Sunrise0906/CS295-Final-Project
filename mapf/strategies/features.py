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
