"""Learned focal node ordering for ECBS.

In bounded-suboptimal CBS the focal set offers a free choice of which node to
expand. The standard heuristic orders by fewest conflicts. We instead learn a
classifier that predicts whether a node lies on the path to a solution, from
cheap node-level features (no MDDs in the hot loop), and expand the highest-
scoring node first.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

FOCAL_FEATURE_NAMES = [
    "n_conflicts", "n_vertex", "n_edge",
    "cost", "depth",
    "n_agents_in_conflict", "max_agent_degree",
    "mean_conf_time", "min_conf_time",
    "conf_per_agent",
]
N_FOCAL_FEATURES = len(FOCAL_FEATURE_NAMES)


def focal_features(node, n_agents: int) -> np.ndarray:
    confs = node.conflicts
    nc = len(confs)
    n_vertex = sum(1 for c in confs if c.kind == "vertex")
    n_edge = nc - n_vertex
    deg = Counter()
    times = []
    for c in confs:
        deg[c.a1] += 1
        deg[c.a2] += 1
        times.append(c.time)
    makespan = max((len(p) for p in node.paths), default=1)
    mean_t = (sum(times) / len(times) / makespan) if times else 0.0
    min_t = (min(times) / makespan) if times else 0.0
    return np.array([
        nc,
        n_vertex,
        n_edge,
        node.cost,
        len(node.constraints),
        len(deg),
        max(deg.values()) if deg else 0,
        mean_t,
        min_t,
        nc / max(1, n_agents),
    ], dtype=np.float64)


class LearnedFocalSelector:
    """Focal ordering by predicted P(node on solution path); expand highest P."""
    name = "learned-focal"

    def __init__(self, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray,
                 n_agents: int):
        self.w = w
        self.b = b
        self.mean = mean
        self.std = np.where(std < 1e-8, 1.0, std)
        self.n_agents = n_agents

    @classmethod
    def load(cls, path: str, n_agents: int) -> "LearnedFocalSelector":
        d = np.load(path)
        return cls(d["w"], float(d["b"]), d["mean"], d["std"], n_agents)

    def priority(self, node, solver) -> tuple:
        f = (focal_features(node, self.n_agents) - self.mean) / self.std
        score = float(f @ self.w + self.b)        # logit of P(on-path)
        return (-score, node.cost)                 # higher P first, tie-break cost


class BlendedFocalSelector(LearnedFocalSelector):
    """Robust focal ordering: fewest-conflicts primary (stable), learned
    on-path score as a soft adjustment. priority = n_conflicts - lam * P(on-path).
    With small lam the learned model only reorders near-ties, bounding downside."""
    name = "blended-focal"

    def __init__(self, *args, lam: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam

    @classmethod
    def load(cls, path, n_agents, lam: float = 1.0):
        d = __import__("numpy").load(path)
        return cls(d["w"], float(d["b"]), d["mean"], d["std"], n_agents, lam=lam)

    def priority(self, node, solver) -> tuple:
        f = (focal_features(node, self.n_agents) - self.mean) / self.std
        z = float(f @ self.w + self.b)
        p = 1.0 / (1.0 + 2.718281828 ** (-z))      # P(on-path) in [0,1]
        return (len(node.conflicts) - self.lam * p, node.cost)
