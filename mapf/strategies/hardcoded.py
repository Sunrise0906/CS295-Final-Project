"""Hand-crafted conflict-selection strategies.

Each strategy exposes ``select(node, solver) -> Conflict``. ``solver`` is the CBS
instance, used by classification-based strategies to query MDDs/cardinality.
"""
from __future__ import annotations

import random
from collections import Counter

from ..core import Conflict


class ConflictSelector:
    name = "base"

    def select(self, node, solver) -> Conflict:
        raise NotImplementedError


class FirstSelector(ConflictSelector):
    """Naive CBS: the first conflict found (arbitrary but deterministic)."""
    name = "first"

    def select(self, node, solver) -> Conflict:
        return node.conflicts[0]


class RandomSelector(ConflictSelector):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select(self, node, solver) -> Conflict:
        return self.rng.choice(node.conflicts)


class EarliestSelector(ConflictSelector):
    """Pick the conflict with the smallest timestep."""
    name = "earliest"

    def select(self, node, solver) -> Conflict:
        return min(node.conflicts, key=lambda c: c.time)


class MostConflictsSelector(ConflictSelector):
    """Degree heuristic: prefer conflicts whose agents are involved in many
    conflicts at this node."""
    name = "most-conflicts"

    def select(self, node, solver) -> Conflict:
        deg = Counter()
        for c in node.conflicts:
            deg[c.a1] += 1
            deg[c.a2] += 1
        return max(node.conflicts, key=lambda c: deg[c.a1] + deg[c.a2])


_CARD_RANK = {"cardinal": 0, "semi": 1, "non": 2}


class CardinalSelector(ConflictSelector):
    """ICBS rule: cardinal > semi-cardinal > non-cardinal, tie-break earliest.

    This is the strong hand-crafted baseline our learned methods must beat."""
    name = "cardinal"

    def select(self, node, solver) -> Conflict:
        solver.classify(node)
        return min(
            node.conflicts,
            key=lambda c: (_CARD_RANK.get(c.cardinality, 2), c.time),
        )
