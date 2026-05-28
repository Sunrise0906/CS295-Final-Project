"""Hybrid selectors that compose cardinal with learned tie-breaking.

The cardinal rule is sharp: in the common case where exactly one conflict is
cardinal, picking it is the right move. The learned linear ranker is most useful
when several cardinal conflicts are tied and need a finer tie-breaker. The
hybrid restricts the learned ranker to that tie-breaking role and falls back to
cardinal otherwise.
"""
from __future__ import annotations

from .features import extract_node_features


class HybridCardinalLearned:
    """Use cardinal's classification, then break ties among cardinal conflicts
    with the learned ranker. If no cardinal conflict exists, semi/non in turn."""
    name = "hybrid-card-linear"

    def __init__(self, learned):
        self.learned = learned
        self.chosen_feature = None

    def select(self, node, solver):
        solver.classify(node)
        # Bucket by cardinality.
        card = [(i, c) for i, c in enumerate(node.conflicts) if c.cardinality == "cardinal"]
        semi = [(i, c) for i, c in enumerate(node.conflicts) if c.cardinality == "semi"]
        non  = [(i, c) for i, c in enumerate(node.conflicts) if c.cardinality == "non"]
        bucket = card or semi or non
        if len(bucket) == 1:
            return bucket[0][1]
        # Otherwise score with the learned ranker and pick the best within the
        # bucket.
        scores, conflicts = self.learned.scores(node, solver)
        best_local, best_score = bucket[0][0], scores[bucket[0][0]]
        for i, _ in bucket[1:]:
            if scores[i] > best_score:
                best_score = scores[i]
                best_local = i
        return conflicts[best_local]
