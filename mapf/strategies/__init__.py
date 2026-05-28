"""Conflict-selection strategy registry."""
from __future__ import annotations

from .hardcoded import (
    ConflictSelector,
    FirstSelector,
    RandomSelector,
    EarliestSelector,
    MostConflictsSelector,
    CardinalSelector,
)

# name -> zero-arg (or default-arg) constructor for the experiment harness.
REGISTRY = {
    "first": FirstSelector,
    "random": RandomSelector,
    "earliest": EarliestSelector,
    "most-conflicts": MostConflictsSelector,
    "cardinal": CardinalSelector,
}


def make_selector(name: str, **kwargs) -> ConflictSelector:
    if name not in REGISTRY:
        raise KeyError(f"unknown selector '{name}'; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
