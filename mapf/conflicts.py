"""Conflict detection over a joint MAPF solution."""
from __future__ import annotations

from .core import Cell, Conflict, pos_at


def find_first_conflict(paths: list[list[Cell]]):
    """First conflict in time order (used by vanilla CBS), or None."""
    confs = find_all_conflicts(paths, first_only=True)
    return confs[0] if confs else None


def find_all_conflicts(paths: list[list[Cell]], first_only: bool = False) -> list[Conflict]:
    """All pairwise vertex/edge conflicts in the joint solution.

    A vertex conflict: two agents share a cell at time t.
    An edge conflict: two agents swap cells between t-1 and t (head-on).
    """
    n = len(paths)
    horizon = max((len(p) for p in paths), default=1)
    conflicts: list[Conflict] = []

    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = paths[i], paths[j]
            prev_i = pos_at(pi, 0)
            prev_j = pos_at(pj, 0)
            for t in range(horizon):
                ci = pos_at(pi, t)
                cj = pos_at(pj, t)
                # Vertex conflict.
                if ci == cj:
                    conflicts.append(Conflict(i, j, "vertex", ci, None, t))
                    if first_only:
                        return [_earliest(conflicts)]
                # Edge (swap) conflict between t-1 and t.
                if t > 0 and ci == prev_j and cj == prev_i and ci != cj:
                    # a_i moves prev_i -> ci ; a_j moves prev_j -> cj = prev_i
                    conflicts.append(Conflict(i, j, "edge", prev_i, ci, t))
                    if first_only:
                        return [_earliest(conflicts)]
                prev_i, prev_j = ci, cj

    if first_only and conflicts:
        return [_earliest(conflicts)]
    return conflicts


def _earliest(conflicts: list[Conflict]) -> Conflict:
    return min(conflicts, key=lambda c: c.time)
