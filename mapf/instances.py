"""Benchmark instances: random-grid generator, MovingAI loader, validator."""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Optional

from .core import Agent, Cell, GridMap, MAPFInstance, MOVES, pos_at
from .conflicts import find_all_conflicts


# --- random grids ------------------------------------------------------------

def _reachable_from(grid: GridMap, src: Cell) -> set[Cell]:
    seen = {src}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nb in grid.neighbors(cur):
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def random_instance(
    height: int,
    width: int,
    n_agents: int,
    obstacle_density: float = 0.0,
    seed: int = 0,
    max_tries: int = 2000,
) -> Optional[MAPFInstance]:
    """Random grid with random obstacles and start/goal pairs.

    Each agent's goal is reachable from its start (same connected component).
    Returns None if it cannot place all agents within ``max_tries``."""
    rng = random.Random(seed)
    n_obstacles = int(round(height * width * obstacle_density))
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    obstacles = set(rng.sample(all_cells, n_obstacles)) if n_obstacles else set()
    grid = GridMap(height, width, obstacles)

    free = [c for c in all_cells if c not in obstacles]
    if len(free) < 2 * n_agents:
        return None

    agents: list[Agent] = []
    used_starts: set[Cell] = set()
    used_goals: set[Cell] = set()
    tries = 0
    while len(agents) < n_agents and tries < max_tries:
        tries += 1
        s = rng.choice(free)
        g = rng.choice(free)
        if s == g or s in used_starts or g in used_goals:
            continue
        if g not in _reachable_from(grid, s):
            continue
        agents.append(Agent(len(agents), s, g))
        used_starts.add(s)
        used_goals.add(g)

    if len(agents) < n_agents:
        return None
    return MAPFInstance(grid, agents)


def place_agents(grid: GridMap, n_agents: int, seed: int = 0,
                 max_tries: int = 4000) -> Optional[MAPFInstance]:
    """Place ``n_agents`` random start/goal pairs on a *given* grid (goals
    reachable from starts). Used to put agents on structured maps."""
    rng = random.Random(seed)
    free = [(r, c) for r in range(grid.height) for c in range(grid.width)
            if grid.is_free((r, c))]
    if len(free) < 2 * n_agents:
        return None
    agents, used_s, used_g, tries = [], set(), set(), 0
    while len(agents) < n_agents and tries < max_tries:
        tries += 1
        s, g = rng.choice(free), rng.choice(free)
        if s == g or s in used_s or g in used_g:
            continue
        if g not in _reachable_from(grid, s):
            continue
        agents.append(Agent(len(agents), s, g))
        used_s.add(s)
        used_g.add(g)
    if len(agents) < n_agents:
        return None
    return MAPFInstance(grid, agents)


def rooms_grid(size: int = 13) -> GridMap:
    """A structured 4-rooms map: walls along the central row and column, each
    split by two doorways. A classic non-random MAPF topology (bottlenecks)."""
    mid = size // 2
    doors = {size // 4, 3 * size // 4}
    obstacles = set()
    for i in range(size):
        if i not in doors:
            obstacles.add((mid, i))
            obstacles.add((i, mid))
    return GridMap(size, size, obstacles)


# --- MovingAI format ---------------------------------------------------------

def load_movingai_map(path: str | Path) -> GridMap:
    lines = Path(path).read_text().splitlines()
    height = width = 0
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.startswith("height"):
            height = int(ln.split()[1])
        elif ln.startswith("width"):
            width = int(ln.split()[1])
        elif ln.strip() == "map":
            body_start = i + 1
            break
    obstacles = set()
    for r, ln in enumerate(lines[body_start:body_start + height]):
        for c, ch in enumerate(ln):
            if ch not in (".", "G", "S"):  # passable tokens in MovingAI
                obstacles.add((r, c))
    return GridMap(height, width, obstacles)


def load_movingai_scen(path: str | Path, grid: GridMap, n_agents: int) -> MAPFInstance:
    """Load the first ``n_agents`` start/goal pairs from a .scen file.

    MovingAI columns: bucket map width height sx sy gx gy optimal, with (x=col,
    y=row)."""
    lines = Path(path).read_text().splitlines()
    agents: list[Agent] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 9 or parts[0] == "version":
            continue
        sx, sy, gx, gy = int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])
        agents.append(Agent(len(agents), (sy, sx), (gy, gx)))
        if len(agents) == n_agents:
            break
    return MAPFInstance(grid, agents)


# --- validation --------------------------------------------------------------

def validate(instance: MAPFInstance, paths: list[list[Cell]]) -> tuple[bool, str]:
    """Check that a solution is well-formed and collision-free."""
    grid = instance.grid
    for i, (a, p) in enumerate(zip(instance.agents, paths)):
        if not p or p[0] != a.start:
            return False, f"agent {i} does not start at its start cell"
        if p[-1] != a.goal:
            return False, f"agent {i} does not end at its goal cell"
        for t in range(1, len(p)):
            if not grid.is_free(p[t]):
                return False, f"agent {i} occupies blocked cell {p[t]} at t={t}"
            dr = abs(p[t][0] - p[t - 1][0])
            dc = abs(p[t][1] - p[t - 1][1])
            if (dr, dc) not in ((0, 0), (1, 0), (0, 1)):
                return False, f"agent {i} makes an illegal move at t={t}"
    conflicts = find_all_conflicts(paths)
    if conflicts:
        return False, f"{len(conflicts)} residual conflict(s), e.g. {conflicts[0]}"
    return True, "ok"
