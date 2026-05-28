# Learning Conflict Prioritization for Conflict-Based Search

Course Project 3 (MAPF). Authors: Natalie Pham, Chaoyang Wang.

This repository contains a Conflict-Based Search (CBS) implementation with a
pluggable conflict-selection interface, several learned conflict selectors, and
the experiments reported in `overleaf/main.tex`.

## Summary of results

- Conflict selection changes high-level CBS expansions by up to an order of
  magnitude on random 8x8 grids. The cardinal heuristic is hard to beat with
  naive rules; `earliest`-time selection is often worse than `first`.
- A learned linear ranker, trained to imitate a subtree-minimizing oracle (a
  one-step policy improvement over cardinal), beats cardinal on hard instances.
  Per-instance geometric-mean expansion ratio drops to 0.48 with win-rate 75%
  at density 0.2, 12 agents. The advantage grows with agent and obstacle
  density.
- The win transfers to unseen 10x10 and 12x12 grids without retraining, and
  survives in wall-clock time (about 2x faster on hard cells). Solution cost
  remains optimal throughout.
- The linear ranker beats a one-hidden-layer MLP that overfits the oracle. A
  feature ablation shows the win comes from continuous MDD widths used to break
  ties among cardinal conflicts; cost/distance/timing features are redundant
  with cardinality.
- Two further directions did not robustly improve over their baselines:
  - Learned focal node ordering in bounded-suboptimal ECBS (high variance, an
    on-path classifier is a poor stand-alone priority).
  - A GRU over the search history of resolved conflicts (no gain over the
    memoryless linear ranker).

## Layout

```
mapf/
  core.py          data structures
  low_level.py     space-time A* with constraints
  conflicts.py     vertex/edge conflict detection
  mdd.py           MDD construction and cardinality classification
  cbs.py           optimal CBS with pluggable selector and subtree rollouts
  ecbs.py          bounded-suboptimal focal search
  instances.py    random grids, structured maps, MovingAI loader, validator
  strategies/
    hardcoded.py   first / random / earliest / most-conflicts / cardinal
    features.py    per-conflict feature extraction
    oracle.py      one-step lookahead oracle + subtree-minimizing oracle
    learned.py     linear and MLP rankers (numpy)
    sequence.py    GRU sequence selector (torch)
    focal.py       learned and blended focal node ordering for ECBS
scripts/
  smoke_test.py        correctness tests
  collect_strong.py    subtree-oracle imitation data
  train_selector.py    train linear / MLP rankers
  sweep_learned.py     per-instance learned-vs-cardinal sweep
  analyze_learned.py   per-instance statistics and figure
  run_experiments.py   hand-crafted strategy sweep
  plot.py              hand-crafted strategy figures
  ablate_features.py   feature ablation for the linear ranker
  collect_focal.py     ECBS focal-ordering data
  train_focal.py       train focal classifier
  eval_focal.py        ECBS focal evaluation
  train_seq_selector.py train GRU (torch)
  eval_seq.py          sequence-model evaluation
  reproduce.py         one-command reproduction of the core results
data/, results/, models/, overleaf/
```

## Setup

Core library and experiments run on Python 3.10+ with numpy and matplotlib.

```
pip install -r requirements.txt
```

The sequence-model training and evaluation need PyTorch with CUDA; run those
in a separate environment.

## Reproduce the main results

```
python -m scripts.smoke_test
python -m scripts.run_experiments         # hand-crafted strategies (RQ1, RQ3)
python -m scripts.plot
python -m scripts.collect_strong          # oracle data
python -m scripts.train_selector --data data/trajectories/strong_train.npz --model linear
python -m scripts.train_selector --data data/trajectories/strong_train.npz --model mlp
python -m scripts.sweep_learned           # learned vs cardinal
python -m scripts.analyze_learned
python -m scripts.ablate_features
```

Or run `python -m scripts.reproduce` (add `--quick` for a faster sanity pass).

ECBS focal ordering and the sequence model are optional and run from the
corresponding `collect_*`, `train_*`, and `eval_*` scripts.

## References

- Sharon, Stern, Felner, Sturtevant. Conflict-Based Search for Optimal
  Multi-Agent Pathfinding. AIJ, 2015.
- Barer, Sharon, Stern, Felner. Suboptimal Variants of the Conflict-Based
  Search Algorithm. SoCS, 2014.
- Boyarski et al. ICBS: Improved Conflict-Based Search. IJCAI, 2015.
- Huang, Koenig, Dilkina. Learning to Resolve Conflicts for MAPF with CBS.
  AAAI, 2021.
- Gu, Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
  arXiv:2312.00752, 2023.
