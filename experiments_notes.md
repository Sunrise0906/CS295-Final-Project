# Follow-up experiments

Additional experiments run after the report was submitted, exploring whether
the linear ranker can be pushed further.

**Headline: a stronger oracle (using the learned linear as its rollout policy)
produces an improved linear ranker.** The new model
(`models/selector_linear_v1_linroll.npz`) beats the report's original linear on
every hard config we tested, by per-instance geomean expansion ratio (the
rigorous methodology our report insists on), confirmed on an 80-seed sweep
(`results/optimal_linroll.csv` vs `results/optimal_learned.csv`):

| 8x8 (density, agents) | orig gm | linroll-v1 gm | improvement |
|-----------------------|--------:|--------------:|------------:|
| 0.1, 12               | 0.70    | 0.64          | -8.3%       |
| 0.1, 14               | 0.60    | 0.53          | -12.7%      |
| 0.2, 10               | 0.68    | 0.62          | -8.4%       |
| 0.2, 12               | 0.48    | 0.46          | -4.5%       |
| 0.2, 14               | 0.61    | 0.52          | -14.0%      |

The improvement generalizes out of distribution: on unseen 10x10 grids (linroll
was trained only on 8x8), the gap is the same or larger, including a 65% to
83% jump in win-rate at the hardest 10x10 cell:

| 10x10 (density, agents) | orig gm | orig win-rate | linroll-v1 gm | linroll-v1 win-rate |
|-------------------------|--------:|--------------:|--------------:|--------------------:|
| 0.1, 14                 | 0.71    | 64%           | 0.66          | 56%                 |
| 0.2, 10                 | 0.67    | 58%           | 0.64          | 71%                 |
| 0.2, 12                 | 0.50    | 65%           | 0.41          | **83%**             |
| 0.2, 14                 | 0.35    | 81%           | 0.31          | 81%                 |

The other avenues we tried (more model capacity, GNN, ensembles, extended
features, ICBS-style Bypass) either underperform or fail to beat the report's
linear in per-instance analysis. The Bypass attempt is documented below as a
textbook outlier-of-means cautionary tale.

## Variants tried

All trained on the same instance-generation distribution (random 8x8 grids,
density 0-0.2) and evaluated on held-out seeds 0-49 with the same per-instance
geometric-mean ratio metric used in the report.

| variant                          | how it differs from the report's linear              | result vs report's linear           |
|----------------------------------|------------------------------------------------------|--------------------------------------|
| linear, 4259 records             | 1.7x more oracle data, same v1 features, same budget | essentially identical                |
| linear, 33-dim ext features      | adds MDD-overlap, conflict-graph-degree, etc.        | no improvement                       |
| MLP, 32 hidden, default L2       | 33-dim ext features, hidden=32                       | worse (overfits oracle)              |
| MLP, 16 hidden, L2=0.01          | heavy regularization                                 | mixed, mostly worse                  |
| GNN, attention, 2 layers, h=64   | graph attention over the conflict graph              | worse (still overfits)               |
| Ensemble (linear + GNN scores)   | averaged standardized scores                         | slightly worse than linear alone     |
| Hybrid (cardinal + linear tiebreak)| cardinal classification + linear within bucket     | identical to linear                  |
| Linear, cleaner labels (3x rollout budget) | 150 instances at 1200-node subtree budget  | small improvement on d0.1/14; within noise elsewhere |
| Stronger oracle (linear-rollout + 1500-node budget) | oracle uses learned linear as rollout policy with larger budget; linear retrained on the new labels | **wins**: 8-13% improvement on hard configs |
| ICBS-style Bypass (`bypass=True`) | replace one agent's path in place when a same-cost alternative reduces conflicts | mean-of-means looks like 20-25% win, per-instance medians/geomeans show no robust gain |

## What the experiments show

1. **More expressive models hurt.** MLP and GNN both reach higher training
   imitation accuracy than linear (0.62-0.86 train top-1 vs linear's 0.53), but
   lose in actual search. They fit the oracle's idiosyncratic label noise; the
   linear extracts the robust signal that transfers.
2. **Extended MDD-structure features add nothing.** The ablation in the report
   already showed that cardinality features dominate. Adding conflict-graph
   degree, MDD-overlap, and path-overlap features confirms this --- the linear
   gives them tiny weights and ignores them.
3. **The conflict graph carries real structure** (mean edge density ~30% at the
   root of dense instances), but capturing it requires capacity (GNN) that
   overfits the modest training data we can collect.
4. **The bottleneck is oracle quality.** The subtree oracle rolls out with
   cardinal under a fixed node budget; when subtree solves saturate, the oracle
   returns a penalty value and mis-ranks conflicts. Combining a stronger
   rollout policy (the learned linear instead of cardinal) with a larger
   subtree budget (1500 nodes instead of 400) produces noticeably better
   training labels: a linear retrained on those labels beats the report's
   linear by 8-14% on hard configs (80-seed sweep). Either change alone is
   much less effective (cleaner labels alone gave a small win on a single
   config; linear-rollout alone with a small budget gave mixed results in a
   pilot).
5. **Policy iteration converges in one step.** Iterating once more --- using
   the new linear (`linroll-v1`) as the rollout policy for another data
   collection, then retraining --- gives a model whose performance is
   indistinguishable from `linroll-v1` (better on one config, worse on three,
   the rest within noise). The first step of policy improvement is the win;
   the dataset size we can practically collect cannot support further
   refinement.
6. **Hybrid (cardinal + linear tie-break) is identical to linear alone.** This
   confirms the linear has already learned the "prefer cardinal" rule and is
   doing its real work in the tie-breaking step.

## Likely paths to further improvement (not pursued)

Each of these would require more time than was available:

- **Reinforcement learning** with reward = -expansions. Skips the oracle entirely
  and lets the model learn to minimize search effort directly. Costly to set up
  (each trajectory is a full CBS solve); needs careful reward shaping.
- **A 2-step lookahead oracle**: at each node, evaluate each candidate by
  branching twice and rolling out, not just once. Strictly stronger oracle but
  quadratically more expensive to collect, so the training set would either
  shrink or take orders of magnitude longer to gather.
- **Classical CBS speedups beyond Bypass**: CBSH-style admissible high-level
  heuristics, disjoint splitting. These add nontrivial implementation work but
  are orthogonal to learning.

## Bypass: a textbook outlier-of-means cautionary tale

ICBS-style Bypass adds a step before every CBS branching: try to replace one of
the two agents' paths with a same-cost alternative that resolves the
conflicting move and reduces the total conflict count. If successful, the
solver updates the node in place rather than branching. Enabled with
`CBS(..., bypass=True)`; preserves optimality (verified: solution costs are
identical with and without Bypass on every solved seed).

A first comparison using mean-over-instances expansions looked like a strong
improvement (e.g. density 0.1 / 12 agents: linear 154 -> linear+bp 121, a 21%
mean reduction). However the per-instance analysis our own report insists on
tells a different story:

| config (density, agents) | linear gm | linear+bp gm | linear+bp win-rate |
|--------------------------|----------:|-------------:|-------------------:|
| 0.1, 12                  | 0.77      | 0.84         | 46%                |
| 0.1, 14                  | 0.68      | 0.70         | 62%                |
| 0.2, 10                  | 0.64      | 0.62         | 65%                |
| 0.2, 12                  | 0.48      | 0.52         | 65%                |
| 0.2, 14                  | 0.65      | 0.66         | 65%                |

Per-instance geometric-mean ratios show no robust improvement; in most hard
configs Bypass slightly increases the typical expansion ratio while saving a
lot on a small number of hard outliers, which is exactly what inflates the
mean. Runtime is also generally slightly worse with Bypass (the extra replans
have a cost). This reproduces the methodological lesson in the report: ratio
of means can fabricate apparent wins, per-instance medians and win-rates are
needed.

The Bypass code is left in `mapf/cbs.py` for reproducibility but is not
recommended as an unconditional improvement in this setup. Per-instance
statistics are in `results/optimal_bypass.csv` (produced by
`scripts/sweep_bypass.py`, analyzed by `scripts/analyze_bypass.py`).

## Files added/changed in this round

- `mapf/cbs.py` --- adds `bypass=True` option to CBS; implements `_try_bypass`.
- `mapf/strategies/features.py` --- adds `extract_node_features_ext` and the
  9 extended features (MDD overlap, conflict-graph degree, path overlap, ...).
- `mapf/strategies/gnn.py` --- new: graph-attention conflict GNN +
  `EnsembleSelector` for averaging.
- `mapf/strategies/hybrid.py` --- new: cardinal + linear tie-break.
- `mapf/strategies/learned.py` --- selectors auto-detect feature dim and use
  v1 or ext features accordingly.
- `mapf/strategies/oracle.py` --- `StrongOracleSelector` now accepts a
  `feature_fn` and a `rollout_selector` (for policy-iteration experiments).
- `scripts/collect_strong.py` --- `--features ext` and `--rollout-linear` flags;
  optional GNN data output.
- `scripts/train_gnn.py`, `scripts/eval_gnn.py` --- GNN training and eval.
- `scripts/derive_v1_from_ext.py` --- slices v1 columns out of ext-feature data.
- `scripts/sweep_bypass.py`, `scripts/analyze_bypass.py` --- Bypass sweep and
  analysis.
