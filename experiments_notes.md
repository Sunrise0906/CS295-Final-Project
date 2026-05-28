# Follow-up experiments

Additional experiments run after the report was submitted, exploring whether
the linear ranker can be pushed further. The honest finding is that the linear
model is at the achievable ceiling in this setup; the bottleneck is the
oracle's training signal, not the model architecture.

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
| Stronger oracle (linear-rollout) | oracle uses learned linear as rollout policy         | small improvement only in dense regimes, much slower |

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
4. **The bottleneck is oracle noise.** The subtree oracle rolls out with cardinal
   under a fixed node budget; when subtree solves saturate, the oracle returns
   a penalty value and mis-ranks conflicts. A 3x larger budget gives slightly
   cleaner labels (`data/trajectories/strong_train_clean.npz`) and a small win
   on one config (d=0.1, 14 agents: geometric-mean ratio 0.60 vs 0.68), but the
   improvement is within noise on others.
5. **Hybrid (cardinal + linear tie-break) is identical to linear alone.** This
   confirms the linear has already learned the "prefer cardinal" rule and is
   doing its real work in the tie-breaking step.

## Likely paths to further improvement (not pursued)

Each of these would require more time than was available:

- **Reinforcement learning** with reward = -expansions. Skips the oracle entirely
  and lets the model learn to minimize search effort directly. Costly to set up
  (each trajectory is a full CBS solve); needs careful reward shaping.
- **A 2-step lookahead oracle**: at each node, evaluate each candidate by
  branching twice and rolling out, not just once. Strictly stronger oracle but
  quadratically more expensive to collect.
- **Classical CBS speedups composed with the learned selector**: Bypass moves,
  CBSH-style admissible high-level heuristics, disjoint splitting. These are
  orthogonal to learning and would reliably reduce expansions.

## Files added/changed in this round

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
