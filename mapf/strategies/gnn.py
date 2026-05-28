"""GNN-based conflict selector over the conflict graph.

For a CBS node with k conflicts, the conflict graph has those k conflicts as
nodes and connects two conflicts if they share at least one agent. A small
graph-attention network passes messages over this graph and outputs a score for
each conflict; the argmax is selected. Trained by listwise softmax cross-entropy
against the oracle's choice, identical objective to the linear/MLP rankers.

Requires torch. Not imported by ``strategies/__init__`` to keep the numpy core
torch-free.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .features import extract_node_features_ext


def build_edge_index(agent_pairs: np.ndarray) -> np.ndarray:
    """Conflict-graph edges (both directions). ``agent_pairs`` is (n, 2)."""
    n = len(agent_pairs)
    src, dst = [], []
    for i in range(n):
        ai1, ai2 = agent_pairs[i]
        for j in range(i + 1, n):
            aj1, aj2 = agent_pairs[j]
            if ai1 == aj1 or ai1 == aj2 or ai2 == aj1 or ai2 == aj2:
                src.extend([i, j])
                dst.extend([j, i])
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


class GraphAttnLayer(nn.Module):
    """One layer of graph attention. Aggregates incoming messages with softmax
    attention; residual + tanh."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin_self = nn.Linear(hidden, hidden)
        self.lin_msg = nn.Linear(hidden, hidden)
        self.attn = nn.Linear(2 * hidden, 1)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return torch.tanh(self.lin_self(h))
        src, dst = edge_index[0], edge_index[1]
        h_pair = torch.cat([h[dst], h[src]], dim=1)
        e = self.attn(h_pair).squeeze(-1)
        e = e - e.max()
        e_exp = torch.exp(e)
        n = h.shape[0]
        denom = torch.zeros(n, device=h.device).scatter_add_(0, dst, e_exp) + 1e-9
        alpha = e_exp / denom[dst]
        msg = self.lin_msg(h)[src] * alpha.unsqueeze(-1)
        agg = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, dst, msg)
        return torch.tanh(self.lin_self(h) + agg)


class ConflictGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(n_features, hidden)
        self.layers = nn.ModuleList([GraphAttnLayer(hidden) for _ in range(n_layers)])
        self.scorer = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.in_proj(X))
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.dropout(h)
        return self.scorer(h).squeeze(-1)


class GNNSelector:
    """CBS conflict selector backed by a trained ConflictGNN."""
    name = "learned-gnn"

    def __init__(self, model: ConflictGNN, mean: np.ndarray, std: np.ndarray,
                 device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.mean = mean
        self.std = np.where(std < 1e-8, 1.0, std)
        self.chosen_feature = None

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GNNSelector":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = ConflictGNN(**ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["mean"], ckpt["std"], device)

    @torch.no_grad()
    def _scores(self, node, solver):
        X, conflicts = extract_node_features_ext(node, solver)
        pairs = np.array([[c.a1, c.a2] for c in conflicts])
        edges = build_edge_index(pairs)
        Xn = (X - self.mean) / self.std
        Xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        et = torch.tensor(edges, dtype=torch.int64, device=self.device)
        return self.model(Xt, et).cpu().numpy(), conflicts, X

    def select(self, node, solver):
        scores, conflicts, X = self._scores(node, solver)
        idx = int(np.argmax(scores))
        self.chosen_feature = X[idx]
        return conflicts[idx]


class EnsembleSelector:
    """Average the standardized scores of multiple base selectors (linear or
    GNN). Both must expose a ``scores(node, solver) -> (np.ndarray, conflicts)``
    or a ``_scores(node, solver) -> (np.ndarray, conflicts, X)`` method."""
    name = "ensemble"

    def __init__(self, bases: list):
        self.bases = bases
        self.chosen_feature = None

    def select(self, node, solver):
        all_scores = []
        conflicts = None
        for b in self.bases:
            if hasattr(b, "_scores"):
                s, c, _ = b._scores(node, solver)
            else:
                s, c = b.scores(node, solver)
            conflicts = c
            s = np.asarray(s, dtype=np.float64)
            s = (s - s.mean()) / (s.std() + 1e-9)
            all_scores.append(s)
        idx = int(np.argmax(np.mean(all_scores, axis=0)))
        return conflicts[idx]
