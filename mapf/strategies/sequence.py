"""Sequence-model conflict selector.

A GRU encodes the root-to-node history of resolved-conflict feature vectors into
a context vector; each candidate conflict at the current node is then scored by
an MLP on ``[context, candidate_features]``. Trained by imitation of the oracle
with the same listwise softmax cross-entropy as the memoryless rankers.

This module requires torch and is not imported by ``strategies/__init__`` so the
pure-numpy core does not depend on torch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .features import extract_node_features, N_FEATURES


class SeqConflictNet(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64,
                 score_hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden))
        self.scorer = nn.Sequential(
            nn.Linear(hidden + n_features, score_hidden),
            nn.Tanh(),
            nn.Linear(score_hidden, 1),
        )

    def context(self, history: torch.Tensor) -> torch.Tensor:
        """history: (L, F) -> context (hidden,). Empty history -> learned h0."""
        if history.shape[0] == 0:
            return self.h0.view(-1)
        out, hn = self.gru(history.unsqueeze(0), self.h0.contiguous())
        return hn.view(-1)

    def forward(self, history: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """candidates: (n, F) -> scores (n,)."""
        ctx = self.context(history)
        ctx_rep = ctx.unsqueeze(0).expand(candidates.shape[0], -1)
        return self.scorer(torch.cat([ctx_rep, candidates], dim=1)).squeeze(-1)


class SeqSelector:
    """CBS conflict selector backed by a trained SeqConflictNet. Requires the
    solver to run with ``track_history=True``."""
    name = "sequence"

    def __init__(self, model: SeqConflictNet, mean: np.ndarray, std: np.ndarray,
                 device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.mean = mean
        self.std = np.where(std < 1e-8, 1.0, std)
        self.chosen_feature = None

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SeqSelector":
        ckpt = torch.load(path, map_location=device)
        model = SeqConflictNet(**ckpt.get("arch", {}))
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["mean"], ckpt["std"], device)

    def _std(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    @torch.no_grad()
    def select(self, node, solver):
        X, conflicts = extract_node_features(node, solver)
        Xs = torch.tensor(self._std(X), dtype=torch.float32, device=self.device)
        if node.history:
            H = np.stack(node.history, axis=0)
            Hs = torch.tensor(self._std(H), dtype=torch.float32, device=self.device)
        else:
            Hs = torch.zeros((0, N_FEATURES), dtype=torch.float32, device=self.device)
        scores = self.model(Hs, Xs).cpu().numpy()
        idx = int(np.argmax(scores))
        self.chosen_feature = X[idx]   # raw, to match history storage
        return conflicts[idx]
