"""Learned conflict selectors: a linear ranker and a small MLP ranker.

Both score every conflict at a CBS node and pick the argmax (Huang et al.,
AAAI 2021). Inference is implemented in numpy. Training lives in
``scripts/train_selector.py``.
"""
from __future__ import annotations

import numpy as np

from .features import extract_node_features, N_FEATURES


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = np.where(std < 1e-8, 1.0, std)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class LearnedLinearSelector:
    """score(conflict) = w . standardize(features) + b ; pick argmax."""
    name = "learned-linear"

    def __init__(self, w: np.ndarray, b: float, std: Standardizer):
        self.w = w
        self.b = b
        self.std = std

    @classmethod
    def load(cls, path: str) -> "LearnedLinearSelector":
        d = np.load(path)
        return cls(d["w"], float(d["b"]), Standardizer(d["mean"], d["std"]))

    def scores(self, node, solver):
        X, conflicts = extract_node_features(node, solver)
        return self.std(X) @ self.w + self.b, conflicts

    def select(self, node, solver):
        s, conflicts = self.scores(node, solver)
        return conflicts[int(np.argmax(s))]


class LearnedMLPSelector:
    """One hidden layer (tanh) ranker, pure-numpy forward pass."""
    name = "learned-mlp"

    def __init__(self, W1, b1, W2, b2, std: Standardizer):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.std = std

    @classmethod
    def load(cls, path: str) -> "LearnedMLPSelector":
        d = np.load(path)
        return cls(d["W1"], d["b1"], d["W2"], d["b2"],
                   Standardizer(d["mean"], d["std"]))

    def scores(self, node, solver):
        X, conflicts = extract_node_features(node, solver)
        h = np.tanh(self.std(X) @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel(), conflicts

    def select(self, node, solver):
        s, conflicts = self.scores(node, solver)
        return conflicts[int(np.argmax(s))]
