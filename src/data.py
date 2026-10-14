"""Data loading, stratified splitting, cross-validation folds and scaling."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Breast Cancer Wisconsin (Diagnostic). Label 1 = malignant (positive class).

    scikit-learn encodes 0 = malignant, so the labels are flipped.
    """
    data = load_breast_cancer()
    return data.data, 1 - data.target.astype(float), list(data.feature_names)


def stratified_split(y: np.ndarray, fractions: list[float], seed: int = 42) -> list[np.ndarray]:
    """Split indices so every part keeps the class ratio of ``y``."""
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in fractions]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        cuts = (np.cumsum(fractions)[:-1] * len(idx)).astype(int)
        for part, chunk in zip(parts, np.split(idx, cuts)):
            part.extend(chunk.tolist())
    return [rng.permutation(np.array(p)) for p in parts]


def stratified_kfold(
    y: np.ndarray, k: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` pairs with stratified folds."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for f, chunk in enumerate(np.array_split(idx, k)):
            folds[f].extend(chunk.tolist())
    arrays = [np.array(f) for f in folds]
    for i in range(k):
        train = np.concatenate([arrays[j] for j in range(k) if j != i])
        yield train, arrays[i]


def fit_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std from the training set only (constant features get std = 1)."""
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def transform(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Standardize with previously fitted statistics."""
    return (x - mu) / sigma
