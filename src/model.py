"""Core model: sigmoid, regularized cost, gradient and batch gradient descent."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (the exponent is clipped to avoid overflow)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict_proba(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Probability of the positive class for every row of ``x``."""
    return sigmoid(x @ w + b)


def cost(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> float:
    """L2-regularized binary cross-entropy. The bias ``b`` is not penalized."""
    m = len(x)
    f = np.clip(predict_proba(x, w, b), 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(f) + (1 - y) * np.log(1 - f))
    return float(loss + (lambda_ / (2 * m)) * np.sum(w ** 2))


def gradient(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0
) -> tuple[np.ndarray, float]:
    """Gradient of the regularized cost with respect to ``w`` and ``b``."""
    m = len(x)
    error = predict_proba(x, w, b) - y
    dw = (x.T @ error) / m + (lambda_ / m) * w
    db = float(np.mean(error))
    return dw, db


def fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.5,
    max_iter: int = 20000,
    lambda_: float = 1.0,
    tol: float = 1e-12,
    verbose: bool = True,
) -> tuple[np.ndarray, float, list[float]]:
    """Batch gradient descent with early stopping on cost improvement.

    Returns the weights, the bias and the cost history.
    """
    w = np.zeros(x.shape[1])
    b = 0.0
    history: list[float] = []
    for i in range(max_iter):
        dw, db = gradient(x, y, w, b, lambda_)
        w = w - alpha * dw
        b = b - alpha * db
        history.append(cost(x, y, w, b, lambda_))
        if verbose and i % 500 == 0:
            print(f"iter {i:5d} | cost = {history[-1]:.6f}")
        if i > 0 and abs(history[-2] - history[-1]) < tol:
            if verbose:
                print(f"converged at iter {i} | cost = {history[-1]:.6f}")
            break
    return w, b, history
