"""Figures saved as PNG files (headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def plot_cost(history: list[float], path: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.plot(history, color="#1f77b4")
    plt.xlabel("iteration")
    plt.ylabel("regularized cost")
    plt.title("Convergence of gradient descent")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_thresholds(rows: np.ndarray, selected: float, path: Path) -> None:
    plt.figure(figsize=(6.5, 3.8))
    plt.plot(rows[:, 0], rows[:, 1], label="precision")
    plt.plot(rows[:, 0], rows[:, 2], label="recall")
    plt.plot(rows[:, 0], rows[:, 4], label="F2", linestyle="--")
    plt.axvline(selected, color="grey", linestyle=":", label=f"selected = {selected:.2f}")
    plt.xlabel("decision threshold")
    plt.ylabel("score (validation)")
    plt.title("Precision / recall trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion(cm: np.ndarray, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["benign", "malignant"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["benign", "malignant"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"Test confusion matrix (threshold {threshold:.2f})")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weights(w: np.ndarray, names: list[str], path: Path, top: int = 10) -> None:
    order = np.argsort(-np.abs(w))[:top]
    colors = ["#d62728" if w[i] > 0 else "#1f77b4" for i in order][::-1]
    plt.figure(figsize=(7, 4))
    plt.barh([names[i] for i in order][::-1], w[order][::-1], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("weight (standardized features)")
    plt.title(f"Top {top} features by |weight|   (red: raises malignancy risk)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
