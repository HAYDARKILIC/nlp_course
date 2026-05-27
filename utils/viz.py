"""Plotting helpers used across the course notebooks."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_attention(matrix: np.ndarray, x_labels: Sequence[str], y_labels: Sequence[str], title: str = "Attention") -> plt.Figure:
    """Visualize an attention weight matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 0.6), max(4, len(y_labels) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_training_curves(history: dict[str, list[float]], title: str = "Training") -> plt.Figure:
    """Plot one curve per key in `history`."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def project_embeddings_2d(emb: np.ndarray, method: str = "pca") -> np.ndarray:
    """Project a (N, d) embedding matrix to 2D for visualization.

    Defaults to PCA (cheap, deterministic). Switch to `t-sne` or `umap` for
    nonlinear visualization once the relevant library is available.
    """
    if method == "pca":
        x = emb - emb.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return (u[:, :2] * s[:2])
    raise ValueError(f"Unsupported method: {method!r}")
