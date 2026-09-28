"""Confusion matrix image export (Phase 6 spec sec. 18, 37)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs from CLI scripts, never a GUI
import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_image(
    confusion_matrix: list[list[int]], labels: list[str], title: str, path: str | Path
) -> Path:
    cm = np.array(confusion_matrix)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([label.capitalize() for label in labels], rotation=45, ha="right")
    ax.set_yticklabels([label.capitalize() for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    max_val = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if max_val and value > max_val / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
