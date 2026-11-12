"""Small helpers shared by the training scripts."""
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed):
    keras.utils.set_random_seed(seed)


def plot_history(history, path):
    """Save training and validation curves of a Keras History object."""
    metrics = [m for m in history.history if not m.startswith("val_")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        ax.plot(history.history[metric], label="train")
        if f"val_{metric}" in history.history:
            ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
