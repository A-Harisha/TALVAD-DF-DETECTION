from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_gbwso_convergence(curve: List[float], out_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(curve)
    plt.title("GBWSO Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Global Best Fitness")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: str | Path) -> None:
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, out_path: str | Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_attention_maps(attention: np.ndarray, out_path: str | Path, title: str = "Attention Weights") -> None:
    plt.figure(figsize=(8, 4))
    sns.heatmap(attention, cmap="viridis")
    plt.title(title)
    plt.xlabel("Component")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], out_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    for name, values in history.items():
        plt.plot(values, label=name)
    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(scores: np.ndarray, out_path: str | Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(scores)), scores)
    plt.title("Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_population_evolution(population: np.ndarray, out_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.imshow(population, aspect="auto", cmap="magma")
    plt.title("GBWSO Shark Population Evolution")
    plt.xlabel("Feature Index")
    plt.ylabel("Shark Index")
    plt.colorbar(label="Position")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
