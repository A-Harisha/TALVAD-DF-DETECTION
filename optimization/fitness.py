from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class FitnessWeights:
    accuracy: float = 0.3
    f1: float = 0.4
    auc: float = 0.3


class FitnessEvaluator:
    def __init__(self, k_folds: int = 5, random_state: int = 42, weights: Dict | None = None) -> None:
        self.k_folds = k_folds
        self.random_state = random_state
        if weights is None:
            self.weights = FitnessWeights()
        else:
            self.weights = FitnessWeights(**weights)

    def evaluate(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        idx = np.where(mask > 0.5)[0]
        if len(idx) < 2 or len(np.unique(y)) < 2:
            return 0.0

        xs = x[:, idx]
        class_counts = np.bincount(y.astype(int))
        effective_folds = min(self.k_folds, int(class_counts.min())) if len(class_counts) > 1 else 0
        if effective_folds < 2:
            return 0.0

        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=self.random_state)

        accs, f1s, aucs = [], [], []
        for tr, va in skf.split(xs, y):
            model = RandomForestClassifier(
                n_estimators=25, max_depth=6, class_weight="balanced",
                n_jobs=1, random_state=self.random_state,
            )
            model.fit(xs[tr], y[tr])
            pred = model.predict(xs[va])
            prob = model.predict_proba(xs[va])[:, 1]

            accs.append(accuracy_score(y[va], pred))
            f1s.append(f1_score(y[va], pred, zero_division=0))
            try:
                aucs.append(roc_auc_score(y[va], prob))
            except ValueError:
                aucs.append(0.5)

        acc = float(np.mean(accs))
        f1 = float(np.mean(f1s))
        auc = float(np.mean(aucs))

        return self.weights.accuracy * acc + self.weights.f1 * f1 + self.weights.auc * auc
