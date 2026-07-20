from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from optimization.fitness import FitnessEvaluator


@dataclass
class GBWSOState:
    positions: np.ndarray
    velocities: np.ndarray
    personal_best: np.ndarray
    personal_best_fitness: np.ndarray
    global_best: np.ndarray
    global_best_fitness: float


def _worker_eval(args) -> float:
    evaluator, x, y, mask = args
    return evaluator.evaluate(x, y, mask)


class GBWSOFeatureSelector:
    """
    Research-grade implementation enhancements preserving the original GBWSO multimodal optimization framework.
    """

    def __init__(
        self,
        n_features: int,
        n_sharks: int,
        max_iter: int,
        min_features: int,
        max_features: int,
        workers: int = 4,
        random_state: int = 42,
        checkpoint_path: str | Path | None = None,
        mask_cache_dir: str | Path | None = None,
    ) -> None:
        self.n_features = n_features
        self.n_sharks = n_sharks
        self.max_iter = max_iter
        self.min_features = min_features
        self.max_features = min(max_features, n_features)
        self.workers = workers
        self.rng = np.random.default_rng(random_state)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir else None

        if self.mask_cache_dir:
            self.mask_cache_dir.mkdir(parents=True, exist_ok=True)

    def _structured_mask(self, position: np.ndarray, groups: Dict[str, Tuple[int, int]] | None) -> np.ndarray:
        if groups is None:
            return (position > 0.5).astype(np.float32)

        mask = np.zeros_like(position, dtype=np.float32)
        for _, (start, end) in groups.items():
            sub = position[start:end]
            # 30th-percentile threshold: retain top 70% within each group.
            thresh = np.quantile(sub, 0.3)
            mask[start:end] = (sub >= thresh).astype(np.float32)
        return mask

    def _clamp_mask(self, mask: np.ndarray) -> np.ndarray:
        idx = np.where(mask > 0.5)[0]
        if len(idx) < self.min_features:
            add = self.rng.choice(np.arange(self.n_features), self.min_features - len(idx), replace=False)
            mask[add] = 1.0
            idx = np.where(mask > 0.5)[0]
        if len(idx) > self.max_features:
            keep = self.rng.choice(idx, self.max_features, replace=False)
            new_mask = np.zeros_like(mask)
            new_mask[keep] = 1.0
            return new_mask
        return mask

    def _cache_key(self, mask: np.ndarray) -> str:
        return hashlib.sha1(mask.astype(np.int8).tobytes()).hexdigest()

    def _load_cached_fitness(self, mask: np.ndarray) -> float | None:
        if self.mask_cache_dir is None:
            return None
        p = self.mask_cache_dir / f"{self._cache_key(mask)}.json"
        if not p.exists():
            return None
        return float(json.loads(p.read_text(encoding="utf-8"))["fitness"])

    def _save_cached_fitness(self, mask: np.ndarray, fitness: float) -> None:
        if self.mask_cache_dir is None:
            return
        p = self.mask_cache_dir / f"{self._cache_key(mask)}.json"
        p.write_text(json.dumps({"fitness": float(fitness)}), encoding="utf-8")

    def _initialize(self, groups: Dict[str, Tuple[int, int]] | None = None) -> GBWSOState:
        positions = self.rng.random((self.n_sharks, self.n_features))
        velocities = self.rng.uniform(-0.1, 0.1, size=(self.n_sharks, self.n_features))

        if groups is not None:
            for i in range(self.n_sharks):
                m = self._structured_mask(positions[i], groups)
                positions[i] = m

        personal_best = positions.copy()
        personal_best_fitness = np.full(self.n_sharks, -np.inf)
        global_best = positions[0].copy()
        global_best_fitness = -np.inf
        return GBWSOState(positions, velocities, personal_best, personal_best_fitness, global_best, global_best_fitness)

    def _adaptive_sigma(self, iteration: int) -> float:
        # Larger variance early (exploration), smaller later (exploitation).
        t = iteration / max(self.max_iter - 1, 1)
        return 1.0 * (1.0 - t) + 0.1 * t

    def _evaluate_population(
        self,
        state: GBWSOState,
        x: np.ndarray,
        y: np.ndarray,
        evaluator: FitnessEvaluator,
        groups: Dict[str, Tuple[int, int]] | None,
    ) -> np.ndarray:
        masks = []
        cached = []
        tasks = []
        task_idx = []

        for i in range(self.n_sharks):
            m = self._structured_mask(state.positions[i], groups)
            m = self._clamp_mask(m)
            masks.append(m)
            fit = self._load_cached_fitness(m)
            cached.append(fit)
            if fit is None:
                tasks.append((evaluator, x, y, m))
                task_idx.append(i)

        values = np.zeros(self.n_sharks, dtype=np.float32)
        for i, c in enumerate(cached):
            if c is not None:
                values[i] = c

        if tasks:
            if self.workers <= 1:
                out = [_worker_eval(t) for t in tasks]
            else:
                with ProcessPoolExecutor(max_workers=self.workers) as pool:
                    out = list(pool.map(_worker_eval, tasks))
            for idx, fit in zip(task_idx, out):
                values[idx] = fit
                self._save_cached_fitness(masks[idx], float(fit))

        cache_hits = sum(1 for c in cached if c is not None)
        return values, cache_hits

    def _save_checkpoint(self, iteration: int, state: GBWSOState, curve: List[float]) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "iteration": iteration,
            "positions": state.positions,
            "velocities": state.velocities,
            "personal_best": state.personal_best,
            "personal_best_fitness": state.personal_best_fitness,
            "global_best": state.global_best,
            "global_best_fitness": state.global_best_fitness,
            "curve": np.array(curve),
        }
        np.savez(self.checkpoint_path, **payload)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        evaluator: FitnessEvaluator,
        groups: Dict[str, Tuple[int, int]] | None = None,
    ) -> Tuple[np.ndarray, List[float]]:
        state = self._initialize(groups)
        convergence: List[float] = []

        for it in range(self.max_iter):
            fitness, cache_hits = self._evaluate_population(state, x, y, evaluator, groups)

            improved = fitness > state.personal_best_fitness
            state.personal_best[improved] = state.positions[improved]
            state.personal_best_fitness[improved] = fitness[improved]

            best_i = int(np.argmax(fitness))
            if fitness[best_i] > state.global_best_fitness:
                state.global_best_fitness = float(fitness[best_i])
                state.global_best = state.positions[best_i].copy()

            # WSO velocity update + oscillatory movement + prey tracking.
            r1 = self.rng.random((self.n_sharks, self.n_features))
            r2 = self.rng.random((self.n_sharks, self.n_features))
            omega = 0.7 + 0.3 * np.cos(2 * np.pi * (it / max(self.max_iter, 1)))
            prey = state.global_best[None, :] - state.positions
            personal = state.personal_best - state.positions

            state.velocities = (
                omega * state.velocities
                + 1.5 * r1 * personal
                + 1.2 * r2 * prey
                + 0.05 * self.rng.standard_normal((self.n_sharks, self.n_features))
            )
            state.positions = np.clip(state.positions + state.velocities, 0.0, 1.0)

            # Gaussian barebones strategy with adaptive variance.
            sigma = self._adaptive_sigma(it)
            gb_mean = 0.5 * (state.personal_best + state.global_best[None, :])
            gb_std = sigma * np.abs(state.personal_best - state.global_best[None, :])
            gb_sample = self.rng.normal(gb_mean, gb_std + 1e-6)

            use_gb = self.rng.random((self.n_sharks, self.n_features)) < 0.5
            state.positions = np.where(use_gb, gb_sample, state.positions)
            state.positions = np.clip(state.positions, 0.0, 1.0)

            convergence.append(state.global_best_fitness)
            self._save_checkpoint(it, state, convergence)
            selected = int(self._clamp_mask(self._structured_mask(state.global_best, groups)).sum())
            print(
                f"  [GBWSO] iter {it + 1:>3}/{self.max_iter}"
                f"  fitness={state.global_best_fitness:.4f}"
                f"  selected_features={selected}"
                f"  cache_hits={cache_hits}/{self.n_sharks}",
                flush=True,
            )

        best_mask = self._structured_mask(state.global_best, groups)
        best_mask = self._clamp_mask(best_mask)
        return best_mask, convergence
