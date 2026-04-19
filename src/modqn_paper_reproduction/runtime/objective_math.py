"""Objective scalarization and reward-calibration helpers."""

from __future__ import annotations

import numpy as np

from .trainer_spec import TrainerConfig


def scalarize_objectives(
    reward_vector: np.ndarray | tuple[float, float, float],
    objective_weights: tuple[float, float, float],
) -> float:
    """Scalarize a three-objective reward vector with the given weights."""
    rewards = np.asarray(reward_vector, dtype=np.float64)
    weights = np.asarray(objective_weights, dtype=np.float64)
    if rewards.shape != (3,) or weights.shape != (3,):
        raise ValueError(
            "scalarize_objectives requires shape-(3,) rewards and weights, "
            f"got rewards={rewards.shape}, weights={weights.shape}"
        )
    return float(np.dot(weights, rewards))


def apply_reward_calibration(
    reward_vector: np.ndarray | tuple[float, float, float],
    config: TrainerConfig,
) -> np.ndarray:
    """Apply the opt-in trainer-side reward calibration transform."""
    rewards = np.asarray(reward_vector, dtype=np.float64)
    if rewards.shape != (3,):
        raise ValueError(
            "apply_reward_calibration requires a shape-(3,) reward vector, "
            f"got {rewards.shape}"
        )
    if not config.reward_calibration_enabled:
        return rewards.copy()
    if config.reward_calibration_mode == "divide-by-fixed-scales":
        scales = np.asarray(config.reward_calibration_scales, dtype=np.float64)
        return rewards / scales
    raise ValueError(
        f"Unsupported reward_calibration_mode={config.reward_calibration_mode!r}"
    )
