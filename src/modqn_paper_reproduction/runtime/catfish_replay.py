"""Replay utilities for the Phase 04-B Catfish-MODQN opt-in path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class MixedReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    masks: np.ndarray
    next_masks: np.ndarray
    dones: np.ndarray
    composition: dict[str, Any]


def quality_score(
    reward_vector: np.ndarray | tuple[float, float, float],
    weights: tuple[float, float, float],
) -> float:
    """Return the Phase 04-B local MODQN quality score."""
    reward = np.asarray(reward_vector, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if reward.shape != (3,) or weight.shape != (3,):
        raise ValueError(
            "quality_score requires shape-(3,) rewards and weights, "
            f"got reward={reward.shape}, weights={weight.shape}"
        )
    return float(np.dot(weight, reward))


def distribution_summary(values: list[float] | tuple[float, ...]) -> dict[str, Any]:
    """Return JSON-ready scalar distribution diagnostics."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def component_distribution_summary(
    reward_vectors: list[tuple[float, float, float]],
) -> dict[str, Any]:
    """Summarize r1/r2/r3 distributions for a replay partition."""
    if not reward_vectors:
        return {
            "r1": distribution_summary([]),
            "r2": distribution_summary([]),
            "r3": distribution_summary([]),
        }
    arr = np.asarray(reward_vectors, dtype=np.float64)
    return {
        "r1": distribution_summary(arr[:, 0].tolist()),
        "r2": distribution_summary(arr[:, 1].tolist()),
        "r3": distribution_summary(arr[:, 2].tolist()),
    }


def target_catfish_sample_count(batch_size: int, catfish_ratio: float) -> int:
    """Compute the requested Catfish sample count for one mixed update."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if not 0.0 <= catfish_ratio < 1.0:
        raise ValueError(
            f"catfish_ratio must be in [0, 1), got {catfish_ratio}"
        )
    if catfish_ratio == 0.0:
        return 0
    return min(batch_size - 1, max(1, int(round(batch_size * catfish_ratio))))


def sample_mixed_replay_batch(
    *,
    main_replay: ReplayBuffer,
    catfish_replay: ReplayBuffer,
    batch_size: int,
    catfish_ratio: float,
    rng: Generator,
) -> MixedReplayBatch:
    """Sample a batch with an explicit main/Catfish composition."""
    catfish_count = target_catfish_sample_count(batch_size, catfish_ratio)
    main_count = batch_size - catfish_count
    if len(main_replay) < main_count:
        raise ValueError(
            f"main replay has {len(main_replay)} samples, needs {main_count}"
        )
    if len(catfish_replay) < catfish_count:
        raise ValueError(
            f"catfish replay has {len(catfish_replay)} samples, needs {catfish_count}"
        )

    main_batch = main_replay.sample(main_count, rng)
    if catfish_count > 0:
        catfish_batch = catfish_replay.sample(catfish_count, rng)
        stacked = tuple(
            np.concatenate([main_part, catfish_part], axis=0)
            for main_part, catfish_part in zip(main_batch, catfish_batch)
        )
        sources = np.array(
            ["main"] * main_count + ["catfish"] * catfish_count,
            dtype=object,
        )
        order = rng.permutation(batch_size)
        batch = tuple(part[order] for part in stacked)
        sources = sources[order]
    else:
        batch = main_batch
        sources = np.array(["main"] * main_count, dtype=object)

    actual_catfish_count = int(np.sum(sources == "catfish"))
    actual_main_count = int(np.sum(sources == "main"))
    composition = {
        "batch_size": int(batch_size),
        "configured_catfish_ratio": float(catfish_ratio),
        "target_catfish_sample_count": int(catfish_count),
        "actual_catfish_sample_count": actual_catfish_count,
        "actual_main_sample_count": actual_main_count,
        "actual_catfish_ratio": float(actual_catfish_count / batch_size),
        "source_counts": {
            "main": actual_main_count,
            "catfish": actual_catfish_count,
        },
    }

    return MixedReplayBatch(
        states=batch[0],
        actions=batch[1],
        rewards=batch[2],
        next_states=batch[3],
        masks=batch[4],
        next_masks=batch[5],
        dones=batch[6],
        composition=composition,
    )
