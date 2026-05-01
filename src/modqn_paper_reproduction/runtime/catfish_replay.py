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


def r2_distribution_summary(rewards: np.ndarray) -> dict[str, Any]:
    """Summarize r2 values in a sampled replay slice."""
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.size == 0:
        summary = distribution_summary([])
        summary["negative_count"] = 0
        summary["negative_share"] = None
        return summary
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"rewards must have shape (N, 3), got {arr.shape}")
    r2_values = arr[:, 1]
    negative_count = int(np.sum(r2_values < 0.0))
    summary = distribution_summary(r2_values.tolist())
    summary["negative_count"] = negative_count
    summary["negative_share"] = float(negative_count / max(r2_values.size, 1))
    return summary


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


def target_source_sample_counts(
    *,
    batch_size: int,
    source_ratios: dict[str, float],
) -> dict[str, int]:
    """Compute deterministic integer counts for several replay sources.

    Counts use a largest-remainder allocation so the total source count matches
    the rounded total Catfish budget while preserving stable tie handling.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if not source_ratios:
        return {}
    if any(value < 0.0 for value in source_ratios.values()):
        raise ValueError(f"source ratios must be >= 0, got {source_ratios!r}")

    total_ratio = float(sum(source_ratios.values()))
    if not 0.0 <= total_ratio < 1.0:
        raise ValueError(
            f"sum(source_ratios) must be in [0, 1), got {total_ratio}"
        )
    if total_ratio == 0.0:
        return {name: 0 for name in source_ratios}

    target_total = min(batch_size - 1, max(1, int(round(batch_size * total_ratio))))
    ideals = {
        name: float(ratio) * float(batch_size)
        for name, ratio in source_ratios.items()
    }
    counts = {name: int(np.floor(value)) for name, value in ideals.items()}
    remaining = target_total - int(sum(counts.values()))
    if remaining > 0:
        order = sorted(
            source_ratios,
            key=lambda name: (-(ideals[name] - counts[name]), name),
        )
        for name in order[:remaining]:
            counts[name] += 1
    elif remaining < 0:
        order = sorted(
            source_ratios,
            key=lambda name: ((ideals[name] - counts[name]), name),
        )
        for name in order[: -remaining]:
            if counts[name] > 0:
                counts[name] -= 1

    return counts


def _stack_and_shuffle_sources(
    *,
    sampled_parts: list[tuple[np.ndarray, ...]],
    sources: list[str],
    batch_size: int,
    rng: Generator,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    stacked = tuple(
        np.concatenate([part[idx] for part in sampled_parts], axis=0)
        for idx in range(7)
    )
    source_array = np.array(sources, dtype=object)
    order = rng.permutation(batch_size)
    batch = tuple(part[order] for part in stacked)
    source_array = source_array[order]
    return batch, source_array


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
        batch, sources = _stack_and_shuffle_sources(
            sampled_parts=[main_batch, catfish_batch],
            sources=["main"] * main_count + ["catfish"] * catfish_count,
            batch_size=batch_size,
            rng=rng,
        )
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
        "matched_main_batch_r2_distribution": r2_distribution_summary(
            main_batch[2]
        ),
        "injected_batch_r2_distribution": r2_distribution_summary(
            catfish_batch[2] if catfish_count > 0 else np.empty((0, 3))
        ),
        "r2_guard": {
            "enabled": False,
            "passed": None,
            "reason": "unguarded-catfish-mixed-replay",
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


def sample_equal_budget_random_replay_batch(
    *,
    main_replay: ReplayBuffer,
    batch_size: int,
    injected_ratio: float,
    rng: Generator,
) -> MixedReplayBatch:
    """Sample a random-control batch with the same injected replay budget."""
    injected_count = target_catfish_sample_count(batch_size, injected_ratio)
    main_count = batch_size - injected_count
    if len(main_replay) < batch_size:
        raise ValueError(
            f"main replay has {len(main_replay)} samples, needs {batch_size}"
        )

    batch = main_replay.sample(batch_size, rng)
    sources = np.array(
        ["main"] * main_count + ["random-control"] * injected_count,
        dtype=object,
    )
    order = rng.permutation(batch_size)
    batch = tuple(part[order] for part in batch)
    sources = sources[order]

    actual_random_count = int(np.sum(sources == "random-control"))
    actual_main_count = int(np.sum(sources == "main"))
    composition = {
        "batch_size": int(batch_size),
        "configured_catfish_ratio": float(injected_ratio),
        "configured_injected_ratio": float(injected_ratio),
        "source_mode": "random-main-replay",
        "target_catfish_sample_count": int(injected_count),
        "target_injected_sample_count": int(injected_count),
        "actual_catfish_sample_count": actual_random_count,
        "actual_random_control_sample_count": actual_random_count,
        "actual_main_sample_count": actual_main_count,
        "actual_catfish_ratio": float(actual_random_count / batch_size),
        "actual_injected_ratio": float(actual_random_count / batch_size),
        "source_counts": {
            "main": actual_main_count,
            "catfish": 0,
            "random-control": actual_random_count,
        },
        "matched_main_batch_r2_distribution": r2_distribution_summary(
            batch[2][sources == "main"]
        ),
        "injected_batch_r2_distribution": r2_distribution_summary(
            batch[2][sources == "random-control"]
        ),
        "r2_guard": {
            "enabled": False,
            "passed": None,
            "reason": "random-equal-budget-control",
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


def sample_r2_guarded_mixed_replay_batch(
    *,
    main_replay: ReplayBuffer,
    catfish_replay: ReplayBuffer,
    batch_size: int,
    catfish_ratio: float,
    rng: Generator,
    max_attempts: int,
    strict_no_handover_samples: bool = False,
) -> tuple[MixedReplayBatch | None, dict[str, Any]]:
    """Sample a guarded Catfish batch or return a skip diagnostic.

    The guard enforces the Phase 07-D non-silent rule: injected Catfish r2<0
    share must not exceed the matched main-replay r2<0 share. If no candidate
    passes within the configured attempt budget, callers must skip Catfish
    injection rather than falling back to unguarded replay.
    """
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be > 0, got {max_attempts}")

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
    main_r2 = r2_distribution_summary(main_batch[2])
    main_negative_share = float(main_r2["negative_share"] or 0.0)
    max_catfish_negative_count = (
        0
        if strict_no_handover_samples
        else int(np.floor(main_negative_share * catfish_count + 1e-12))
    )

    best_guard: dict[str, Any] | None = None
    selected_catfish_batch: tuple[np.ndarray, ...] | None = None
    for attempt in range(1, max_attempts + 1):
        catfish_batch = catfish_replay.sample(catfish_count, rng)
        catfish_r2 = r2_distribution_summary(catfish_batch[2])
        catfish_negative_count = int(catfish_r2["negative_count"])
        guard = {
            "enabled": True,
            "passed": catfish_negative_count <= max_catfish_negative_count,
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "strict_no_handover_samples": bool(strict_no_handover_samples),
            "matched_main_r2_negative_share": main_negative_share,
            "matched_main_r2_negative_count": int(main_r2["negative_count"]),
            "catfish_r2_negative_share": float(
                catfish_r2["negative_share"] or 0.0
            ),
            "catfish_r2_negative_count": catfish_negative_count,
            "max_allowed_catfish_r2_negative_count": int(
                max_catfish_negative_count
            ),
            "reason": None,
        }
        if best_guard is None or catfish_negative_count < int(
            best_guard["catfish_r2_negative_count"]
        ):
            best_guard = dict(guard)
        if guard["passed"]:
            selected_catfish_batch = catfish_batch
            best_guard = guard
            break

    if selected_catfish_batch is None:
        skip_guard = dict(best_guard or {})
        skip_guard.update(
            {
                "enabled": True,
                "passed": False,
                "max_attempts": int(max_attempts),
                "strict_no_handover_samples": bool(strict_no_handover_samples),
                "matched_main_r2_negative_share": main_negative_share,
                "matched_main_r2_negative_count": int(main_r2["negative_count"]),
                "max_allowed_catfish_r2_negative_count": int(
                    max_catfish_negative_count
                ),
                "reason": "r2-batch-share-exceeds-main-replay",
            }
        )
        return None, skip_guard

    batch, sources = _stack_and_shuffle_sources(
        sampled_parts=[main_batch, selected_catfish_batch],
        sources=["main"] * main_count + ["catfish"] * catfish_count,
        batch_size=batch_size,
        rng=rng,
    )
    actual_catfish_count = int(np.sum(sources == "catfish"))
    actual_main_count = int(np.sum(sources == "main"))
    catfish_r2 = r2_distribution_summary(selected_catfish_batch[2])
    composition = {
        "batch_size": int(batch_size),
        "configured_catfish_ratio": float(catfish_ratio),
        "target_catfish_sample_count": int(catfish_count),
        "actual_catfish_sample_count": actual_catfish_count,
        "actual_main_sample_count": actual_main_count,
        "actual_catfish_ratio": float(actual_catfish_count / batch_size),
        "source_mode": "catfish-replay",
        "source_counts": {
            "main": actual_main_count,
            "catfish": actual_catfish_count,
            "random-control": 0,
        },
        "matched_main_batch_r2_distribution": main_r2,
        "injected_batch_r2_distribution": catfish_r2,
        "r2_guard": best_guard,
    }

    return (
        MixedReplayBatch(
            states=batch[0],
            actions=batch[1],
            rewards=batch[2],
            next_states=batch[3],
            masks=batch[4],
            next_masks=batch[5],
            dones=batch[6],
            composition=composition,
        ),
        best_guard,
    )


def sample_multi_source_replay_batch(
    *,
    main_replay: ReplayBuffer,
    source_replays: dict[str, ReplayBuffer],
    batch_size: int,
    source_ratios: dict[str, float],
    rng: Generator,
) -> MixedReplayBatch:
    """Sample a batch from main replay plus several labeled Catfish sources."""
    source_counts = target_source_sample_counts(
        batch_size=batch_size,
        source_ratios=source_ratios,
    )
    catfish_count = int(sum(source_counts.values()))
    main_count = batch_size - catfish_count
    if len(main_replay) < main_count:
        raise ValueError(
            f"main replay has {len(main_replay)} samples, needs {main_count}"
        )
    for name, count in source_counts.items():
        if len(source_replays[name]) < count:
            raise ValueError(
                f"{name} replay has {len(source_replays[name])} samples, "
                f"needs {count}"
            )

    sampled_parts = [main_replay.sample(main_count, rng)]
    sources = ["main"] * main_count
    for name, count in source_counts.items():
        if count <= 0:
            continue
        sampled_parts.append(source_replays[name].sample(count, rng))
        sources.extend([name] * count)

    stacked = tuple(
        np.concatenate([part[idx] for part in sampled_parts], axis=0)
        for idx in range(7)
    )
    source_array = np.array(sources, dtype=object)
    order = rng.permutation(batch_size)
    batch = tuple(part[order] for part in stacked)
    source_array = source_array[order]

    actual_source_counts = {
        "main": int(np.sum(source_array == "main")),
        **{
            name: int(np.sum(source_array == name))
            for name in source_ratios
        },
    }
    actual_catfish_count = int(sum(actual_source_counts[name] for name in source_ratios))
    composition = {
        "batch_size": int(batch_size),
        "configured_source_ratios": {
            name: float(ratio) for name, ratio in source_ratios.items()
        },
        "configured_catfish_ratio": float(sum(source_ratios.values())),
        "target_source_sample_counts": {
            name: int(count) for name, count in source_counts.items()
        },
        "target_catfish_sample_count": int(catfish_count),
        "actual_source_sample_counts": actual_source_counts,
        "actual_catfish_sample_count": actual_catfish_count,
        "actual_main_sample_count": int(actual_source_counts["main"]),
        "actual_catfish_ratio": float(actual_catfish_count / batch_size),
        "source_counts": actual_source_counts,
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
