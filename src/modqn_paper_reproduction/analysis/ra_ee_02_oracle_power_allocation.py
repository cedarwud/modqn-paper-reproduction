"""RA-EE-02 offline oracle / heuristic power-allocation audit.

This module fixes association trajectories and rescores them under bounded
beam-power decisions. It does not train, does not invoke Catfish, and does not
mutate the frozen MODQN baseline. The method label is scoped to the
RA-EE-MDP / RA-EE-MODQN extension family only.
"""

from __future__ import annotations

import copy
import csv
import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import read_run_metadata
from ..config_loader import (
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from ..env.step import HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
from ._common import write_json
from .phase03a_diagnostics import distribution, select_counterfactual_actions


DEFAULT_PHASE03C_C_LEARNED_RUN_DIR = (
    "artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot"
)

COUNTERFACTUAL_POLICIES = (
    "hold-current",
    "random-valid",
    "spread-valid",
)

POWER_CANDIDATES = (
    "fixed-control",
    "load-concave",
    "budget-trim",
    "qos-tail-boost",
    "constrained-oracle",
)

_POLICY_SEED_OFFSETS = {
    "hold-current": 1009,
    "random-valid": 2003,
    "spread-valid": 4001,
}


@dataclass(frozen=True)
class _AuditSettings:
    method_label: str
    codebook_levels_w: tuple[float, ...]
    fixed_control_power_w: float
    total_power_budget_w: float
    per_beam_max_power_w: float
    active_base_power_w: float
    load_scale_power_w: float
    load_exponent: float
    p05_min_ratio_vs_control: float
    served_ratio_min_delta_vs_control: float
    outage_ratio_max_delta_vs_control: float
    oracle_max_demoted_beams: int


@dataclass(frozen=True)
class _StepSnapshot:
    trajectory_policy: str
    evaluation_seed: int
    step_index: int
    assignments: np.ndarray
    active_mask: np.ndarray
    beam_loads: np.ndarray
    unit_snr_by_user: np.ndarray
    bandwidth_hz: float
    handover_count: int
    r2_mean: float


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values.tolist())


def _categorical_distribution(values: list[str]) -> dict[str, Any]:
    counts = Counter(str(value) for value in values)
    return {
        "count": int(sum(counts.values())),
        "distinct": sorted(counts),
        "distinct_count": int(len(counts)),
        "histogram": {
            key: int(value)
            for key, value in sorted(counts.items(), key=lambda item: item[0])
        },
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _correlation(x_values: list[float], y_values: list[float]) -> dict[str, float | None]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"pearson": None, "spearman": None}
    pearson = None
    if float(np.std(x)) != 0.0 and float(np.std(y)) != 0.0:
        pearson = float(np.corrcoef(x, y)[0, 1])
    rx = _rank(x)
    ry = _rank(y)
    spearman = None
    if float(np.std(rx)) != 0.0 and float(np.std(ry)) != 0.0:
        spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=np.float64)))


def _pct_delta(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or abs(float(reference)) < 1e-12:
        return None
    return float((float(value) - float(reference)) / abs(float(reference)))


def _unique_float_values(values: list[float], *, places: int = 12) -> list[float]:
    return sorted({round(float(value), places) for value in values})


def _ra_ee_02_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("ra_ee_02_oracle_power_allocation_audit", {})
        .get("value", {})
    )
    return value if isinstance(value, dict) else {}


def _power_surface_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("hobs_power_surface", {})
        .get("value", {})
    )
    return value if isinstance(value, dict) else {}


def _settings_from_config(cfg: dict[str, Any]) -> _AuditSettings:
    audit = _ra_ee_02_value(cfg)
    power = _power_surface_value(cfg)
    levels_raw = audit.get(
        "codebook_levels_w",
        power.get("power_codebook_levels_w", [0.5, 0.75, 1.0, 1.5, 2.0]),
    )
    levels = tuple(float(level) for level in levels_raw)
    if not levels:
        raise ValueError("RA-EE-02 requires at least one codebook level.")
    if tuple(sorted(levels)) != levels:
        raise ValueError(f"RA-EE-02 codebook levels must be sorted, got {levels!r}.")

    return _AuditSettings(
        method_label=str(audit.get("method_label", "RA-EE-MDP / RA-EE-MODQN")),
        codebook_levels_w=levels,
        fixed_control_power_w=float(audit.get("fixed_control_power_w", 1.0)),
        total_power_budget_w=float(
            audit.get("total_power_budget_w", power.get("total_power_budget_w", 8.0))
        ),
        per_beam_max_power_w=float(
            audit.get("per_beam_max_power_w", power.get("max_power_w", 2.0))
        ),
        active_base_power_w=float(
            audit.get("active_base_power_w", power.get("active_base_power_w", 0.25))
        ),
        load_scale_power_w=float(
            audit.get("load_scale_power_w", power.get("load_scale_power_w", 0.35))
        ),
        load_exponent=float(
            audit.get("load_exponent", power.get("load_exponent", 0.5))
        ),
        p05_min_ratio_vs_control=float(
            audit.get("p05_throughput_min_ratio_vs_control", 0.95)
        ),
        served_ratio_min_delta_vs_control=float(
            audit.get("served_ratio_min_delta_vs_control", 0.0)
        ),
        outage_ratio_max_delta_vs_control=float(
            audit.get("outage_ratio_max_delta_vs_control", 0.0)
        ),
        oracle_max_demoted_beams=int(audit.get("oracle_max_demoted_beams", 3)),
    )


def _cfg_for_unit_power_replay(base_cfg: dict[str, Any], settings: _AuditSettings) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    resolved = cfg.setdefault("resolved_assumptions", {})
    block = resolved.setdefault("hobs_power_surface", {})
    block["assumption_id"] = "ASSUME-MODQN-REP-026"
    block["value"] = {
        "mode": HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
        "inactive_beam_policy": "zero-w",
        "power_codebook_profile": "fixed-mid",
        "power_codebook_levels_w": [settings.fixed_control_power_w],
        "total_power_budget_w": settings.total_power_budget_w,
        "active_base_power_w": settings.active_base_power_w,
        "load_scale_power_w": settings.load_scale_power_w,
        "load_exponent": settings.load_exponent,
        "max_power_w": settings.per_beam_max_power_w,
        "units": "linear-w",
        "provenance": (
            "RA-EE-02 unit-power replay surface. It is used only to recover "
            "fixed-trajectory per-user unit-SINR for offline power rescoring."
        ),
    }
    return cfg


def _checkpoint_path(metadata: Any) -> tuple[str, Path] | None:
    files = metadata.checkpoint_files
    if files.secondary_best_eval is not None:
        return "best-eval", Path(files.secondary_best_eval)
    if files.primary_final is not None:
        return "final", Path(files.primary_final)
    return None


def _rollout_learned_trajectory(
    *,
    run_dir: Path,
    evaluation_seed_set: tuple[int, ...],
    max_steps: int | None,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, Any]]:
    metadata = read_run_metadata(run_dir / "run_metadata.json")
    checkpoint = _checkpoint_path(metadata)
    if checkpoint is None:
        raise ValueError(f"No checkpoint is available in learned run {run_dir}.")
    checkpoint_role, checkpoint_path = checkpoint
    cfg = metadata.resolved_config_snapshot
    env = build_environment(cfg)
    trainer_cfg = build_trainer_config(cfg)
    trainer = MODQNTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=metadata.seeds.train_seed,
        env_seed=metadata.seeds.environment_seed,
        mobility_seed=metadata.seeds.mobility_seed,
    )
    payload = trainer.load_checkpoint(checkpoint_path, load_optimizers=False)
    policy_label = f"phase03c-c-candidate-{checkpoint_role}"

    trajectories: dict[str, dict[int, list[np.ndarray]]] = {
        policy_label: defaultdict(list)
    }
    for eval_seed in evaluation_seed_set:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)
        encoded = trainer.encode_states(states)
        steps_seen = 0
        while True:
            if max_steps is not None and steps_seen >= max_steps:
                break
            actions = trainer.select_actions(encoded, masks, eps=0.0)
            result = trainer.env.step(actions, env_rng)
            trajectories[policy_label][int(eval_seed)].append(actions.copy())
            steps_seen += 1
            if result.done:
                break
            encoded = trainer.encode_states(result.user_states)
            masks = result.action_masks

    return trajectories, {
        "available": True,
        "run_dir": str(run_dir),
        "policy_label": policy_label,
        "checkpoint_role": checkpoint_role,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_kind": str(payload.get("checkpoint_kind")),
        "checkpoint_episode": int(payload.get("episode")),
    }


def _rollout_counterfactual_trajectories(
    *,
    cfg: dict[str, Any],
    evaluation_seed_set: tuple[int, ...],
    max_steps: int | None,
    policies: tuple[str, ...],
) -> dict[str, dict[int, list[np.ndarray]]]:
    trajectories: dict[str, dict[int, list[np.ndarray]]] = {}
    for policy in policies:
        if policy not in COUNTERFACTUAL_POLICIES:
            raise ValueError(
                f"Unsupported RA-EE-02 trajectory policy {policy!r}; "
                f"supported values are {COUNTERFACTUAL_POLICIES!r}."
            )
        env = build_environment(cfg)
        policy_rows: dict[int, list[np.ndarray]] = defaultdict(list)
        for eval_seed in evaluation_seed_set:
            env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
            env_rng = np.random.default_rng(env_seed_seq)
            mobility_rng = np.random.default_rng(mobility_seed_seq)
            policy_rng = np.random.default_rng(
                int(eval_seed) + _POLICY_SEED_OFFSETS[policy]
            )
            _states, masks, _diag = env.reset(env_rng, mobility_rng)
            steps_seen = 0
            while True:
                if max_steps is not None and steps_seen >= max_steps:
                    break
                current_assignments = env.current_assignments()
                internal_policy = (
                    "spread-valid-heuristic"
                    if policy == "spread-valid"
                    else policy
                )
                actions, _diagnostics = select_counterfactual_actions(
                    internal_policy,
                    current_assignments=current_assignments,
                    masks=masks,
                    rng=policy_rng,
                )
                result = env.step(actions, env_rng)
                policy_rows[int(eval_seed)].append(actions.copy())
                steps_seen += 1
                if result.done:
                    break
                masks = result.action_masks
        trajectories[policy] = policy_rows
    return trajectories


def _unit_snr_from_throughput(
    *,
    throughput_bps: float,
    load: float,
    bandwidth_hz: float,
) -> float:
    if throughput_bps <= 0.0 or load <= 0.0 or bandwidth_hz <= 0.0:
        return 0.0
    exponent = float(throughput_bps) * float(load) / float(bandwidth_hz)
    return float(math.exp(math.log(2.0) * exponent) - 1.0)


def _build_unit_power_snapshots(
    *,
    base_cfg: dict[str, Any],
    settings: _AuditSettings,
    trajectories: dict[str, dict[int, list[np.ndarray]]],
) -> list[_StepSnapshot]:
    cfg = _cfg_for_unit_power_replay(base_cfg, settings)
    env = build_environment(cfg)
    snapshots: list[_StepSnapshot] = []
    for policy_label, by_seed in trajectories.items():
        for eval_seed, action_rows in by_seed.items():
            env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
            env_rng = np.random.default_rng(env_seed_seq)
            mobility_rng = np.random.default_rng(mobility_seed_seq)
            env.reset(env_rng, mobility_rng)
            for actions in action_rows:
                result = env.step(actions, env_rng)
                beam_loads = result.user_states[0].beam_loads.astype(
                    np.float64, copy=True
                )
                unit_snr = np.zeros(len(actions), dtype=np.float64)
                for uid, reward in enumerate(result.rewards):
                    beam_idx = int(actions[uid])
                    load = (
                        float(beam_loads[beam_idx])
                        if 0 <= beam_idx < beam_loads.size
                        else 0.0
                    )
                    unit_snr[uid] = _unit_snr_from_throughput(
                        throughput_bps=float(reward.r1_throughput),
                        load=max(load, 1.0),
                        bandwidth_hz=env.channel_config.bandwidth_hz,
                    )
                snapshots.append(
                    _StepSnapshot(
                        trajectory_policy=policy_label,
                        evaluation_seed=int(eval_seed),
                        step_index=int(result.step_index),
                        assignments=actions.astype(np.int32, copy=True),
                        active_mask=result.active_beam_mask.astype(bool, copy=True),
                        beam_loads=beam_loads,
                        unit_snr_by_user=unit_snr,
                        bandwidth_hz=float(env.channel_config.bandwidth_hz),
                        handover_count=int(
                            sum(1 for reward in result.rewards if reward.r2_handover < 0.0)
                        ),
                        r2_mean=float(
                            np.mean([float(reward.r2_handover) for reward in result.rewards])
                        ),
                    )
                )
                if result.done:
                    break
    return snapshots


def _level_at_or_below(levels: tuple[float, ...], target: float) -> float:
    candidates = [float(level) for level in levels if float(level) <= float(target)]
    return max(candidates) if candidates else float(levels[0])


def _level_at_or_above(levels: tuple[float, ...], target: float) -> float:
    candidates = [float(level) for level in levels if float(level) >= float(target)]
    return min(candidates) if candidates else float(levels[-1])


def _base_power_vector(snapshot: _StepSnapshot, value: float) -> np.ndarray:
    powers = np.zeros(snapshot.beam_loads.shape, dtype=np.float64)
    powers[snapshot.active_mask] = float(value)
    return powers


def _budget_trim_power_vector(snapshot: _StepSnapshot, settings: _AuditSettings) -> np.ndarray:
    powers = _base_power_vector(snapshot, settings.codebook_levels_w[-1])
    active_indices = np.flatnonzero(snapshot.active_mask)
    if float(np.sum(powers[active_indices], dtype=np.float64)) <= settings.total_power_budget_w:
        return powers

    level_indices = {
        int(idx): len(settings.codebook_levels_w) - 1
        for idx in active_indices.tolist()
    }
    demotion_order = sorted(
        (int(idx) for idx in active_indices.tolist()),
        key=lambda idx: (float(snapshot.beam_loads[idx]), idx),
    )
    while float(np.sum(powers[active_indices], dtype=np.float64)) > settings.total_power_budget_w:
        changed = False
        for idx in demotion_order:
            if level_indices[idx] <= 0:
                continue
            level_indices[idx] -= 1
            powers[idx] = float(settings.codebook_levels_w[level_indices[idx]])
            changed = True
            if float(np.sum(powers[active_indices], dtype=np.float64)) <= settings.total_power_budget_w:
                return powers
        if not changed:
            return powers
    return powers


def _load_concave_power_vector(snapshot: _StepSnapshot, settings: _AuditSettings) -> np.ndarray:
    powers = np.zeros(snapshot.beam_loads.shape, dtype=np.float64)
    active_indices = np.flatnonzero(snapshot.active_mask)
    if active_indices.size == 0:
        return powers
    active_loads = snapshot.beam_loads[active_indices]
    raw = (
        settings.active_base_power_w
        + settings.load_scale_power_w * np.power(active_loads, settings.load_exponent)
    )
    raw = np.minimum(raw, settings.per_beam_max_power_w)
    quantized = [
        min(
            settings.codebook_levels_w,
            key=lambda level: (abs(float(level) - float(value)), float(level)),
        )
        for value in raw
    ]
    powers[active_indices] = np.asarray(quantized, dtype=np.float64)
    return powers


def _qos_tail_boost_power_vector(snapshot: _StepSnapshot, settings: _AuditSettings) -> np.ndarray:
    powers = np.zeros(snapshot.beam_loads.shape, dtype=np.float64)
    active_indices = np.flatnonzero(snapshot.active_mask)
    if active_indices.size == 0:
        return powers
    low = _level_at_or_below(settings.codebook_levels_w, settings.fixed_control_power_w)
    mid = _level_at_or_above(settings.codebook_levels_w, settings.fixed_control_power_w)
    high = float(settings.codebook_levels_w[-1])
    active_loads = snapshot.beam_loads[active_indices]
    active_power = np.full(active_indices.shape, mid, dtype=np.float64)
    if active_indices.size == 1:
        active_power[0] = high
    else:
        tail_threshold = float(np.percentile(active_loads, 75))
        light_threshold = float(np.percentile(active_loads, 25))
        active_power[active_loads >= tail_threshold] = high
        active_power[active_loads <= light_threshold] = low
    powers[active_indices] = active_power
    return powers


def _power_vector_for_candidate(
    snapshot: _StepSnapshot,
    settings: _AuditSettings,
    candidate: str,
) -> np.ndarray:
    if candidate == "fixed-control":
        return _base_power_vector(snapshot, settings.fixed_control_power_w)
    if candidate == "load-concave":
        return _load_concave_power_vector(snapshot, settings)
    if candidate == "budget-trim":
        return _budget_trim_power_vector(snapshot, settings)
    if candidate == "qos-tail-boost":
        return _qos_tail_boost_power_vector(snapshot, settings)
    raise ValueError(f"Unsupported direct RA-EE-02 power candidate {candidate!r}.")


def _profile_signature(power_vector: np.ndarray, active_mask: np.ndarray) -> str:
    active = power_vector[active_mask].astype(np.float64, copy=False)
    counts = Counter(round(float(value), 6) for value in active.tolist())
    parts = [f"{value:g}Wx{count}" for value, count in sorted(counts.items())]
    return "oracle:" + ",".join(parts)


def _candidate_power_vectors_for_oracle(
    snapshot: _StepSnapshot,
    settings: _AuditSettings,
) -> list[tuple[str, np.ndarray]]:
    active_indices = [int(idx) for idx in np.flatnonzero(snapshot.active_mask).tolist()]
    vectors: list[tuple[str, np.ndarray]] = [
        ("fixed-control", _power_vector_for_candidate(snapshot, settings, "fixed-control")),
        ("load-concave", _power_vector_for_candidate(snapshot, settings, "load-concave")),
        ("budget-trim", _power_vector_for_candidate(snapshot, settings, "budget-trim")),
        ("qos-tail-boost", _power_vector_for_candidate(snapshot, settings, "qos-tail-boost")),
    ]
    for level in settings.codebook_levels_w:
        vectors.append((f"fixed-{float(level):g}w", _base_power_vector(snapshot, float(level))))

    lower_levels = [
        float(level)
        for level in settings.codebook_levels_w
        if float(level) < settings.fixed_control_power_w
    ]
    if active_indices and lower_levels and settings.oracle_max_demoted_beams > 0:
        max_demoted = min(settings.oracle_max_demoted_beams, len(active_indices))
        for size in range(1, max_demoted + 1):
            for beam_subset in itertools.combinations(active_indices, size):
                for demoted_levels in itertools.product(lower_levels, repeat=size):
                    powers = _base_power_vector(snapshot, settings.fixed_control_power_w)
                    for beam_idx, level in zip(beam_subset, demoted_levels):
                        powers[beam_idx] = float(level)
                    vectors.append((_profile_signature(powers, snapshot.active_mask), powers))

    unique: list[tuple[str, np.ndarray]] = []
    seen_vectors: set[str] = set()
    for label, powers in vectors:
        key = _format_vector(powers)
        if key in seen_vectors:
            continue
        seen_vectors.add(key)
        unique.append((label, powers))
    return unique


def _compute_user_throughputs_from_power(
    snapshot: _StepSnapshot,
    power_vector: np.ndarray,
) -> np.ndarray:
    throughputs = np.zeros(snapshot.assignments.shape, dtype=np.float64)
    for uid, beam_idx_raw in enumerate(snapshot.assignments.tolist()):
        beam_idx = int(beam_idx_raw)
        if beam_idx < 0 or beam_idx >= snapshot.beam_loads.size:
            continue
        load = max(float(snapshot.beam_loads[beam_idx]), 1.0)
        power_w = max(float(power_vector[beam_idx]), 0.0)
        snr = float(snapshot.unit_snr_by_user[uid]) * power_w
        throughputs[uid] = (snapshot.bandwidth_hz / load) * math.log2(1.0 + snr)
    return throughputs


def _evaluate_power_vector(
    *,
    snapshot: _StepSnapshot,
    power_semantics: str,
    selected_power_profile: str,
    power_vector: np.ndarray,
    settings: _AuditSettings,
) -> dict[str, Any]:
    active_mask = snapshot.active_mask.astype(bool, copy=False)
    inactive_nonzero = bool(np.any(power_vector[~active_mask] > 1e-12))
    per_beam_excess = max(0.0, float(np.max(power_vector[active_mask], initial=0.0)) - settings.per_beam_max_power_w)
    total_active_power = float(np.sum(power_vector[active_mask], dtype=np.float64))
    budget_excess = max(0.0, total_active_power - settings.total_power_budget_w)
    user_throughputs = _compute_user_throughputs_from_power(snapshot, power_vector)
    throughput_sum = float(np.sum(user_throughputs, dtype=np.float64))
    ee_system = None if total_active_power <= 0.0 else throughput_sum / total_active_power
    served_count = int(np.sum(user_throughputs > 0.0))
    outage_count = int(user_throughputs.size - served_count)

    beam_throughputs = np.zeros(snapshot.beam_loads.shape, dtype=np.float64)
    for uid, beam_idx_raw in enumerate(snapshot.assignments.tolist()):
        beam_idx = int(beam_idx_raw)
        if 0 <= beam_idx < beam_throughputs.size:
            beam_throughputs[beam_idx] += float(user_throughputs[uid])
    active_beam_thr = beam_throughputs[active_mask]
    load_balance_gap = (
        0.0
        if active_beam_thr.size < 2
        else float(np.max(active_beam_thr) - np.min(active_beam_thr))
    )

    return {
        "method_label": settings.method_label,
        "trajectory_policy": snapshot.trajectory_policy,
        "power_semantics": power_semantics,
        "selected_power_profile": selected_power_profile,
        "evaluation_seed": int(snapshot.evaluation_seed),
        "step_index": int(snapshot.step_index),
        "active_beam_count": int(np.count_nonzero(active_mask)),
        "active_beam_mask": " ".join("1" if value else "0" for value in active_mask),
        "beam_loads": _format_vector(snapshot.beam_loads),
        "beam_transmit_power_w": _format_vector(power_vector),
        "inactive_beam_nonzero_power": inactive_nonzero,
        "per_beam_max_power_w": settings.per_beam_max_power_w,
        "per_beam_power_violation": bool(per_beam_excess > 1e-12),
        "per_beam_power_excess_w": per_beam_excess,
        "total_power_budget_w": settings.total_power_budget_w,
        "total_active_beam_power_w": total_active_power,
        "budget_violation": bool(budget_excess > 1e-12),
        "budget_excess_w": budget_excess,
        "sum_user_throughput_bps": throughput_sum,
        "throughput_mean_user_step_bps": throughput_sum / max(user_throughputs.size, 1),
        "throughput_p05_user_step_bps": float(np.percentile(user_throughputs, 5)),
        "EE_system_bps_per_w": ee_system,
        "served_count": served_count,
        "outage_count": outage_count,
        "served_ratio": served_count / max(user_throughputs.size, 1),
        "outage_ratio": outage_count / max(user_throughputs.size, 1),
        "handover_count": int(snapshot.handover_count),
        "r2_mean": float(snapshot.r2_mean),
        "active_beam_throughput_gap_bps": load_balance_gap,
        "_user_throughputs": user_throughputs,
    }


def _qos_guardrails_pass(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
    settings: _AuditSettings,
) -> bool:
    return (
        float(candidate["throughput_p05_user_step_bps"])
        >= settings.p05_min_ratio_vs_control
        * float(control["throughput_p05_user_step_bps"])
        and float(candidate["served_ratio"])
        >= float(control["served_ratio"]) + settings.served_ratio_min_delta_vs_control
        and float(candidate["outage_ratio"])
        <= float(control["outage_ratio"]) + settings.outage_ratio_max_delta_vs_control
    )


def _select_oracle_step(
    *,
    snapshot: _StepSnapshot,
    control_row: dict[str, Any],
    settings: _AuditSettings,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for profile, powers in _candidate_power_vectors_for_oracle(snapshot, settings):
        row = _evaluate_power_vector(
            snapshot=snapshot,
            power_semantics="constrained-oracle",
            selected_power_profile=profile,
            power_vector=powers,
            settings=settings,
        )
        constraints_ok = (
            not bool(row["budget_violation"])
            and not bool(row["per_beam_power_violation"])
            and not bool(row["inactive_beam_nonzero_power"])
            and _qos_guardrails_pass(candidate=row, control=control_row, settings=settings)
        )
        if not constraints_ok:
            continue
        if best is None:
            best = row
            continue
        row_ee = -math.inf if row["EE_system_bps_per_w"] is None else float(row["EE_system_bps_per_w"])
        best_ee = -math.inf if best["EE_system_bps_per_w"] is None else float(best["EE_system_bps_per_w"])
        if (
            row_ee > best_ee + 1e-12
            or (
                abs(row_ee - best_ee) <= 1e-12
                and float(row["sum_user_throughput_bps"])
                > float(best["sum_user_throughput_bps"])
            )
        ):
            best = row

    if best is not None:
        return best
    return _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="constrained-oracle",
        selected_power_profile="oracle:fallback-fixed-control",
        power_vector=_power_vector_for_candidate(snapshot, settings, "fixed-control"),
        settings=settings,
    )


def _evaluate_snapshots(
    *,
    snapshots: list[_StepSnapshot],
    settings: _AuditSettings,
    power_candidates: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for snapshot in snapshots:
        control_row = _evaluate_power_vector(
            snapshot=snapshot,
            power_semantics="fixed-control",
            selected_power_profile=f"fixed-{settings.fixed_control_power_w:g}w-control",
            power_vector=_power_vector_for_candidate(snapshot, settings, "fixed-control"),
            settings=settings,
        )
        candidate_rows: list[dict[str, Any]] = []
        for candidate in power_candidates:
            if candidate == "fixed-control":
                candidate_rows.append(control_row)
            elif candidate == "constrained-oracle":
                candidate_rows.append(
                    _select_oracle_step(
                        snapshot=snapshot,
                        control_row=control_row,
                        settings=settings,
                    )
                )
            else:
                candidate_rows.append(
                    _evaluate_power_vector(
                        snapshot=snapshot,
                        power_semantics=candidate,
                        selected_power_profile=candidate,
                        power_vector=_power_vector_for_candidate(
                            snapshot,
                            settings,
                            candidate,
                        ),
                        settings=settings,
                    )
                )
        for row in candidate_rows:
            throughputs = row.pop("_user_throughputs")
            rows.append(row)
            user_throughputs_by_key[
                (str(row["trajectory_policy"]), str(row["power_semantics"]))
            ].extend(float(value) for value in throughputs.tolist())
    return rows, user_throughputs_by_key


def _summarize_group(
    *,
    rows: list[dict[str, Any]],
    user_throughputs: list[float],
) -> dict[str, Any]:
    first = rows[0]
    active_counts = [int(row["active_beam_count"]) for row in rows]
    total_powers = [float(row["total_active_beam_power_w"]) for row in rows]
    selected_profiles = [str(row["selected_power_profile"]) for row in rows]
    step_throughput = [float(row["sum_user_throughput_bps"]) for row in rows]
    step_pairs = [
        (float(row["sum_user_throughput_bps"]), float(row["EE_system_bps_per_w"]))
        for row in rows
        if row["EE_system_bps_per_w"] is not None
    ]
    step_ee = [ee for _throughput, ee in step_pairs]
    total_throughput = float(np.sum(step_throughput, dtype=np.float64))
    total_power = float(np.sum(total_powers, dtype=np.float64))
    served_count = sum(int(row["served_count"]) for row in rows)
    outage_count = sum(int(row["outage_count"]) for row in rows)
    budget_excess = [
        float(row["budget_excess_w"])
        for row in rows
        if row["budget_excess_w"] is not None
    ]
    per_beam_excess = [
        float(row["per_beam_power_excess_w"])
        for row in rows
        if row["per_beam_power_excess_w"] is not None
    ]

    return {
        "trajectory_policy": first["trajectory_policy"],
        "power_semantics": first["power_semantics"],
        "step_count": len(rows),
        "evaluation_seeds": sorted({int(row["evaluation_seed"]) for row in rows}),
        "EE_system_aggregate_bps_per_w": (
            None if total_power <= 0.0 else total_throughput / total_power
        ),
        "EE_system_step_mean_bps_per_w": _mean(step_ee),
        "throughput_mean_user_step_bps": _mean(user_throughputs),
        "throughput_p05_user_step_bps": (
            None
            if not user_throughputs
            else float(np.percentile(user_throughputs, 5))
        ),
        "served_ratio": served_count / max(served_count + outage_count, 1),
        "outage_ratio": outage_count / max(served_count + outage_count, 1),
        "handover_count": int(sum(int(row["handover_count"]) for row in rows)),
        "r2_mean": _mean([float(row["r2_mean"]) for row in rows]),
        "active_beam_throughput_gap_bps_mean": _mean(
            [float(row["active_beam_throughput_gap_bps"]) for row in rows]
        ),
        "active_beam_count_distribution": distribution(active_counts),
        "selected_power_profile_distribution": _categorical_distribution(selected_profiles),
        "selected_profile_distinct_count": len(set(selected_profiles)),
        "total_active_beam_power_w_distribution": distribution(total_powers),
        "denominator_varies_in_eval": len(_unique_float_values(total_powers)) > 1,
        "one_active_beam_step_ratio": float(
            np.mean([count == 1 for count in active_counts])
        ),
        "budget_violations": {
            "budget_w": first["total_power_budget_w"],
            "step_count": int(sum(bool(row["budget_violation"]) for row in rows)),
            "step_ratio": float(np.mean([bool(row["budget_violation"]) for row in rows])),
            "max_excess_w": None if not budget_excess else float(np.max(budget_excess)),
        },
        "per_beam_power_violations": {
            "max_power_w": first["per_beam_max_power_w"],
            "step_count": int(sum(bool(row["per_beam_power_violation"]) for row in rows)),
            "step_ratio": float(
                np.mean([bool(row["per_beam_power_violation"]) for row in rows])
            ),
            "max_excess_w": (
                None if not per_beam_excess else float(np.max(per_beam_excess))
            ),
        },
        "inactive_beam_nonzero_power_step_count": int(
            sum(bool(row["inactive_beam_nonzero_power"]) for row in rows)
        ),
        "throughput_vs_EE_system_correlation": _correlation(
            [throughput for throughput, _ee in step_pairs],
            step_ee,
        ),
        "total_throughput_bps": total_throughput,
        "total_active_beam_power_sum_w": total_power,
    }


def _summarize_all(
    *,
    rows: list[dict[str, Any]],
    user_throughputs_by_key: dict[tuple[str, str], list[float]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["trajectory_policy"]), str(row["power_semantics"]))].append(row)
    return [
        _summarize_group(
            rows=group_rows,
            user_throughputs=user_throughputs_by_key[(policy, semantics)],
        )
        for (policy, semantics), group_rows in sorted(by_key.items())
    ]


def _build_ranking_checks(
    candidate_summaries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in candidate_summaries:
        by_policy[str(summary["trajectory_policy"])].append(summary)

    for policy, summaries in sorted(by_policy.items()):
        throughput_ranking = [
            row["power_semantics"]
            for row in sorted(
                summaries,
                key=lambda row: (
                    row["throughput_mean_user_step_bps"]
                    if row["throughput_mean_user_step_bps"] is not None
                    else -math.inf
                ),
                reverse=True,
            )
        ]
        ee_ranking = [
            row["power_semantics"]
            for row in sorted(
                summaries,
                key=lambda row: (
                    row["EE_system_aggregate_bps_per_w"]
                    if row["EE_system_aggregate_bps_per_w"] is not None
                    else -math.inf
                ),
                reverse=True,
            )
        ]
        fixed = next(
            (row for row in summaries if row["power_semantics"] == "fixed-control"),
            None,
        )
        candidate_deltas = {}
        for row in summaries:
            candidate_deltas[row["power_semantics"]] = {
                "throughput_pct_delta_vs_fixed_control": _pct_delta(
                    None if fixed is None else fixed["throughput_mean_user_step_bps"],
                    row["throughput_mean_user_step_bps"],
                ),
                "EE_system_pct_delta_vs_fixed_control": _pct_delta(
                    None if fixed is None else fixed["EE_system_aggregate_bps_per_w"],
                    row["EE_system_aggregate_bps_per_w"],
                ),
            }
        checks.append(
            {
                "trajectory_policy": policy,
                "throughput_rescore_ranking": throughput_ranking,
                "EE_rescore_ranking": ee_ranking,
                "throughput_rescore_winner": throughput_ranking[0],
                "EE_rescore_winner": ee_ranking[0],
                "same_policy_throughput_rescore_vs_EE_rescore_ranking_changes": (
                    throughput_ranking != ee_ranking
                ),
                "same_policy_throughput_rescore_vs_EE_rescore_top_changes": (
                    throughput_ranking[0] != ee_ranking[0]
                ),
                "candidate_deltas_vs_fixed_control": candidate_deltas,
            }
        )
    return checks


def _guardrail_result(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
    settings: _AuditSettings,
) -> dict[str, Any]:
    p05_threshold = settings.p05_min_ratio_vs_control * float(
        control["throughput_p05_user_step_bps"]
    )
    p05_pass = (
        candidate["throughput_p05_user_step_bps"] is not None
        and float(candidate["throughput_p05_user_step_bps"]) >= p05_threshold
    )
    served_threshold = (
        float(control["served_ratio"]) + settings.served_ratio_min_delta_vs_control
    )
    served_pass = float(candidate["served_ratio"]) >= served_threshold
    outage_threshold = (
        float(control["outage_ratio"]) + settings.outage_ratio_max_delta_vs_control
    )
    outage_pass = float(candidate["outage_ratio"]) <= outage_threshold
    budget_pass = int(candidate["budget_violations"]["step_count"]) == 0
    per_beam_pass = int(candidate["per_beam_power_violations"]["step_count"]) == 0
    inactive_pass = int(candidate["inactive_beam_nonzero_power_step_count"]) == 0
    ee_delta = (
        None
        if candidate["EE_system_aggregate_bps_per_w"] is None
        or control["EE_system_aggregate_bps_per_w"] is None
        else float(candidate["EE_system_aggregate_bps_per_w"])
        - float(control["EE_system_aggregate_bps_per_w"])
    )
    return {
        "trajectory_policy": candidate["trajectory_policy"],
        "power_semantics": candidate["power_semantics"],
        "EE_system_delta_vs_fixed_control": ee_delta,
        "EE_system_pct_delta_vs_fixed_control": _pct_delta(
            control["EE_system_aggregate_bps_per_w"],
            candidate["EE_system_aggregate_bps_per_w"],
        ),
        "throughput_p05_ratio_vs_fixed_control": (
            None
            if control["throughput_p05_user_step_bps"] is None
            or abs(float(control["throughput_p05_user_step_bps"])) < 1e-12
            else float(candidate["throughput_p05_user_step_bps"])
            / abs(float(control["throughput_p05_user_step_bps"]))
        ),
        "throughput_p05_pct_delta_vs_fixed_control": _pct_delta(
            control["throughput_p05_user_step_bps"],
            candidate["throughput_p05_user_step_bps"],
        ),
        "p05_threshold_bps": p05_threshold,
        "p05_guardrail_pass": p05_pass,
        "served_ratio_threshold": served_threshold,
        "served_ratio_guardrail_pass": served_pass,
        "outage_ratio_threshold": outage_threshold,
        "outage_guardrail_pass": outage_pass,
        "budget_guardrail_pass": budget_pass,
        "per_beam_power_guardrail_pass": per_beam_pass,
        "inactive_beam_zero_w_guardrail_pass": inactive_pass,
        "QoS_guardrails_pass": bool(p05_pass and served_pass and outage_pass),
        "accepted": bool(
            ee_delta is not None
            and ee_delta > 0.0
            and p05_pass
            and served_pass
            and outage_pass
            and budget_pass
            and per_beam_pass
            and inactive_pass
        ),
    }


def _build_guardrail_checks(
    *,
    candidate_summaries: list[dict[str, Any]],
    settings: _AuditSettings,
) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for summary in candidate_summaries:
        by_policy[str(summary["trajectory_policy"])][str(summary["power_semantics"])] = summary

    checks: list[dict[str, Any]] = []
    for _policy, rows in sorted(by_policy.items()):
        control = rows.get("fixed-control")
        if control is None:
            continue
        for semantics, candidate in sorted(rows.items()):
            if semantics == "fixed-control":
                continue
            checks.append(
                _guardrail_result(
                    candidate=candidate,
                    control=control,
                    settings=settings,
                )
            )
    return checks


def _build_decision(
    *,
    candidate_summaries: list[dict[str, Any]],
    ranking_checks: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    guardrail_accepted = [row for row in guardrail_checks if bool(row["accepted"])]
    guardrail_accepted_keys = {
        (row["trajectory_policy"], row["power_semantics"])
        for row in guardrail_accepted
    }
    guardrail_accepted_summaries = [
        row
        for row in candidate_summaries
        if (row["trajectory_policy"], row["power_semantics"])
        in guardrail_accepted_keys
    ]
    proof_summaries = [
        row
        for row in guardrail_accepted_summaries
        if float(row["one_active_beam_step_ratio"]) < 1.0
        and int(row["selected_profile_distinct_count"]) > 1
        and len(row["total_active_beam_power_w_distribution"]["distinct"]) > 1
    ]
    proof_keys = {
        (row["trajectory_policy"], row["power_semantics"])
        for row in proof_summaries
    }
    proof_accepted = [
        row
        for row in guardrail_accepted
        if (row["trajectory_policy"], row["power_semantics"]) in proof_keys
    ]

    denominator_changed = False
    for policy in sorted({row["trajectory_policy"] for row in candidate_summaries}):
        powers = [
            row["total_active_beam_power_sum_w"]
            for row in candidate_summaries
            if row["trajectory_policy"] == policy
        ]
        if len(_unique_float_values(powers, places=9)) > 1:
            denominator_changed = True
            break

    ranking_separates = any(
        bool(row["same_policy_throughput_rescore_vs_EE_rescore_ranking_changes"])
        for row in ranking_checks
    )
    has_budget_respecting_candidate = any(
        bool(row["budget_guardrail_pass"]) and bool(row["per_beam_power_guardrail_pass"])
        for row in guardrail_checks
    )
    oracle_or_heuristic_beats = bool(proof_accepted)
    qos_pass = any(bool(row["QoS_guardrails_pass"]) for row in proof_accepted)
    selected_profile_not_single = bool(proof_summaries)
    active_power_not_single = bool(proof_summaries)
    no_budget_violations_for_accepted = all(
        int(row["budget_violations"]["step_count"]) == 0 for row in proof_summaries
    )
    denominator_varies_for_accepted = any(
        bool(row["denominator_varies_in_eval"]) for row in proof_summaries
    )

    proof_flags = {
        "denominator_changed_by_power_decision": denominator_changed,
        "denominator_varies_for_accepted_candidate": denominator_varies_for_accepted,
        "ranking_separates_under_same_policy_rescore": ranking_separates,
        "has_budget_respecting_candidate": has_budget_respecting_candidate,
        "oracle_or_heuristic_beats_fixed_control_on_EE": oracle_or_heuristic_beats,
        "QoS_guardrails_pass": qos_pass,
        "selected_profile_not_single_point_on_noncollapsed_trajectories": (
            selected_profile_not_single
        ),
        "active_power_not_single_point_on_noncollapsed_trajectories": (
            active_power_not_single
        ),
        "no_budget_violations_for_accepted_candidate": no_budget_violations_for_accepted,
    }
    pass_required = all(bool(value) for value in proof_flags.values())

    stop_conditions = {
        "oracle_cannot_beat_fixed_control_under_QoS_guardrails": not oracle_or_heuristic_beats,
        "EE_gain_from_throughput_tail_or_served_ratio_collapse": any(
            float(row["EE_system_delta_vs_fixed_control"] or 0.0) > 0.0
            and not bool(row["QoS_guardrails_pass"])
            for row in guardrail_checks
        ),
        "denominator_remains_fixed": not denominator_changed,
        "accepted_candidates_single_profile_or_power_on_noncollapsed_trajectories": (
            bool(guardrail_accepted_summaries)
            and not (selected_profile_not_single and active_power_not_single)
        ),
        "accepted_candidate_has_budget_violation": not no_budget_violations_for_accepted,
    }

    if pass_required:
        decision = "PASS to RA-EE-03 design"
    elif stop_conditions["oracle_cannot_beat_fixed_control_under_QoS_guardrails"]:
        decision = "BLOCKED"
    elif any(bool(value) for value in stop_conditions.values()):
        decision = "BLOCKED"
    else:
        decision = "NEEDS MORE EVIDENCE"

    return {
        "ra_ee_02_decision": decision,
        "proof_flags": proof_flags,
        "accepted_candidates": proof_accepted,
        "guardrail_accepted_candidates": guardrail_accepted,
        "stop_conditions": stop_conditions,
        "allowed_next_step": (
            "RA-EE-03 resource-allocation MDP design only; no RL training claim yet"
            if decision == "PASS to RA-EE-03 design"
            else "do not proceed to RA-EE-03 design without resolving blockers"
        ),
    }


def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_02_decision"]
    proof = summary["proof_flags"]
    accepted = summary["decision_detail"]["accepted_candidates"]
    lines = [
        "# RA-EE-02 Oracle Power-Allocation Audit Review",
        "",
        "Offline fixed-trajectory oracle / heuristic audit only. No RL training, "
        "Catfish, multi-Catfish, frozen baseline mutation, HOBS optimizer claim, "
        "or old EE-MODQN effectiveness claim was performed.",
        "",
        "## Protocol",
        "",
        f"- method label: `{summary['protocol']['method_label']}`",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- evaluation seeds: `{summary['inputs']['evaluation_seed_set']}`",
        f"- trajectories: `{summary['protocol']['trajectory_policies']}`",
        f"- candidates: `{summary['protocol']['power_candidates']}`",
        f"- p05 throughput guardrail ratio: "
        f"`{summary['constraints']['p05_throughput_min_ratio_vs_control']}`",
        "",
        "## Proof Flags",
        "",
        f"- denominator changed by power decision: "
        f"`{proof['denominator_changed_by_power_decision']}`",
        f"- same-policy throughput-vs-EE ranking separates: "
        f"`{proof['ranking_separates_under_same_policy_rescore']}`",
        f"- has budget-respecting candidate: "
        f"`{proof['has_budget_respecting_candidate']}`",
        f"- oracle/heuristic beats fixed control on EE: "
        f"`{proof['oracle_or_heuristic_beats_fixed_control_on_EE']}`",
        f"- QoS guardrails pass: `{proof['QoS_guardrails_pass']}`",
        f"- selected profile not single-point on non-collapsed trajectories: "
        f"`{proof['selected_profile_not_single_point_on_noncollapsed_trajectories']}`",
        f"- active power not single-point on non-collapsed trajectories: "
        f"`{proof['active_power_not_single_point_on_noncollapsed_trajectories']}`",
        f"- no budget violations for accepted candidate: "
        f"`{proof['no_budget_violations_for_accepted_candidate']}`",
        "",
        "## Accepted Candidates",
        "",
        f"- accepted count: `{len(accepted)}`",
    ]
    for row in accepted:
        lines.append(
            "- "
            f"`{row['trajectory_policy']}::{row['power_semantics']}` "
            f"EE delta `{row['EE_system_delta_vs_fixed_control']}`, "
            f"p05 ratio `{row['throughput_p05_ratio_vs_fixed_control']}`, "
            f"QoS `{row['QoS_guardrails_pass']}`, "
            f"budget `{row['budget_guardrail_pass']}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- RA-EE-02 decision: `{decision}`",
            f"- allowed next step: {summary['decision_detail']['allowed_next_step']}",
            "",
            "## Forbidden Claims",
            "",
            "- Do not claim old EE-MODQN effectiveness.",
            "- Do not claim HOBS optimizer behavior.",
            "- Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.",
            "- Do not treat per-user EE credit as system EE.",
            "- Do not use scalar reward alone as success evidence.",
            "- Do not claim full paper-faithful reproduction or absolute energy saving.",
        ]
    )
    return lines


def export_ra_ee_02_oracle_power_allocation_audit(
    config_path: str | Path = "configs/ra-ee-02-oracle-power-allocation-audit.resolved.yaml",
    output_dir: str | Path = "artifacts/ra-ee-02-oracle-power-allocation-audit",
    *,
    learned_run_dir: str | Path | None = DEFAULT_PHASE03C_C_LEARNED_RUN_DIR,
    include_learned: bool = True,
    evaluation_seed_set: tuple[int, ...] | None = None,
    max_steps: int | None = None,
    policies: tuple[str, ...] = COUNTERFACTUAL_POLICIES,
    power_candidates: tuple[str, ...] = POWER_CANDIDATES,
) -> dict[str, Any]:
    """Export the RA-EE-02 fixed-trajectory oracle power-allocation audit."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    if not power_candidates:
        raise ValueError("At least one RA-EE-02 power candidate is required.")
    if "fixed-control" not in power_candidates:
        power_candidates = ("fixed-control",) + tuple(power_candidates)
    unsupported = sorted(set(power_candidates) - set(POWER_CANDIDATES))
    if unsupported:
        raise ValueError(f"Unsupported RA-EE-02 power candidates: {unsupported!r}")

    cfg = load_training_yaml(config_path)
    settings = _settings_from_config(cfg)
    seeds = get_seeds(cfg)
    eval_seeds = tuple(
        int(seed)
        for seed in (
            evaluation_seed_set
            if evaluation_seed_set is not None
            else tuple(seeds["evaluation_seed_set"])
        )
    )
    if not eval_seeds:
        raise ValueError("RA-EE-02 audit requires evaluation seeds.")

    trajectories: dict[str, dict[int, list[np.ndarray]]] = {}
    learned_trajectory = {"available": False, "reason": "not-requested"}
    if include_learned and learned_run_dir is not None:
        run_dir = Path(learned_run_dir)
        if run_dir.exists() and (run_dir / "run_metadata.json").exists():
            learned_paths, learned_trajectory = _rollout_learned_trajectory(
                run_dir=run_dir,
                evaluation_seed_set=eval_seeds,
                max_steps=max_steps,
            )
            trajectories.update(learned_paths)
        else:
            learned_trajectory = {
                "available": False,
                "run_dir": str(run_dir),
                "reason": "run directory or run_metadata.json unavailable",
            }

    trajectories.update(
        _rollout_counterfactual_trajectories(
            cfg=cfg,
            evaluation_seed_set=eval_seeds,
            max_steps=max_steps,
            policies=policies,
        )
    )

    snapshots = _build_unit_power_snapshots(
        base_cfg=cfg,
        settings=settings,
        trajectories=trajectories,
    )
    step_rows, user_throughputs_by_key = _evaluate_snapshots(
        snapshots=snapshots,
        settings=settings,
        power_candidates=power_candidates,
    )
    candidate_summaries = _summarize_all(
        rows=step_rows,
        user_throughputs_by_key=user_throughputs_by_key,
    )
    ranking_checks = _build_ranking_checks(candidate_summaries)
    guardrail_checks = _build_guardrail_checks(
        candidate_summaries=candidate_summaries,
        settings=settings,
    )
    decision_detail = _build_decision(
        candidate_summaries=candidate_summaries,
        ranking_checks=ranking_checks,
        guardrail_checks=guardrail_checks,
    )

    out_dir = Path(output_dir)
    summary = {
        "inputs": {
            "config_path": str(Path(config_path)),
            "output_dir": str(out_dir),
            "learned_run_dir": None if learned_run_dir is None else str(learned_run_dir),
            "evaluation_seed_set": list(eval_seeds),
            "max_steps": max_steps,
        },
        "protocol": {
            "phase": "RA-EE-02",
            "method_label": settings.method_label,
            "method_family": "RA-EE-MDP / RA-EE-MODQN",
            "training": "not-run",
            "catfish": "disabled",
            "multi_catfish": "disabled",
            "frozen_baseline_mutation": "forbidden/not-performed",
            "old_EE_MODQN_claim": "forbidden/not-made",
            "hobs_optimizer_claim": "forbidden/not-made",
            "controller_claim": (
                "offline finite-codebook oracle / heuristic upper-bound proof; "
                "new extension only"
            ),
            "trajectory_policies": list(trajectories.keys()),
            "counterfactual_policies": list(policies),
            "power_candidates": list(power_candidates),
            "system_EE_primary": True,
            "per_user_EE_credit_is_system_EE": False,
            "scalar_reward_success_basis": False,
            "finite_horizon_oracle": "not-implemented; stepwise finite-codebook oracle only",
        },
        "constraints": {
            "per_beam_max_power_w": settings.per_beam_max_power_w,
            "total_power_budget_w": settings.total_power_budget_w,
            "inactive_beam_policy": "zero-w",
            "zero_budget_violations_required_for_acceptance": True,
            "served_ratio_min_delta_vs_control": (
                settings.served_ratio_min_delta_vs_control
            ),
            "outage_ratio_max_delta_vs_control": (
                settings.outage_ratio_max_delta_vs_control
            ),
            "p05_throughput_min_ratio_vs_control": (
                settings.p05_min_ratio_vs_control
            ),
            "handover_and_load_balance": "diagnostic-only under fixed trajectories",
            "codebook_levels_w": list(settings.codebook_levels_w),
            "oracle_max_demoted_beams": settings.oracle_max_demoted_beams,
        },
        "learned_trajectory": learned_trajectory,
        "candidate_summaries": candidate_summaries,
        "ranking_separation_result": {
            "ranking_separates_under_same_policy_rescore": decision_detail[
                "proof_flags"
            ]["ranking_separates_under_same_policy_rescore"],
            "ranking_checks": ranking_checks,
        },
        "guardrail_checks": guardrail_checks,
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "ra_ee_02_decision": decision_detail["ra_ee_02_decision"],
        "remaining_blockers": [
            "This is offline fixed-trajectory evidence, not learned RA-EE-MODQN evidence.",
            "The finite-codebook oracle is a new-extension upper-bound surface, not a HOBS optimizer.",
            "A PASS authorizes RA-EE-03 design only; it does not authorize RL training by itself.",
        ],
        "forbidden_claims_still_active": [
            "Do not claim old EE-MODQN effectiveness.",
            "Do not claim full paper-faithful reproduction.",
            "Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.",
            "Do not treat per-user EE credit as system EE.",
            "Do not use scalar reward alone as success evidence.",
            "Do not label this oracle or heuristic as a HOBS optimizer.",
        ],
    }

    step_csv = _write_csv(
        out_dir / "ra_ee_02_step_metrics.csv",
        step_rows,
        fieldnames=list(step_rows[0].keys()),
    )
    summary_csv = _write_csv(
        out_dir / "ra_ee_02_candidate_summary.csv",
        [
            {
                "trajectory_policy": row["trajectory_policy"],
                "power_semantics": row["power_semantics"],
                "EE_system_aggregate_bps_per_w": row["EE_system_aggregate_bps_per_w"],
                "EE_system_step_mean_bps_per_w": row["EE_system_step_mean_bps_per_w"],
                "throughput_mean_user_step_bps": row[
                    "throughput_mean_user_step_bps"
                ],
                "throughput_p05_user_step_bps": row[
                    "throughput_p05_user_step_bps"
                ],
                "served_ratio": row["served_ratio"],
                "outage_ratio": row["outage_ratio"],
                "active_beam_count_distribution": row[
                    "active_beam_count_distribution"
                ],
                "selected_power_profile_distribution": row[
                    "selected_power_profile_distribution"
                ],
                "total_active_beam_power_w_distribution": row[
                    "total_active_beam_power_w_distribution"
                ],
                "denominator_varies_in_eval": row["denominator_varies_in_eval"],
                "one_active_beam_step_ratio": row["one_active_beam_step_ratio"],
                "budget_violation_step_count": row["budget_violations"][
                    "step_count"
                ],
                "throughput_EE_pearson": row[
                    "throughput_vs_EE_system_correlation"
                ]["pearson"],
                "throughput_EE_spearman": row[
                    "throughput_vs_EE_system_correlation"
                ]["spearman"],
            }
            for row in candidate_summaries
        ],
        fieldnames=[
            "trajectory_policy",
            "power_semantics",
            "EE_system_aggregate_bps_per_w",
            "EE_system_step_mean_bps_per_w",
            "throughput_mean_user_step_bps",
            "throughput_p05_user_step_bps",
            "served_ratio",
            "outage_ratio",
            "active_beam_count_distribution",
            "selected_power_profile_distribution",
            "total_active_beam_power_w_distribution",
            "denominator_varies_in_eval",
            "one_active_beam_step_ratio",
            "budget_violation_step_count",
            "throughput_EE_pearson",
            "throughput_EE_spearman",
        ],
    )
    summary_path = write_json(
        out_dir / "ra_ee_02_oracle_power_allocation_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "ra_ee_02_oracle_power_allocation_summary": summary_path,
        "ra_ee_02_step_metrics": step_csv,
        "ra_ee_02_candidate_summary": summary_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = ["export_ra_ee_02_oracle_power_allocation_audit"]
