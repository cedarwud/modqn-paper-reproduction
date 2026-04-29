"""Shared constants, settings, and config helpers for RA-EE-09."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import get_seeds
from .ra_ee_02_oracle_power_allocation import _AuditSettings
from .ra_ee_05_fixed_association_robustness import (
    CALIBRATION_BUCKET,
    CALIBRATION_TRAJECTORIES,
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_HELD_OUT_SEEDS,
    HELD_OUT_BUCKET,
    HELD_OUT_TRAJECTORIES,
    _BucketSpec,
)
from .ra_ee_07_constrained_power_allocator_distillation import (
    DEPLOYABLE_ALLOCATORS,
    DETERMINISTIC_HYBRID,
    _RAEE07Settings,
)


DEFAULT_CONFIG = "configs/ra-ee-09-fixed-association-rb-bandwidth-control.resolved.yaml"


DEFAULT_OUTPUT_DIR = "artifacts/ra-ee-09-fixed-association-rb-bandwidth-control-pilot"


DEFAULT_CANDIDATE_OUTPUT_DIR = (
    "artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot"
)


DEFAULT_COMPARISON_OUTPUT_DIR = (
    "artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/"
    "paired-comparison-vs-control"
)


RA_EE_09_ASSUMPTION_KEY = "ra-ee-09-fixed-association-rb-bandwidth-control"


RA_EE_09_GATE_ID = "RA-EE-09"


RA_EE_09_METHOD_LABEL = "fixed-association RB / bandwidth allocation design gate"


RA_EE_09_CONTROL = (
    "fixed-association+deployable-stronger-power-allocator+"
    "equal-share-resource-control"
)


RA_EE_09_RESOURCE_UNIT = "normalized_per_beam_bandwidth_fraction"


RA_EE_09_EQUAL_SHARE_ALLOCATOR = "ra-ee-09-equal-share-control"


RA_EE_09_CANDIDATE_ALLOCATOR = "bounded-qos-slack-resource-share-allocator"


RA_EE_09_POWER_ALLOCATOR_ID = "RA-EE-07 deployable-stronger-power-allocator"


RA_EE_09_THROUGHPUT_FORMULA_VERSION = "ra-ee-09-generalized-resource-share-v1"


RA_EE_09_CANDIDATE = (
    "fixed-association+deployable-stronger-power-allocator+"
    "bounded-qos-slack-resource-share-allocator"
)


RA_EE_09_CANDIDATE_MIN_EQUAL_SHARE_MULTIPLIER = 0.25


RA_EE_09_CANDIDATE_MAX_EQUAL_SHARE_MULTIPLIER = 4.0


RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC = (
    "p05_throughput_per_active_resource_budget"
)


RA_EE_09_MAX_POSITIVE_GAIN_CONTRIBUTION_SHARE = 0.80


@dataclass(frozen=True)
class _ResourceSettings:
    resource_unit: str
    resource_allocator_id: str
    per_beam_budget: float
    inactive_beam_resource_policy: str
    per_user_min_equal_share_multiplier: float
    per_user_max_equal_share_multiplier: float
    resource_sum_tolerance: float
    throughput_formula_version: str


@dataclass(frozen=True)
class _RAEE09Settings:
    method_label: str
    implementation_sublabel: str
    power_settings: _RAEE07Settings
    resource: _ResourceSettings
    metadata: dict[str, Any]


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


def _ra_ee_09_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get(RA_EE_09_ASSUMPTION_KEY, {})
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


def _tuple_ints(raw: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    values = raw if raw is not None else fallback
    return tuple(int(value) for value in values)


def _tuple_strings(raw: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    values = raw if raw is not None else fallback
    return tuple(str(value) for value in values)


def _config_path_has_ra_ee_09_namespace(config_path: str | Path | None) -> bool:
    if config_path is None:
        return True
    return Path(config_path).name.startswith("ra-ee-09-")


def ra_ee_09_resource_accounting_enabled(
    cfg: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> bool:
    """Return whether the explicit RA-EE-09 resource path is enabled."""
    gate = _ra_ee_09_value(cfg)
    if not gate or not bool(gate.get("enabled", False)):
        return False
    if str(cfg.get("track", {}).get("phase", "")) != RA_EE_09_GATE_ID:
        return False
    if not _config_path_has_ra_ee_09_namespace(config_path):
        return False
    artifact_namespace = str(gate.get("artifact_namespace", ""))
    return artifact_namespace.startswith("artifacts/ra-ee-09-")


def _bucket_specs_from_config(
    gate: dict[str, Any],
    seeds: dict[str, Any],
) -> tuple[_BucketSpec, ...]:
    buckets = gate.get("fixed_association_buckets", {})
    if not isinstance(buckets, dict):
        buckets = {}
    calibration = buckets.get(CALIBRATION_BUCKET, {})
    if not isinstance(calibration, dict):
        calibration = {}
    held_out = buckets.get(HELD_OUT_BUCKET, buckets.get("held_out", {}))
    if not isinstance(held_out, dict):
        held_out = {}

    specs = (
        _BucketSpec(
            name=CALIBRATION_BUCKET,
            trajectory_families=_tuple_strings(
                calibration.get("trajectory_families"),
                CALIBRATION_TRAJECTORIES,
            ),
            evaluation_seed_set=_tuple_ints(
                calibration.get("evaluation_seed_set"),
                _tuple_ints(seeds.get("evaluation_seed_set"), DEFAULT_CALIBRATION_SEEDS),
            ),
        ),
        _BucketSpec(
            name=HELD_OUT_BUCKET,
            trajectory_families=_tuple_strings(
                held_out.get("trajectory_families"),
                HELD_OUT_TRAJECTORIES,
            ),
            evaluation_seed_set=_tuple_ints(
                held_out.get("evaluation_seed_set"),
                DEFAULT_HELD_OUT_SEEDS,
            ),
        ),
    )
    supported = set(CALIBRATION_TRAJECTORIES + HELD_OUT_TRAJECTORIES)
    for spec in specs:
        if not spec.evaluation_seed_set:
            raise ValueError(f"RA-EE-09 bucket {spec.name!r} requires eval seeds.")
        unsupported = sorted(set(spec.trajectory_families) - supported)
        if unsupported:
            raise ValueError(
                f"Unsupported RA-EE-09 fixed trajectories in {spec.name!r}: "
                f"{unsupported!r}"
            )
    return specs


def _settings_from_config(
    cfg: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> _RAEE09Settings:
    if not ra_ee_09_resource_accounting_enabled(cfg, config_path=config_path):
        raise ValueError(
            "RA-EE-09 resource accounting requires an explicit enabled "
            "`ra-ee-09-*` config namespace."
        )

    gate = _ra_ee_09_value(cfg)
    power = _power_surface_value(cfg)
    seeds = get_seeds(cfg)

    if gate.get("ra_ee_gate_id") != RA_EE_09_GATE_ID:
        raise ValueError("RA-EE-09 config must declare ra_ee_gate_id = RA-EE-09.")
    if str(gate.get("resource_unit")) != RA_EE_09_RESOURCE_UNIT:
        raise ValueError(
            "RA-EE-09 resource_unit must be "
            f"{RA_EE_09_RESOURCE_UNIT!r}."
        )
    if str(gate.get("resource_allocator_id")) != RA_EE_09_EQUAL_SHARE_ALLOCATOR:
        raise ValueError("Slice 09A-09C supports only equal-share control.")
    if str(gate.get("power_allocator_id")) != RA_EE_09_POWER_ALLOCATOR_ID:
        raise ValueError("RA-EE-09 must use the RA-EE-07 deployable power boundary.")
    if bool(gate.get("resource_allocation_feedback_to_power_decision")):
        raise ValueError("RA-EE-09 resource allocation must not feed back to power.")

    levels_raw = gate.get(
        "codebook_levels_w",
        power.get("power_codebook_levels_w", [0.5, 0.75, 1.0, 1.5, 2.0]),
    )
    levels = tuple(float(level) for level in levels_raw)
    if not levels:
        raise ValueError("RA-EE-09 requires at least one power codebook level.")
    if tuple(sorted(levels)) != levels:
        raise ValueError(f"RA-EE-09 codebook levels must be sorted, got {levels!r}.")

    deployable = _tuple_strings(
        gate.get("deployable_allocator_candidates"),
        DEPLOYABLE_ALLOCATORS,
    )
    unsupported = sorted(set(deployable) - set(DEPLOYABLE_ALLOCATORS))
    if unsupported:
        raise ValueError(f"Unsupported RA-EE-09 deployable allocators: {unsupported!r}")
    primary = str(gate.get("primary_deployable_allocator", DETERMINISTIC_HYBRID))
    if primary not in deployable:
        raise ValueError("RA-EE-09 primary deployable allocator must be a candidate.")

    audit = _AuditSettings(
        method_label=str(gate.get("method_label", RA_EE_09_METHOD_LABEL)),
        codebook_levels_w=levels,
        fixed_control_power_w=float(gate.get("fixed_control_power_w", 1.0)),
        total_power_budget_w=float(
            gate.get("total_active_power_budget_w", power.get("total_power_budget_w", 8.0))
        ),
        per_beam_max_power_w=float(
            gate.get("per_beam_max_power_w", power.get("max_power_w", 2.0))
        ),
        active_base_power_w=float(
            gate.get("active_base_power_w", power.get("active_base_power_w", 0.25))
        ),
        load_scale_power_w=float(
            gate.get("load_scale_power_w", power.get("load_scale_power_w", 0.35))
        ),
        load_exponent=float(
            gate.get("load_exponent", power.get("load_exponent", 0.5))
        ),
        p05_min_ratio_vs_control=float(
            gate.get("p05_throughput_min_ratio_vs_control", 0.95)
        ),
        served_ratio_min_delta_vs_control=float(
            gate.get("served_ratio_min_delta_vs_control", 0.0)
        ),
        outage_ratio_max_delta_vs_control=float(
            gate.get("outage_ratio_max_delta_vs_control", 0.0)
        ),
        oracle_max_demoted_beams=int(gate.get("oracle_max_demoted_beams", 3)),
    )
    power_settings = _RAEE07Settings(
        method_label=str(gate.get("method_label", RA_EE_09_METHOD_LABEL)),
        implementation_sublabel=str(
            gate.get(
                "implementation_sublabel",
                "RA-EE-09 Slice 09A-09C equal-share control replay",
            )
        ),
        audit=audit,
        fixed_bucket_specs=_bucket_specs_from_config(gate, seeds),
        diagnostic_bucket_specs=(),
        deployable_allocators=deployable,
        primary_deployable_allocator=primary,
        candidate_max_demoted_beams=int(gate.get("candidate_max_demoted_beams", 3)),
        candidate_step_p05_guardrail_margin=float(
            gate.get("candidate_step_p05_guardrail_margin", 0.005)
        ),
        local_search_max_moves=int(gate.get("local_search_max_moves", 8)),
        p05_trim_max_moves=int(gate.get("p05_trim_max_moves", 6)),
        dp_max_profile_count=int(gate.get("dp_max_profile_count", 512)),
        min_oracle_gap_closed_ratio=float(gate.get("min_oracle_gap_closed_ratio", 0.20)),
        association_diagnostic_policies=(),
        min_active_beams=int(gate.get("min_active_beams", 2)),
        max_active_beams=int(gate.get("max_active_beams", 8)),
        target_users_per_active_beam=int(gate.get("target_users_per_active_beam", 16)),
        load_cap_overflow_users=int(gate.get("load_cap_overflow_users", 2)),
        max_moved_user_ratio_per_step=float(
            gate.get("max_moved_user_ratio_per_step", 0.18)
        ),
        max_moved_user_ratio=float(gate.get("max_moved_user_ratio", 0.20)),
        max_one_active_beam_ratio_for_acceptance=float(
            gate.get("max_one_active_beam_ratio_for_acceptance", 0.25)
        ),
        max_two_beam_overload_step_ratio=float(
            gate.get("max_two_beam_overload_step_ratio", 0.10)
        ),
        diagnostic_max_steps=None,
    )
    resource = _ResourceSettings(
        resource_unit=str(gate.get("resource_unit", RA_EE_09_RESOURCE_UNIT)),
        resource_allocator_id=str(
            gate.get("resource_allocator_id", RA_EE_09_EQUAL_SHARE_ALLOCATOR)
        ),
        per_beam_budget=float(gate.get("per_beam_budget", 1.0)),
        inactive_beam_resource_policy=str(
            gate.get("inactive_beam_resource_policy", "zero")
        ),
        per_user_min_equal_share_multiplier=float(
            gate.get("per_user_min_equal_share_multiplier", 1.0)
        ),
        per_user_max_equal_share_multiplier=float(
            gate.get("per_user_max_equal_share_multiplier", 1.0)
        ),
        resource_sum_tolerance=float(gate.get("resource_sum_tolerance", 1e-12)),
        throughput_formula_version=str(
            gate.get(
                "throughput_formula_version",
                RA_EE_09_THROUGHPUT_FORMULA_VERSION,
            )
        ),
    )
    if resource.per_beam_budget <= 0.0:
        raise ValueError("RA-EE-09 per_beam_budget must be positive.")
    if resource.inactive_beam_resource_policy != "zero":
        raise ValueError("RA-EE-09 inactive beam resource policy must be zero.")
    if resource.per_user_min_equal_share_multiplier < 0.0:
        raise ValueError("RA-EE-09 per-user minimum multiplier must be nonnegative.")
    if resource.per_user_max_equal_share_multiplier < resource.per_user_min_equal_share_multiplier:
        raise ValueError("RA-EE-09 per-user max multiplier must be >= min multiplier.")

    metadata = {
        "ra_ee_gate_id": gate["ra_ee_gate_id"],
        "method_label": str(gate.get("method_label", RA_EE_09_METHOD_LABEL)),
        "implementation_sublabel": power_settings.implementation_sublabel,
        "association_mode": str(gate.get("association_mode", "fixed-replay")),
        "power_allocator_id": str(gate.get("power_allocator_id")),
        "resource_unit": resource.resource_unit,
        "resource_allocator_id": resource.resource_allocator_id,
        "per_beam_budget": resource.per_beam_budget,
        "total_resource_budget": str(gate.get("total_resource_budget", "active_beam_count")),
        "per_user_min": f"{resource.per_user_min_equal_share_multiplier:g}/N_b",
        "per_user_max": f"{resource.per_user_max_equal_share_multiplier:g}/N_b",
        "inactive_beam_resource_policy": resource.inactive_beam_resource_policy,
        "throughput_formula_version": resource.throughput_formula_version,
        "noise_policy": "unchanged",
        "candidate_allocator_enabled": False,
        "learned_association_disabled": bool(gate.get("learned_association_disabled")),
        "hierarchical_RL_disabled": bool(gate.get("hierarchical_RL_disabled")),
        "catfish_disabled": bool(gate.get("catfish_disabled")),
        "phase03c_continuation_disabled": bool(
            gate.get("phase03c_continuation_disabled")
        ),
        "scalar_reward_success_basis": bool(gate.get("scalar_reward_success_basis")),
    }
    return _RAEE09Settings(
        method_label=power_settings.method_label,
        implementation_sublabel=power_settings.implementation_sublabel,
        power_settings=power_settings,
        resource=resource,
        metadata=metadata,
    )


def _candidate_settings_from_control(settings: _RAEE09Settings) -> _RAEE09Settings:
    """Return Slice 09D settings without changing the RA-EE-07 power boundary."""
    resource = replace(
        settings.resource,
        resource_allocator_id=RA_EE_09_CANDIDATE_ALLOCATOR,
        per_user_min_equal_share_multiplier=(
            RA_EE_09_CANDIDATE_MIN_EQUAL_SHARE_MULTIPLIER
        ),
        per_user_max_equal_share_multiplier=(
            RA_EE_09_CANDIDATE_MAX_EQUAL_SHARE_MULTIPLIER
        ),
    )
    metadata = dict(settings.metadata)
    metadata.update(
        {
            "implementation_sublabel": "RA-EE-09 Slice 09D bounded resource-share candidate",
            "resource_allocator_id": RA_EE_09_CANDIDATE_ALLOCATOR,
            "per_user_min": "0.25/N_b",
            "per_user_max": "min(4/N_b, 1.0)",
            "candidate_allocator_enabled": True,
            "learned_association_disabled": True,
            "hierarchical_RL_disabled": True,
            "catfish_disabled": True,
            "phase03c_continuation_disabled": True,
            "scalar_reward_success_basis": False,
        }
    )
    return replace(
        settings,
        implementation_sublabel="RA-EE-09 Slice 09D bounded resource-share candidate",
        resource=resource,
        metadata=metadata,
    )


def _hash_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _hash_int_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=np.float64)))


def _safe_percentile(values: list[float], percentile: float) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.percentile(np.asarray(clean, dtype=np.float64), percentile))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    denom = float(denominator)
    if abs(denom) <= 1e-12:
        return None
    return float(numerator) / denom


def _pct_delta(base: float | None, candidate: float | None) -> float | None:
    if base is None or candidate is None:
        return None
    base_value = float(base)
    if abs(base_value) <= 1e-12:
        return None
    return 100.0 * (float(candidate) - base_value) / abs(base_value)


def _correlation_or_none(xs: list[float], ys: list[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys, strict=False)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([x for x, _y in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _x, y in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _parse_vector(raw: Any) -> list[float]:
    if raw is None:
        return []
    return [
        float(value)
        for value in np.fromstring(str(raw), sep=" ", dtype=np.float64).tolist()
        if math.isfinite(float(value))
    ]


def _run_power_settings_with_fixed_specs(
    settings: _RAEE09Settings,
    *,
    fixed_specs: tuple[_BucketSpec, ...],
    implementation_sublabel: str,
) -> _RAEE07Settings:
    return _RAEE07Settings(
        method_label=settings.power_settings.method_label,
        implementation_sublabel=implementation_sublabel,
        audit=settings.power_settings.audit,
        fixed_bucket_specs=fixed_specs,
        diagnostic_bucket_specs=(),
        deployable_allocators=settings.power_settings.deployable_allocators,
        primary_deployable_allocator=settings.power_settings.primary_deployable_allocator,
        candidate_max_demoted_beams=settings.power_settings.candidate_max_demoted_beams,
        candidate_step_p05_guardrail_margin=(
            settings.power_settings.candidate_step_p05_guardrail_margin
        ),
        local_search_max_moves=settings.power_settings.local_search_max_moves,
        p05_trim_max_moves=settings.power_settings.p05_trim_max_moves,
        dp_max_profile_count=settings.power_settings.dp_max_profile_count,
        min_oracle_gap_closed_ratio=settings.power_settings.min_oracle_gap_closed_ratio,
        association_diagnostic_policies=(),
        min_active_beams=settings.power_settings.min_active_beams,
        max_active_beams=settings.power_settings.max_active_beams,
        target_users_per_active_beam=settings.power_settings.target_users_per_active_beam,
        load_cap_overflow_users=settings.power_settings.load_cap_overflow_users,
        max_moved_user_ratio_per_step=(
            settings.power_settings.max_moved_user_ratio_per_step
        ),
        max_moved_user_ratio=settings.power_settings.max_moved_user_ratio,
        max_one_active_beam_ratio_for_acceptance=(
            settings.power_settings.max_one_active_beam_ratio_for_acceptance
        ),
        max_two_beam_overload_step_ratio=(
            settings.power_settings.max_two_beam_overload_step_ratio
        ),
        diagnostic_max_steps=None,
    )
