"""RA-EE-05 fixed-association robustness and held-out validation.

This module evaluates fixed association trajectories only. It reuses the
RA-EE-04 centralized safe-greedy power allocator, but it does not train
association, does not introduce Catfish, and does not mutate frozen baseline
configs or artifacts.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import build_environment, get_seeds, load_training_yaml
from ..env.step import HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
from ._common import write_json
from .ra_ee_02_oracle_power_allocation import (
    _AuditSettings,
    _StepSnapshot,
    _build_guardrail_checks,
    _build_ranking_checks,
    _build_unit_power_snapshots,
    _evaluate_power_vector,
    _format_vector,
    _power_vector_for_candidate,
    _select_oracle_step,
    _summarize_all,
)
from .ra_ee_04_bounded_power_allocator import (
    RA_EE_04_CANDIDATE,
    RA_EE_04_ORACLE,
)


DEFAULT_CONFIG = "configs/ra-ee-05-fixed-association-robustness.resolved.yaml"
DEFAULT_OUTPUT_DIR = "artifacts/ra-ee-05-fixed-association-robustness"

RA_EE_05_METHOD_LABEL = "RA-EE fixed-association centralized power allocator"
RA_EE_05_CANDIDATE = RA_EE_04_CANDIDATE
RA_EE_05_ORACLE = RA_EE_04_ORACLE

CALIBRATION_BUCKET = "calibration"
HELD_OUT_BUCKET = "held-out"

CALIBRATION_TRAJECTORIES = (
    "hold-current",
    "random-valid",
    "spread-valid",
)
HELD_OUT_TRAJECTORIES = (
    "random-valid-heldout",
    "spread-valid-heldout",
    "load-skewed-heldout",
    "mobility-shift-heldout",
    "mixed-valid-heldout",
)
RA_EE_05_TRAJECTORIES = CALIBRATION_TRAJECTORIES + HELD_OUT_TRAJECTORIES

DEFAULT_CALIBRATION_SEEDS = (100, 200, 300, 400, 500)
DEFAULT_HELD_OUT_SEEDS = (600, 700, 800, 900, 1000)

_POLICY_SEED_OFFSETS = {
    "hold-current": 1009,
    "random-valid": 2003,
    "spread-valid": 4001,
    "random-valid-heldout": 6029,
    "spread-valid-heldout": 7013,
    "load-skewed-heldout": 8011,
    "mobility-shift-heldout": 9011,
    "mixed-valid-heldout": 10007,
}


@dataclass(frozen=True)
class _BucketSpec:
    name: str
    trajectory_families: tuple[str, ...]
    evaluation_seed_set: tuple[int, ...]


@dataclass(frozen=True)
class _RAEE05Settings:
    method_label: str
    implementation_sublabel: str
    audit: _AuditSettings
    bucket_specs: tuple[_BucketSpec, ...]
    candidate_max_demoted_beams: int
    candidate_step_p05_guardrail_margin: float


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


def _ra_ee_05_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("ra_ee_05_fixed_association_robustness", {})
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


def _bucket_specs_from_config(
    pilot: dict[str, Any],
    seeds: dict[str, Any],
) -> tuple[_BucketSpec, ...]:
    buckets = pilot.get("evaluation_buckets", {})
    if not isinstance(buckets, dict):
        buckets = {}

    calibration = buckets.get(CALIBRATION_BUCKET, {})
    if not isinstance(calibration, dict):
        calibration = {}
    held_out = buckets.get(HELD_OUT_BUCKET, buckets.get("held_out", {}))
    if not isinstance(held_out, dict):
        held_out = {}

    calibration_seeds = _tuple_ints(
        calibration.get("evaluation_seed_set"),
        _tuple_ints(seeds.get("evaluation_seed_set"), DEFAULT_CALIBRATION_SEEDS),
    )
    held_out_seeds = _tuple_ints(
        held_out.get("evaluation_seed_set"),
        DEFAULT_HELD_OUT_SEEDS,
    )
    calibration_families = _tuple_strings(
        calibration.get("trajectory_families"),
        CALIBRATION_TRAJECTORIES,
    )
    held_out_families = _tuple_strings(
        held_out.get("trajectory_families"),
        HELD_OUT_TRAJECTORIES,
    )

    specs = (
        _BucketSpec(
            name=CALIBRATION_BUCKET,
            trajectory_families=calibration_families,
            evaluation_seed_set=calibration_seeds,
        ),
        _BucketSpec(
            name=HELD_OUT_BUCKET,
            trajectory_families=held_out_families,
            evaluation_seed_set=held_out_seeds,
        ),
    )
    for spec in specs:
        if not spec.evaluation_seed_set:
            raise ValueError(f"RA-EE-05 bucket {spec.name!r} requires eval seeds.")
        unsupported = sorted(set(spec.trajectory_families) - set(RA_EE_05_TRAJECTORIES))
        if unsupported:
            raise ValueError(
                f"Unsupported RA-EE-05 trajectories in {spec.name!r}: {unsupported!r}"
            )
    return specs


def _settings_from_config(cfg: dict[str, Any]) -> _RAEE05Settings:
    pilot = _ra_ee_05_value(cfg)
    power = _power_surface_value(cfg)
    seeds = get_seeds(cfg)

    levels_raw = pilot.get(
        "codebook_levels_w",
        power.get("power_codebook_levels_w", [0.5, 0.75, 1.0, 1.5, 2.0]),
    )
    levels = tuple(float(level) for level in levels_raw)
    if not levels:
        raise ValueError("RA-EE-05 requires at least one power codebook level.")
    if tuple(sorted(levels)) != levels:
        raise ValueError(f"RA-EE-05 codebook levels must be sorted, got {levels!r}.")

    audit = _AuditSettings(
        method_label=str(pilot.get("method_label", RA_EE_05_METHOD_LABEL)),
        codebook_levels_w=levels,
        fixed_control_power_w=float(pilot.get("fixed_control_power_w", 1.0)),
        total_power_budget_w=float(
            pilot.get("total_active_power_budget_w", power.get("total_power_budget_w", 8.0))
        ),
        per_beam_max_power_w=float(
            pilot.get("per_beam_max_power_w", power.get("max_power_w", 2.0))
        ),
        active_base_power_w=float(
            pilot.get("active_base_power_w", power.get("active_base_power_w", 0.25))
        ),
        load_scale_power_w=float(
            pilot.get("load_scale_power_w", power.get("load_scale_power_w", 0.35))
        ),
        load_exponent=float(
            pilot.get("load_exponent", power.get("load_exponent", 0.5))
        ),
        p05_min_ratio_vs_control=float(
            pilot.get("p05_throughput_min_ratio_vs_control", 0.95)
        ),
        served_ratio_min_delta_vs_control=float(
            pilot.get("served_ratio_min_delta_vs_control", 0.0)
        ),
        outage_ratio_max_delta_vs_control=float(
            pilot.get("outage_ratio_max_delta_vs_control", 0.0)
        ),
        oracle_max_demoted_beams=int(pilot.get("oracle_max_demoted_beams", 3)),
    )
    return _RAEE05Settings(
        method_label=str(pilot.get("method_label", RA_EE_05_METHOD_LABEL)),
        implementation_sublabel=str(
            pilot.get(
                "implementation_sublabel",
                "RA-EE-05 fixed-association robustness and held-out validation",
            )
        ),
        audit=audit,
        bucket_specs=_bucket_specs_from_config(pilot, seeds),
        candidate_max_demoted_beams=int(
            pilot.get("candidate_max_demoted_beams", audit.oracle_max_demoted_beams)
        ),
        candidate_step_p05_guardrail_margin=float(
            pilot.get("candidate_step_p05_guardrail_margin", 0.0)
        ),
    )


def _power_vector_key(power_vector: np.ndarray, active_mask: np.ndarray) -> str:
    return _format_vector(power_vector[active_mask])


def _step_qos_guardrails_pass(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
    settings: _RAEE05Settings,
) -> bool:
    p05_ratio = (
        settings.audit.p05_min_ratio_vs_control
        + settings.candidate_step_p05_guardrail_margin
    )
    return (
        float(candidate["throughput_p05_user_step_bps"])
        >= p05_ratio * float(control["throughput_p05_user_step_bps"])
        and float(candidate["served_ratio"])
        >= float(control["served_ratio"])
        + settings.audit.served_ratio_min_delta_vs_control
        and float(candidate["outage_ratio"])
        <= float(control["outage_ratio"])
        + settings.audit.outage_ratio_max_delta_vs_control
    )


def _safe_greedy_power_vector(
    snapshot: _StepSnapshot,
    settings: _RAEE05Settings,
) -> tuple[np.ndarray, str]:
    audit = settings.audit
    control_vector = _power_vector_for_candidate(snapshot, audit, "fixed-control")
    control_row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile=f"fixed-{audit.fixed_control_power_w:g}w-control",
        power_vector=control_vector,
        settings=audit,
    )
    current = control_vector.copy()
    current_row = control_row
    active_indices = [int(idx) for idx in np.flatnonzero(snapshot.active_mask).tolist()]
    lower_levels = [
        float(level)
        for level in audit.codebook_levels_w
        if float(level) < audit.fixed_control_power_w
    ]
    demoted: set[int] = set()

    while len(demoted) < settings.candidate_max_demoted_beams:
        best_row: dict[str, Any] | None = None
        best_vector: np.ndarray | None = None
        best_idx: int | None = None
        for beam_idx in active_indices:
            if beam_idx in demoted:
                continue
            for level in lower_levels:
                requested = current.copy()
                requested[beam_idx] = float(level)
                row = _evaluate_power_vector(
                    snapshot=snapshot,
                    power_semantics=RA_EE_05_CANDIDATE,
                    selected_power_profile=(
                        f"safe-greedy:{_power_vector_key(requested, snapshot.active_mask)}"
                    ),
                    power_vector=requested,
                    settings=audit,
                )
                constraints_ok = (
                    not bool(row["budget_violation"])
                    and not bool(row["per_beam_power_violation"])
                    and not bool(row["inactive_beam_nonzero_power"])
                    and _step_qos_guardrails_pass(
                        candidate=row,
                        control=control_row,
                        settings=settings,
                    )
                )
                if not constraints_ok:
                    continue
                row_ee = float(row["EE_system_bps_per_w"] or -math.inf)
                current_ee = float(current_row["EE_system_bps_per_w"] or -math.inf)
                if row_ee <= current_ee + 1e-12:
                    continue
                if best_row is None:
                    best_row = row
                    best_vector = requested
                    best_idx = beam_idx
                    continue
                best_ee = float(best_row["EE_system_bps_per_w"] or -math.inf)
                if (
                    row_ee > best_ee + 1e-12
                    or (
                        abs(row_ee - best_ee) <= 1e-12
                        and float(row["sum_user_throughput_bps"])
                        > float(best_row["sum_user_throughput_bps"])
                    )
                ):
                    best_row = row
                    best_vector = requested
                    best_idx = beam_idx

        if best_row is None or best_vector is None or best_idx is None:
            break
        current = best_vector
        current_row = best_row
        demoted.add(best_idx)

    return current, f"safe-greedy:{_power_vector_key(current, snapshot.active_mask)}"


def _control_step_row(snapshot: _StepSnapshot, settings: _RAEE05Settings) -> dict[str, Any]:
    powers = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile=f"fixed-{settings.audit.fixed_control_power_w:g}w-control",
        power_vector=powers,
        settings=settings.audit,
    )
    row["requested_power_vector_w"] = _format_vector(powers)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_repair_used"] = False
    row["association_control"] = "fixed-by-trajectory"
    row["learned_association_enabled"] = False
    return row


def _candidate_step_row(snapshot: _StepSnapshot, settings: _RAEE05Settings) -> dict[str, Any]:
    requested, label = _safe_greedy_power_vector(snapshot, settings)
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_05_CANDIDATE,
        selected_power_profile=label,
        power_vector=requested,
        settings=settings.audit,
    )
    row["requested_power_vector_w"] = _format_vector(requested)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_repair_used"] = False
    row["association_control"] = "fixed-by-trajectory"
    row["learned_association_enabled"] = False
    return row


def _oracle_step_row(
    snapshot: _StepSnapshot,
    control_row: dict[str, Any],
    settings: _RAEE05Settings,
) -> dict[str, Any]:
    row = _select_oracle_step(
        snapshot=snapshot,
        control_row=control_row,
        settings=settings.audit,
    )
    row["power_semantics"] = RA_EE_05_ORACLE
    row["requested_power_vector_w"] = row["beam_transmit_power_w"]
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_repair_used"] = False
    row["association_control"] = "fixed-by-trajectory"
    row["learned_association_enabled"] = False
    return row


def _valid_indices(mask_obj: Any) -> np.ndarray:
    return np.flatnonzero(mask_obj.mask)


def _select_actions_for_policy(
    policy: str,
    *,
    current_assignments: np.ndarray,
    masks: list[Any],
    rng: np.random.Generator,
    step_index: int,
) -> np.ndarray:
    actions = np.zeros(len(masks), dtype=np.int32)
    assigned_counts = np.zeros_like(masks[0].mask, dtype=np.int32)

    def choose_hold(uid: int, valid: np.ndarray) -> int:
        current = int(current_assignments[uid])
        if 0 <= current < masks[uid].mask.size and bool(masks[uid].mask[current]):
            return current
        return int(valid[0])

    def choose_random(valid: np.ndarray, *, avoid_current: int | None = None) -> int:
        choices = valid
        if avoid_current is not None and valid.size > 1:
            filtered = valid[valid != avoid_current]
            if filtered.size:
                choices = filtered
        return int(rng.choice(choices))

    def choose_spread(valid: np.ndarray, *, reverse_tie: bool = False) -> int:
        best_load = int(np.min(assigned_counts[valid]))
        tied = valid[assigned_counts[valid] == best_load]
        return int(tied[-1] if reverse_tie else tied[0])

    def choose_skewed(uid: int, valid: np.ndarray) -> int:
        if valid.size == 1:
            return int(valid[0])
        preferred = int(valid[-1])
        secondary = int(valid[0])
        return preferred if (uid % 4) != 0 else secondary

    for uid, mask_obj in enumerate(masks):
        valid = _valid_indices(mask_obj)
        if valid.size == 0:
            actions[uid] = 0
            continue

        if policy == "hold-current":
            selected = choose_hold(uid, valid)
        elif policy in {"random-valid", "random-valid-heldout"}:
            selected = choose_random(valid)
        elif policy == "spread-valid":
            selected = choose_spread(valid)
        elif policy == "spread-valid-heldout":
            selected = choose_spread(valid, reverse_tie=True)
        elif policy == "load-skewed-heldout":
            selected = choose_skewed(uid, valid)
        elif policy == "mobility-shift-heldout":
            current = int(current_assignments[uid])
            if step_index % 4 == 0:
                selected = choose_random(valid, avoid_current=current)
            else:
                selected = choose_hold(uid, valid)
        elif policy == "mixed-valid-heldout":
            mode = step_index % 3
            if mode == 0:
                selected = choose_random(valid)
            elif mode == 1:
                selected = choose_spread(valid, reverse_tie=bool(uid % 2))
            else:
                selected = choose_skewed(uid, valid)
        else:
            raise ValueError(f"Unsupported RA-EE-05 trajectory policy {policy!r}.")

        actions[uid] = selected
        assigned_counts[selected] += 1

    return actions


def _rollout_fixed_association_trajectories(
    *,
    cfg: dict[str, Any],
    bucket_specs: tuple[_BucketSpec, ...],
    max_steps: int | None,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, str]]:
    trajectories: dict[str, dict[int, list[np.ndarray]]] = {}
    bucket_by_policy: dict[str, str] = {}
    for spec in bucket_specs:
        for policy in spec.trajectory_families:
            env = build_environment(cfg)
            policy_rows: dict[int, list[np.ndarray]] = defaultdict(list)
            for eval_seed in spec.evaluation_seed_set:
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
                    actions = _select_actions_for_policy(
                        policy,
                        current_assignments=env.current_assignments(),
                        masks=masks,
                        rng=policy_rng,
                        step_index=steps_seen,
                    )
                    result = env.step(actions, env_rng)
                    policy_rows[int(eval_seed)].append(actions.copy())
                    steps_seen += 1
                    if result.done:
                        break
                    masks = result.action_masks
            trajectories[policy] = policy_rows
            bucket_by_policy[policy] = spec.name
    return trajectories, bucket_by_policy


def _evaluation_rows(
    *,
    snapshots: list[_StepSnapshot],
    bucket_by_policy: dict[str, str],
    settings: _RAEE05Settings,
    include_oracle: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for snapshot in snapshots:
        control = _control_step_row(snapshot, settings)
        candidate = _candidate_step_row(snapshot, settings)
        step_rows = [control, candidate]
        if include_oracle:
            step_rows.append(_oracle_step_row(snapshot, control, settings))
        for row in step_rows:
            row["evaluation_bucket"] = bucket_by_policy[str(row["trajectory_policy"])]
            throughputs = row.pop("_user_throughputs")
            rows.append(row)
            user_throughputs_by_key[
                (str(row["trajectory_policy"]), str(row["power_semantics"]))
            ].extend(float(value) for value in throughputs.tolist())
    return rows, user_throughputs_by_key


def _augment_summaries(
    summaries: list[dict[str, Any]],
    *,
    step_rows: list[dict[str, Any]],
    bucket_by_policy: dict[str, str],
) -> list[dict[str, Any]]:
    vectors_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in step_rows:
        key = (str(row["trajectory_policy"]), str(row["power_semantics"]))
        vectors_by_key[key].append(str(row["effective_power_vector_w"]))

    for summary in summaries:
        policy = str(summary["trajectory_policy"])
        semantics = str(summary["power_semantics"])
        summary["evaluation_bucket"] = bucket_by_policy[policy]
        summary["selected_power_vector_distribution"] = _categorical_distribution(
            vectors_by_key[(policy, semantics)]
        )
    return summaries


def _comparison_ranking_checks(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_policy[str(row["trajectory_policy"])][str(row["power_semantics"])] = row

    checks: list[dict[str, Any]] = []
    for policy, rows in sorted(by_policy.items()):
        compared = [
            rows[key]
            for key in ("fixed-control", RA_EE_05_CANDIDATE)
            if key in rows
        ]
        if len(compared) < 2:
            continue
        throughput_ranking = [
            row["power_semantics"]
            for row in sorted(
                compared,
                key=lambda row: float(row["throughput_mean_user_step_bps"] or -math.inf),
                reverse=True,
            )
        ]
        ee_ranking = [
            row["power_semantics"]
            for row in sorted(
                compared,
                key=lambda row: float(row["EE_system_aggregate_bps_per_w"] or -math.inf),
                reverse=True,
            )
        ]
        checks.append(
            {
                "evaluation_bucket": compared[0]["evaluation_bucket"],
                "trajectory_policy": policy,
                "compared_power_semantics": ["fixed-control", RA_EE_05_CANDIDATE],
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
            }
        )
    return checks


def _guardrails_with_bucket(
    guardrail_checks: list[dict[str, Any]],
    *,
    bucket_by_policy: dict[str, str],
) -> list[dict[str, Any]]:
    for row in guardrail_checks:
        row["evaluation_bucket"] = bucket_by_policy[str(row["trajectory_policy"])]
    return guardrail_checks


def _oracle_gap_diagnostics(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_policy[str(row["trajectory_policy"])][str(row["power_semantics"])] = row

    diagnostics: list[dict[str, Any]] = []
    for policy, rows in sorted(by_policy.items()):
        candidate = rows.get(RA_EE_05_CANDIDATE)
        oracle = rows.get(RA_EE_05_ORACLE)
        if candidate is None or oracle is None:
            continue
        candidate_ee = candidate["EE_system_aggregate_bps_per_w"]
        oracle_ee = oracle["EE_system_aggregate_bps_per_w"]
        gap = None if candidate_ee is None or oracle_ee is None else float(oracle_ee) - float(candidate_ee)
        pct_gap = (
            None
            if gap is None or abs(float(candidate_ee)) < 1e-12
            else gap / abs(float(candidate_ee))
        )
        diagnostics.append(
            {
                "evaluation_bucket": candidate["evaluation_bucket"],
                "trajectory_policy": policy,
                "candidate_power_semantics": RA_EE_05_CANDIDATE,
                "oracle_power_semantics": RA_EE_05_ORACLE,
                "oracle_is_diagnostic_only": True,
                "candidate_EE_system_aggregate_bps_per_w": candidate_ee,
                "oracle_EE_system_aggregate_bps_per_w": oracle_ee,
                "oracle_EE_gap_vs_candidate_bps_per_w": gap,
                "oracle_EE_pct_gap_vs_candidate": pct_gap,
            }
        )
    return diagnostics


def _accepted_candidate_policies(
    *,
    bucket: str,
    candidate_by_policy: dict[str, dict[str, Any]],
    guardrail_by_policy: dict[str, dict[str, Any]],
    ranking_by_policy: dict[str, dict[str, Any]],
) -> list[str]:
    accepted: list[str] = []
    for policy, summary in sorted(candidate_by_policy.items()):
        if summary["evaluation_bucket"] != bucket:
            continue
        guardrail = guardrail_by_policy.get(policy, {})
        ranking = ranking_by_policy.get(policy, {})
        if not (
            bool(guardrail.get("accepted"))
            and bool(summary["denominator_varies_in_eval"])
            and int(summary["selected_profile_distinct_count"]) > 1
            and int(summary["selected_power_vector_distribution"]["distinct_count"]) > 1
            and len(summary["total_active_beam_power_w_distribution"]["distinct"]) > 1
            and float(summary["one_active_beam_step_ratio"]) < 1.0
            and bool(
                ranking.get(
                    "same_policy_throughput_rescore_vs_EE_rescore_top_changes"
                )
            )
        ):
            continue
        accepted.append(policy)
    return accepted


def _bucket_results(
    *,
    settings: _RAEE05Settings,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
    ranking_checks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_by_policy = {
        str(row["trajectory_policy"]): row
        for row in summaries
        if row["power_semantics"] == RA_EE_05_CANDIDATE
    }
    guardrail_by_policy = {
        str(row["trajectory_policy"]): row
        for row in guardrail_checks
        if row["power_semantics"] == RA_EE_05_CANDIDATE
    }
    ranking_by_policy = {
        str(row["trajectory_policy"]): row for row in ranking_checks
    }

    results: dict[str, dict[str, Any]] = {}
    for spec in settings.bucket_specs:
        policies = list(spec.trajectory_families)
        present = [policy for policy in policies if policy in candidate_by_policy]
        noncollapsed = [
            policy
            for policy in present
            if float(candidate_by_policy[policy]["one_active_beam_step_ratio"]) < 1.0
        ]
        positive = [
            policy
            for policy in noncollapsed
            if float(
                guardrail_by_policy.get(policy, {}).get(
                    "EE_system_delta_vs_fixed_control",
                    -math.inf,
                )
                or -math.inf
            )
            > 0.0
        ]
        accepted = _accepted_candidate_policies(
            bucket=spec.name,
            candidate_by_policy=candidate_by_policy,
            guardrail_by_policy=guardrail_by_policy,
            ranking_by_policy=ranking_by_policy,
        )
        accepted_set = set(accepted)
        positive_set = set(positive)

        no_power_violations = all(
            bool(guardrail_by_policy.get(policy, {}).get("budget_guardrail_pass"))
            and bool(guardrail_by_policy.get(policy, {}).get("per_beam_power_guardrail_pass"))
            and bool(guardrail_by_policy.get(policy, {}).get("inactive_beam_zero_w_guardrail_pass"))
            for policy in present
        )
        qos_pass_for_accepted = all(
            bool(guardrail_by_policy[policy]["p05_guardrail_pass"])
            and bool(guardrail_by_policy[policy]["served_ratio_guardrail_pass"])
            and bool(guardrail_by_policy[policy]["outage_guardrail_pass"])
            for policy in accepted
        )
        ranking_separates_for_accepted = all(
            bool(
                ranking_by_policy.get(policy, {}).get(
                    "same_policy_throughput_rescore_vs_EE_rescore_top_changes"
                )
            )
            for policy in accepted
        )

        results[spec.name] = {
            "bucket": spec.name,
            "evaluation_seed_set": list(spec.evaluation_seed_set),
            "trajectory_families": policies,
            "trajectory_count": len(policies),
            "present_trajectory_count": len(present),
            "noncollapsed_trajectory_count": len(noncollapsed),
            "noncollapsed_trajectories": noncollapsed,
            "positive_EE_delta_trajectory_count": len(positive),
            "positive_EE_delta_trajectories": positive,
            "accepted_candidate_trajectory_count": len(accepted),
            "accepted_candidate_trajectories": accepted,
            "positive_not_accepted_trajectories": sorted(positive_set - accepted_set),
            "majority_noncollapsed_positive_EE_delta": (
                bool(noncollapsed) and len(positive) > len(noncollapsed) / 2.0
            ),
            "majority_noncollapsed_accepted": (
                bool(noncollapsed) and len(accepted) > len(noncollapsed) / 2.0
            ),
            "gains_not_concentrated_in_one_trajectory": len(positive) >= 2,
            "qos_guardrails_pass_for_accepted": qos_pass_for_accepted,
            "zero_budget_per_beam_inactive_power_violations": no_power_violations,
            "denominator_varies_for_accepted": all(
                bool(candidate_by_policy[policy]["denominator_varies_in_eval"])
                for policy in accepted
            ),
            "selected_power_vectors_not_single_point_for_accepted": all(
                int(
                    candidate_by_policy[policy][
                        "selected_power_vector_distribution"
                    ]["distinct_count"]
                )
                > 1
                for policy in accepted
            ),
            "selected_profiles_not_single_point_for_accepted": all(
                int(candidate_by_policy[policy]["selected_profile_distinct_count"]) > 1
                for policy in accepted
            ),
            "total_active_power_not_single_point_for_accepted": all(
                len(
                    candidate_by_policy[policy][
                        "total_active_beam_power_w_distribution"
                    ]["distinct"]
                )
                > 1
                for policy in accepted
            ),
            "throughput_winner_vs_EE_winner_separate_for_accepted": (
                ranking_separates_for_accepted
            ),
            "one_active_beam_collapse_dominates": (
                len(present) > 0
                and len(present) - len(noncollapsed) > len(present) / 2.0
            ),
            "bucket_pass": (
                spec.name == CALIBRATION_BUCKET
                or (
                    bool(noncollapsed)
                    and len(positive) > len(noncollapsed) / 2.0
                    and len(accepted) > len(noncollapsed) / 2.0
                    and len(positive) >= 2
                    and qos_pass_for_accepted
                    and no_power_violations
                    and ranking_separates_for_accepted
                )
            ),
        }
    return results


def _build_decision(
    *,
    settings: _RAEE05Settings,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
    ranking_checks: list[dict[str, Any]],
    bucket_results: dict[str, dict[str, Any]],
    include_oracle: bool,
) -> dict[str, Any]:
    candidate_summaries = [
        row for row in summaries if row["power_semantics"] == RA_EE_05_CANDIDATE
    ]
    candidate_guardrails = [
        row for row in guardrail_checks if row["power_semantics"] == RA_EE_05_CANDIDATE
    ]
    held_out = bucket_results.get(HELD_OUT_BUCKET, {})
    held_out_present = HELD_OUT_BUCKET in bucket_results and bool(
        held_out.get("present_trajectory_count")
    )

    fixed_association_only = all(
        row.get("association_control") in (None, "fixed-by-trajectory")
        for row in candidate_summaries
    )
    learned_association_disabled = all(
        row.get("learned_association_enabled") in (None, False)
        for row in candidate_summaries
    )
    no_power_violations = all(
        bool(row.get("budget_guardrail_pass"))
        and bool(row.get("per_beam_power_guardrail_pass"))
        and bool(row.get("inactive_beam_zero_w_guardrail_pass"))
        for row in candidate_guardrails
    )

    proof_flags = {
        "held_out_bucket_exists_and_reported_separately": held_out_present,
        "fixed_association_only": fixed_association_only,
        "learned_association_disabled": learned_association_disabled,
        "catfish_disabled": True,
        "multi_catfish_disabled": True,
        "joint_association_power_training_disabled": True,
        "old_EE_MODQN_continuation_disabled": True,
        "frozen_baseline_mutation": False,
        "majority_noncollapsed_held_out_positive_EE_delta": bool(
            held_out.get("majority_noncollapsed_positive_EE_delta")
        ),
        "majority_noncollapsed_held_out_accepted": bool(
            held_out.get("majority_noncollapsed_accepted")
        ),
        "held_out_gains_not_concentrated_in_one_trajectory": bool(
            held_out.get("gains_not_concentrated_in_one_trajectory")
        ),
        "p05_throughput_guardrail_pass_for_accepted_held_out": bool(
            held_out.get("qos_guardrails_pass_for_accepted")
        ),
        "served_ratio_does_not_drop_for_accepted_held_out": bool(
            held_out.get("qos_guardrails_pass_for_accepted")
        ),
        "outage_ratio_does_not_increase_for_accepted_held_out": bool(
            held_out.get("qos_guardrails_pass_for_accepted")
        ),
        "zero_budget_per_beam_inactive_power_violations": no_power_violations,
        "denominator_varies_for_accepted_held_out": bool(
            held_out.get("denominator_varies_for_accepted")
        ),
        "selected_power_vectors_not_single_point_for_accepted_held_out": bool(
            held_out.get("selected_power_vectors_not_single_point_for_accepted")
        ),
        "selected_profiles_not_single_point_for_accepted_held_out": bool(
            held_out.get("selected_profiles_not_single_point_for_accepted")
        ),
        "total_active_power_not_single_point_for_accepted_held_out": bool(
            held_out.get("total_active_power_not_single_point_for_accepted")
        ),
        "throughput_winner_vs_EE_winner_separate_for_accepted_held_out": bool(
            held_out.get("throughput_winner_vs_EE_winner_separate_for_accepted")
        ),
        "oracle_upper_bound_diagnostic_only": include_oracle,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
        "physical_energy_saving_claim": False,
        "hobs_optimizer_claim": False,
    }

    positive_qos_fail = any(
        row["evaluation_bucket"] == HELD_OUT_BUCKET
        and float(row["EE_system_delta_vs_fixed_control"] or 0.0) > 0.0
        and not bool(row["QoS_guardrails_pass"])
        for row in candidate_guardrails
    )
    ranking_missing = (
        bool(held_out.get("accepted_candidate_trajectory_count"))
        and not bool(
            held_out.get("throughput_winner_vs_EE_winner_separate_for_accepted")
        )
    )
    denominator_fixed_for_accepted = (
        bool(held_out.get("accepted_candidate_trajectory_count"))
        and not bool(held_out.get("denominator_varies_for_accepted"))
    )

    stop_conditions = {
        "held_out_bucket_missing": not held_out_present,
        "held_out_gains_disappear_or_concentrate_in_one_trajectory": not bool(
            held_out.get("gains_not_concentrated_in_one_trajectory")
        ),
        "p05_throughput_guardrail_fails": positive_qos_fail,
        "denominator_becomes_fixed": denominator_fixed_for_accepted,
        "one_active_beam_collapse_dominates": bool(
            held_out.get("one_active_beam_collapse_dominates")
        ),
        "budget_or_inactive_power_violations": not no_power_violations,
        "ranking_no_longer_separates": ranking_missing,
        "candidate_only_improves_scalar_reward": False,
        "learned_association_added": not learned_association_disabled,
        "catfish_added": False,
        "frozen_baseline_mutated": False,
        "oracle_used_as_candidate_claim": False,
    }

    required_true_fields = (
        "held_out_bucket_exists_and_reported_separately",
        "fixed_association_only",
        "learned_association_disabled",
        "catfish_disabled",
        "multi_catfish_disabled",
        "joint_association_power_training_disabled",
        "old_EE_MODQN_continuation_disabled",
        "majority_noncollapsed_held_out_positive_EE_delta",
        "majority_noncollapsed_held_out_accepted",
        "held_out_gains_not_concentrated_in_one_trajectory",
        "p05_throughput_guardrail_pass_for_accepted_held_out",
        "served_ratio_does_not_drop_for_accepted_held_out",
        "outage_ratio_does_not_increase_for_accepted_held_out",
        "zero_budget_per_beam_inactive_power_violations",
        "denominator_varies_for_accepted_held_out",
        "selected_power_vectors_not_single_point_for_accepted_held_out",
        "selected_profiles_not_single_point_for_accepted_held_out",
        "total_active_power_not_single_point_for_accepted_held_out",
        "throughput_winner_vs_EE_winner_separate_for_accepted_held_out",
        "oracle_upper_bound_diagnostic_only",
    )
    pass_required = all(bool(proof_flags[field]) for field in required_true_fields)
    pass_required = (
        pass_required
        and proof_flags["scalar_reward_success_basis"] is False
        and proof_flags["per_user_EE_credit_success_basis"] is False
        and proof_flags["physical_energy_saving_claim"] is False
        and proof_flags["hobs_optimizer_claim"] is False
    )

    if any(bool(value) for value in stop_conditions.values()):
        decision = "BLOCKED"
    elif pass_required:
        decision = "PASS"
    else:
        decision = "NEEDS MORE EVIDENCE"

    return {
        "ra_ee_05_decision": decision,
        "proof_flags": proof_flags,
        "stop_conditions": stop_conditions,
        "candidate_guardrail_checks": candidate_guardrails,
        "allowed_claim": (
            "PASS only means fixed-association centralized power-allocation "
            "robustness passed the held-out gate. It is not full RA-EE-MODQN."
            if decision == "PASS"
            else "Do not promote RA-EE-05 beyond fixed-association robustness evidence."
        ),
    }


def _compact_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summaries:
        rows.append(
            {
                "evaluation_bucket": row["evaluation_bucket"],
                "trajectory_policy": row["trajectory_policy"],
                "power_semantics": row["power_semantics"],
                "EE_system_aggregate_bps_per_w": row["EE_system_aggregate_bps_per_w"],
                "EE_system_step_mean_bps_per_w": row["EE_system_step_mean_bps_per_w"],
                "throughput_mean_user_step_bps": row["throughput_mean_user_step_bps"],
                "throughput_p05_user_step_bps": row["throughput_p05_user_step_bps"],
                "served_ratio": row["served_ratio"],
                "outage_ratio": row["outage_ratio"],
                "handover_count": row["handover_count"],
                "active_beam_count_distribution": row["active_beam_count_distribution"],
                "selected_power_profile_distribution": row[
                    "selected_power_profile_distribution"
                ],
                "selected_power_vector_distribution": row[
                    "selected_power_vector_distribution"
                ],
                "total_active_beam_power_w_distribution": row[
                    "total_active_beam_power_w_distribution"
                ],
                "denominator_varies_in_eval": row["denominator_varies_in_eval"],
                "one_active_beam_step_ratio": row["one_active_beam_step_ratio"],
                "budget_violation_step_count": row["budget_violations"]["step_count"],
                "per_beam_power_violation_step_count": row[
                    "per_beam_power_violations"
                ]["step_count"],
                "inactive_beam_nonzero_power_step_count": row[
                    "inactive_beam_nonzero_power_step_count"
                ],
                "throughput_EE_pearson": row[
                    "throughput_vs_EE_system_correlation"
                ]["pearson"],
                "throughput_EE_spearman": row[
                    "throughput_vs_EE_system_correlation"
                ]["spearman"],
            }
        )
    return rows


def _trajectory_family_descriptions() -> list[dict[str, str]]:
    return [
        {
            "trajectory_family": "hold-current",
            "bucket": CALIBRATION_BUCKET,
            "association_rule": "hold current beam when still valid",
            "diversity_axis": "RA-EE-04 train-like stability control",
        },
        {
            "trajectory_family": "random-valid",
            "bucket": CALIBRATION_BUCKET,
            "association_rule": "uniform random valid beam",
            "diversity_axis": "RA-EE-04 train-like randomized association pattern",
        },
        {
            "trajectory_family": "spread-valid",
            "bucket": CALIBRATION_BUCKET,
            "association_rule": "least-assigned valid beam with low-index tie break",
            "diversity_axis": "RA-EE-04 train-like load spreading",
        },
        {
            "trajectory_family": "random-valid-heldout",
            "bucket": HELD_OUT_BUCKET,
            "association_rule": "uniform random valid beam with held-out seeds",
            "diversity_axis": "association randomness and held-out mobility seeds",
        },
        {
            "trajectory_family": "spread-valid-heldout",
            "bucket": HELD_OUT_BUCKET,
            "association_rule": "least-assigned valid beam with high-index tie break",
            "diversity_axis": "association spread pattern shift",
        },
        {
            "trajectory_family": "load-skewed-heldout",
            "bucket": HELD_OUT_BUCKET,
            "association_rule": "prefer one valid beam with periodic secondary beam",
            "diversity_axis": "load skew",
        },
        {
            "trajectory_family": "mobility-shift-heldout",
            "bucket": HELD_OUT_BUCKET,
            "association_rule": "hold current except periodic valid reassignment",
            "diversity_axis": "handover and mobility shift stress",
        },
        {
            "trajectory_family": "mixed-valid-heldout",
            "bucket": HELD_OUT_BUCKET,
            "association_rule": "cycle random, spread, and skewed valid selections",
            "diversity_axis": "mixed association pattern",
        },
    ]


def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_05_decision"]
    proof = summary["proof_flags"]
    held_out = summary["bucket_results"][HELD_OUT_BUCKET]
    lines = [
        "# RA-EE-05 Fixed-Association Robustness Review",
        "",
        "Fixed-association centralized power-allocation robustness and held-out "
        "validation only. No learned association, joint association + power "
        "training, Catfish, multi-Catfish, RB / bandwidth allocation, old "
        "EE-MODQN continuation, HOBS optimizer claim, physical energy-saving "
        "claim, or frozen baseline mutation was performed.",
        "",
        "## Protocol",
        "",
        f"- method label: `{summary['protocol']['method_label']}`",
        f"- implementation sublabel: `{summary['protocol']['implementation_sublabel']}`",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- calibration seeds: `{summary['bucket_results'][CALIBRATION_BUCKET]['evaluation_seed_set']}`",
        f"- held-out seeds: `{held_out['evaluation_seed_set']}`",
        f"- calibration trajectories: `{summary['bucket_results'][CALIBRATION_BUCKET]['trajectory_families']}`",
        f"- held-out trajectories: `{held_out['trajectory_families']}`",
        f"- candidate: `{summary['protocol']['candidate_allocator']}`",
        f"- control: `{summary['protocol']['control_allocator']}`",
        f"- oracle: `{summary['protocol']['oracle_upper_bound']}`",
        "",
        "## Held-Out Gate",
        "",
        f"- noncollapsed held-out trajectories: `{held_out['noncollapsed_trajectories']}`",
        f"- positive EE delta trajectories: `{held_out['positive_EE_delta_trajectories']}`",
        f"- accepted candidate trajectories: `{held_out['accepted_candidate_trajectories']}`",
        f"- majority noncollapsed positive EE delta: `{held_out['majority_noncollapsed_positive_EE_delta']}`",
        f"- majority noncollapsed accepted: `{held_out['majority_noncollapsed_accepted']}`",
        f"- gains not concentrated in one trajectory: `{held_out['gains_not_concentrated_in_one_trajectory']}`",
        f"- QoS guardrails pass for accepted: `{held_out['qos_guardrails_pass_for_accepted']}`",
        f"- zero budget / per-beam / inactive-power violations: `{held_out['zero_budget_per_beam_inactive_power_violations']}`",
        f"- ranking separates for accepted: `{held_out['throughput_winner_vs_EE_winner_separate_for_accepted']}`",
        "",
        "## Gate Flags",
        "",
    ]
    for key, value in proof.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- RA-EE-05 decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def export_ra_ee_05_fixed_association_robustness(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    calibration_seed_set: tuple[int, ...] | None = None,
    held_out_seed_set: tuple[int, ...] | None = None,
    calibration_policies: tuple[str, ...] | None = None,
    held_out_policies: tuple[str, ...] | None = None,
    max_steps: int | None = None,
    include_oracle: bool = True,
) -> dict[str, Any]:
    """Export RA-EE-05 fixed-association robustness artifacts."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")

    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    if env.power_surface_config.hobs_power_surface_mode != (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-05 config must opt into the power-codebook surface.")

    settings = _settings_from_config(cfg)
    bucket_specs: list[_BucketSpec] = []
    for spec in settings.bucket_specs:
        if spec.name == CALIBRATION_BUCKET:
            bucket_specs.append(
                _BucketSpec(
                    name=spec.name,
                    trajectory_families=(
                        tuple(calibration_policies)
                        if calibration_policies is not None
                        else spec.trajectory_families
                    ),
                    evaluation_seed_set=(
                        tuple(calibration_seed_set)
                        if calibration_seed_set is not None
                        else spec.evaluation_seed_set
                    ),
                )
            )
        elif spec.name == HELD_OUT_BUCKET:
            bucket_specs.append(
                _BucketSpec(
                    name=spec.name,
                    trajectory_families=(
                        tuple(held_out_policies)
                        if held_out_policies is not None
                        else spec.trajectory_families
                    ),
                    evaluation_seed_set=(
                        tuple(held_out_seed_set)
                        if held_out_seed_set is not None
                        else spec.evaluation_seed_set
                    ),
                )
            )
    run_settings = _RAEE05Settings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        bucket_specs=tuple(bucket_specs),
        candidate_max_demoted_beams=settings.candidate_max_demoted_beams,
        candidate_step_p05_guardrail_margin=(
            settings.candidate_step_p05_guardrail_margin
        ),
    )

    trajectories, bucket_by_policy = _rollout_fixed_association_trajectories(
        cfg=cfg,
        bucket_specs=run_settings.bucket_specs,
        max_steps=max_steps,
    )
    snapshots = _build_unit_power_snapshots(
        base_cfg=cfg,
        settings=run_settings.audit,
        trajectories=trajectories,
    )
    step_rows, user_throughputs_by_key = _evaluation_rows(
        snapshots=snapshots,
        bucket_by_policy=bucket_by_policy,
        settings=run_settings,
        include_oracle=include_oracle,
    )
    summaries = _summarize_all(
        rows=step_rows,
        user_throughputs_by_key=user_throughputs_by_key,
    )
    summaries = _augment_summaries(
        summaries,
        step_rows=step_rows,
        bucket_by_policy=bucket_by_policy,
    )
    all_ranking_checks = _build_ranking_checks(summaries)
    comparison_ranking_checks = _comparison_ranking_checks(summaries)
    guardrail_checks = _build_guardrail_checks(
        candidate_summaries=summaries,
        settings=run_settings.audit,
    )
    guardrail_checks = _guardrails_with_bucket(
        guardrail_checks,
        bucket_by_policy=bucket_by_policy,
    )
    oracle_gap_diagnostics = _oracle_gap_diagnostics(summaries)
    bucket_results = _bucket_results(
        settings=run_settings,
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        ranking_checks=comparison_ranking_checks,
    )
    decision_detail = _build_decision(
        settings=run_settings,
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        ranking_checks=comparison_ranking_checks,
        bucket_results=bucket_results,
        include_oracle=include_oracle,
    )

    out_dir = Path(output_dir)
    step_csv = _write_csv(
        out_dir / "ra_ee_05_step_metrics.csv",
        step_rows,
        fieldnames=list(step_rows[0].keys()),
    )
    compact_rows = _compact_summary_rows(summaries)
    summary_csv = _write_csv(
        out_dir / "ra_ee_05_candidate_summary.csv",
        compact_rows,
        fieldnames=list(compact_rows[0].keys()),
    )

    protocol = {
        "phase": "RA-EE-05",
        "method_label": run_settings.method_label,
        "method_family": "RA-EE fixed-association power allocation",
        "implementation_sublabel": run_settings.implementation_sublabel,
        "training": "none; evaluation-only fixed association trajectories",
        "learned_association": "disabled",
        "association_control": "fixed-by-trajectory",
        "joint_association_power_training": "disabled",
        "catfish": "disabled",
        "multi_catfish": "disabled",
        "rb_bandwidth_allocation": "disabled/not-modeled",
        "old_EE_MODQN_continuation": "forbidden/not-performed",
        "frozen_baseline_mutation": "forbidden/not-performed",
        "hobs_optimizer_claim": "forbidden/not-made",
        "physical_energy_saving_claim": "forbidden/not-made",
        "candidate_action_contract": (
            "centralized per-active-beam discrete power vector; inactive beams 0 W"
        ),
        "control_allocator": "fixed-control-1w-per-active-beam",
        "candidate_allocator": RA_EE_05_CANDIDATE,
        "oracle_upper_bound": RA_EE_05_ORACLE,
        "oracle_upper_bound_diagnostic_only": include_oracle,
        "system_EE_primary": True,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
    }
    constraints = {
        "per_beam_max_power_w": run_settings.audit.per_beam_max_power_w,
        "total_power_budget_w": run_settings.audit.total_power_budget_w,
        "inactive_beam_policy": "zero-w",
        "codebook_levels_w": list(run_settings.audit.codebook_levels_w),
        "p05_throughput_min_ratio_vs_control": (
            run_settings.audit.p05_min_ratio_vs_control
        ),
        "served_ratio_min_delta_vs_control": (
            run_settings.audit.served_ratio_min_delta_vs_control
        ),
        "outage_ratio_max_delta_vs_control": (
            run_settings.audit.outage_ratio_max_delta_vs_control
        ),
        "candidate_max_demoted_beams": run_settings.candidate_max_demoted_beams,
        "candidate_step_p05_guardrail_margin": (
            run_settings.candidate_step_p05_guardrail_margin
        ),
        "power_repair": "not-used; requested and effective vectors are exported",
        "effective_power_vector_contract": (
            "same effective_power_vector_w feeds SINR numerator, throughput, "
            "EE denominator, audit logs, and budget checks"
        ),
    }
    summary = {
        "inputs": {
            "config_path": str(Path(config_path)),
            "output_dir": str(out_dir),
            "max_steps": max_steps,
        },
        "protocol": protocol,
        "constraints": constraints,
        "trajectory_family_descriptions": _trajectory_family_descriptions(),
        "candidate_summaries": summaries,
        "guardrail_checks": guardrail_checks,
        "ranking_separation_result": {
            "comparison_control_vs_candidate": comparison_ranking_checks,
            "all_power_semantics": all_ranking_checks,
        },
        "bucket_results": bucket_results,
        "oracle_gap_diagnostics": oracle_gap_diagnostics,
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "stop_conditions": decision_detail["stop_conditions"],
        "ra_ee_05_decision": decision_detail["ra_ee_05_decision"],
        "remaining_blockers": [
            "This is fixed-association centralized power-allocation robustness evidence only.",
            "No learned association or full RA-EE-MODQN policy exists.",
            "No RB / bandwidth allocation is included.",
            "The oracle upper bound is diagnostic only, not the candidate claim.",
            "A PASS does not claim HOBS optimizer behavior or physical energy saving.",
        ],
        "forbidden_claims_still_active": [
            "Do not call RA-EE-05 full RA-EE-MODQN.",
            "Do not claim learned association effectiveness.",
            "Do not claim joint association + power training.",
            "Do not claim old EE-MODQN effectiveness.",
            "Do not claim HOBS optimizer behavior.",
            "Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.",
            "Do not add or claim RB / bandwidth allocation.",
            "Do not treat per-user EE credit as system EE.",
            "Do not use scalar reward alone as success evidence.",
            "Do not claim full paper-faithful reproduction.",
            "Do not claim physical energy saving.",
        ],
    }

    summary_path = write_json(
        out_dir / "ra_ee_05_fixed_association_robustness_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "ra_ee_05_fixed_association_robustness_summary": summary_path,
        "ra_ee_05_candidate_summary_csv": summary_csv,
        "ra_ee_05_step_metrics": step_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = [
    "CALIBRATION_BUCKET",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "HELD_OUT_BUCKET",
    "RA_EE_05_CANDIDATE",
    "RA_EE_05_METHOD_LABEL",
    "RA_EE_05_ORACLE",
    "export_ra_ee_05_fixed_association_robustness",
]
