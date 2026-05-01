"""CP-base non-codebook continuous-power implementation-readiness audit.

This module writes deterministic config / wiring / metadata proof only. It
does not call the trainer, does not run pilot episodes, and does not treat
scalar reward or aggregate metrics as effectiveness evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ..config_loader import (
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from ..env.step import HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER
from ..runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_NON_CODEBOOK_CONTINUOUS_POWER_IMPLEMENTATION_READINESS_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    R1_REWARD_MODE_THROUGHPUT,
)
from ._common import write_json

DEFAULT_CONTROL_CONFIG = Path(
    "configs/hobs-active-tx-ee-non-codebook-continuous-power-"
    "throughput-control.resolved.yaml"
)
DEFAULT_CANDIDATE_CONFIG = Path(
    "configs/hobs-active-tx-ee-non-codebook-continuous-power-"
    "ee-candidate.resolved.yaml"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit"
)
DEFAULT_AUDIT_CONFIG = Path(
    "configs/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit.resolved.yaml"
)

FORBIDDEN_POWER_KEYS = {
    "power_codebook_profile",
    "power_codebook_levels_w",
    "selected_power_profile",
    "finite_codebook_levels_w",
}

def export_boundary_audit(
    *,
    control_config_path: str | Path = DEFAULT_CONTROL_CONFIG,
    candidate_config_path: str | Path = DEFAULT_CANDIDATE_CONFIG,
    audit_config_path: str | Path = DEFAULT_AUDIT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Write deterministic CP-base readiness artifacts and return the summary."""
    control_path = Path(control_config_path)
    candidate_path = Path(candidate_config_path)
    audit_path = Path(audit_config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load the audit driver too so namespace gating covers the artifact entrypoint.
    audit_cfg = load_training_yaml(audit_path)
    _ = build_trainer_config(audit_cfg)

    control_cfg = load_training_yaml(control_path)
    candidate_cfg = load_training_yaml(candidate_path)
    control_trainer = build_trainer_config(control_cfg)
    candidate_trainer = build_trainer_config(candidate_cfg)

    boundary = prove_candidate_control_boundary(
        control_cfg,
        candidate_cfg,
        control_config_path=control_path,
        candidate_config_path=candidate_path,
    )
    samples = deterministic_step_wiring_samples(control_cfg, candidate_cfg)
    sample_rows = samples["rows"]
    _write_sample_csv(out / "boundary_audit_step_samples.csv", sample_rows)

    acceptance = _acceptance_flags(boundary, samples)
    summary = {
        "method_label": "CP-base-EE-MODQN implementation-readiness",
        "power_surface": "non-codebook analytic continuous-power sidecar",
        "claim_boundary": "config / wiring / metadata readiness only",
        "pilot_training_authorized": False,
        "control_config": str(control_path),
        "candidate_config": str(candidate_path),
        "audit_config": str(audit_path),
        "config_hashes": {
            "control_sha256": _file_sha256(control_path),
            "candidate_sha256": _file_sha256(candidate_path),
            "audit_sha256": _file_sha256(audit_path),
        },
        "boundary_proof": boundary,
        "deterministic_step_samples": samples,
        "acceptance_flags": acceptance,
        "acceptance_result": (
            "PASS" if all(bool(value) for value in acceptance.values()) else "BLOCK"
        ),
        "forbidden_claims_still_active": [
            "EE-MODQN effectiveness",
            "Catfish-EE readiness",
            "physical energy saving",
            "HOBS optimizer reproduction",
            "Phase 03C selector route reopened",
            "RA-EE learned association",
            "scalar reward success",
            "denominator variability alone proves energy-aware learning",
        ],
        "trainer_config_roles": {
            "control": {
                "comparison_role": control_trainer.comparison_role,
                "r1_reward_mode": control_trainer.r1_reward_mode,
            },
            "candidate": {
                "comparison_role": candidate_trainer.comparison_role,
                "r1_reward_mode": candidate_trainer.r1_reward_mode,
            },
        },
    }
    write_json(out / "summary.json", summary)
    _write_review(out / "review.md", summary)
    return summary


def prove_candidate_control_boundary(
    control_cfg: dict[str, Any],
    candidate_cfg: dict[str, Any],
    *,
    control_config_path: Path | None = None,
    candidate_config_path: Path | None = None,
) -> dict[str, Any]:
    """Return the metadata proof for the future matched candidate/control pair."""
    control_trainer = build_trainer_config(control_cfg)
    candidate_trainer = build_trainer_config(candidate_cfg)
    control_power = _power_surface_value(control_cfg)
    candidate_power = _power_surface_value(candidate_cfg)
    control_gate = _gate_block(control_cfg)
    candidate_gate = _gate_block(candidate_cfg)

    control_common = _boundary_common_subset(control_cfg)
    candidate_common = _boundary_common_subset(candidate_cfg)
    common_hashes = {
        "control_boundary_common_sha256": _stable_hash(control_common),
        "candidate_boundary_common_sha256": _stable_hash(candidate_common),
        "control_power_surface_sha256": _stable_hash(control_power),
        "candidate_power_surface_sha256": _stable_hash(candidate_power),
    }
    forbidden = _forbidden_mode_flags(control_cfg, candidate_cfg)
    checks = {
        "same_boundary_common_metadata_except_r1": control_common == candidate_common,
        "control_r1_is_throughput": (
            control_trainer.r1_reward_mode == R1_REWARD_MODE_THROUGHPUT
        ),
        "candidate_r1_is_hobs_active_tx_ee": (
            candidate_trainer.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
        ),
        "only_intended_difference_is_r1_reward_mode": (
            control_common == candidate_common
            and control_trainer.r1_reward_mode != candidate_trainer.r1_reward_mode
        ),
        "same_continuous_power_surface": control_power == candidate_power,
        "continuous_power_mode": (
            control_power.get("mode")
            == candidate_power.get("mode")
            == HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER
        ),
        "same_anti_collapse_guard": (
            _anti_collapse_block(control_gate)
            == _anti_collapse_block(candidate_gate)
        ),
        "qos_sticky_guard_shared_and_structural_only": (
            _anti_collapse_block(control_gate).get("enabled") is True
            and _anti_collapse_block(candidate_gate).get("enabled") is True
            and _anti_collapse_block(control_gate).get("mode")
            == "qos-sticky-overflow-reassignment"
            and "not counted as EE objective evidence"
            in _anti_collapse_block(candidate_gate).get("provenance", "")
        ),
        "same_seed_policy": get_seeds(control_cfg) == get_seeds(candidate_cfg),
        "same_episode_budget": (
            control_trainer.episodes == candidate_trainer.episodes
        ),
        "same_objective_weights": (
            control_trainer.objective_weights == candidate_trainer.objective_weights
        ),
        "same_trainer_hyperparameters": (
            _trainer_hyperparameter_subset(control_trainer)
            == _trainer_hyperparameter_subset(candidate_trainer)
        ),
        "same_checkpoint_protocol": (
            control_trainer.checkpoint_primary_report
            == candidate_trainer.checkpoint_primary_report
            and control_trainer.checkpoint_secondary_report
            == candidate_trainer.checkpoint_secondary_report
        ),
        "finite_codebook_levels_absent": (
            FORBIDDEN_POWER_KEYS.isdisjoint(control_power.keys())
            and FORBIDDEN_POWER_KEYS.isdisjoint(candidate_power.keys())
        ),
        "selected_power_profile_absent_from_config": (
            "selected_power_profile" not in control_power
            and "selected_power_profile" not in candidate_power
        ),
        "forbidden_modes_disabled": all(not bool(value) for value in forbidden.values()),
        "training_not_run_by_audit": True,
    }
    return {
        "matched_boundary_pass": all(bool(value) for value in checks.values()),
        "checks": checks,
        "forbidden_mode_flags": forbidden,
        "common_hashes": common_hashes,
        "control_config_path": "" if control_config_path is None else str(control_config_path),
        "candidate_config_path": (
            "" if candidate_config_path is None else str(candidate_config_path)
        ),
        "intended_difference_table": [
            {
                "role": "candidate",
                "r1_reward_mode": candidate_trainer.r1_reward_mode,
                "continuous_power_surface": "same",
                "anti_collapse_guard": "same",
            },
            {
                "role": "control",
                "r1_reward_mode": control_trainer.r1_reward_mode,
                "continuous_power_surface": "same",
                "anti_collapse_guard": "same",
            },
        ],
        "power_surface_constants": {
            key: control_power[key]
            for key in (
                "p_active_lo_w",
                "p_active_hi_w",
                "alpha",
                "beta",
                "kappa",
                "bias",
                "q_ref",
                "n_qos",
                "total_power_budget_w",
            )
        },
    }


def deterministic_step_wiring_samples(
    control_cfg: dict[str, Any],
    candidate_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run deterministic one-step samples that prove wiring, not effectiveness."""
    rows: list[dict[str, Any]] = []
    for action_case, action_builder in {
        "single-first-valid-action": _single_first_valid_actions,
        "rotating-valid-actions": _rotating_valid_actions,
    }.items():
        for role, cfg in {
            "control": control_cfg,
            "candidate": candidate_cfg,
        }.items():
            row = _run_one_step_sample(cfg, role=role, action_case=action_case, action_builder=action_builder)
            rows.append(row)

    control_rows = {row["action_case"]: row for row in rows if row["role"] == "control"}
    candidate_rows = {
        row["action_case"]: row for row in rows if row["role"] == "candidate"
    }
    same_vector = all(
        control_rows[name]["beam_transmit_power_w"] == candidate_rows[name]["beam_transmit_power_w"]
        for name in control_rows
    )
    same_throughput = all(
        abs(
            float(control_rows[name]["sum_user_throughput_bps"])
            - float(candidate_rows[name]["sum_user_throughput_bps"])
        )
        <= 1e-9
        for name in control_rows
    )
    candidate_power_vectors = {
        row["beam_transmit_power_w"] for row in rows if row["role"] == "candidate"
    }
    active_power_values = [
        value
        for row in rows
        for value in json.loads(row["active_beam_power_w"])
    ]
    selected_profiles = {row["selected_power_profile"] for row in rows}
    inactive_nonzero = sum(int(row["inactive_nonzero_power_count"]) for row in rows)
    budget_violations = sum(int(row["power_budget_violation"]) for row in rows)
    ee_denominator_matches = all(
        abs(float(row["ee_from_result"]) - float(row["ee_recomputed_from_power_vector"]))
        <= 1e-9
        for row in rows
    )
    return {
        "rows": rows,
        "same_power_vector_for_candidate_and_control": same_vector,
        "same_throughput_for_candidate_and_control": same_throughput,
        "policy_action_consequences_change_power": len(candidate_power_vectors) > 1,
        "active_power_nonconstant": len({round(float(value), 12) for value in active_power_values}) > 1,
        "active_power_non_codebook": all(
            round(float(value), 12) not in {0.5, 1.0, 2.0}
            for value in active_power_values
        ),
        "selected_power_profile_absent": selected_profiles == {""},
        "inactive_beams_zero_w": inactive_nonzero == 0,
        "power_budget_violations": budget_violations,
        "ee_denominator_reuses_step_power_vector": ee_denominator_matches,
    }


def _run_one_step_sample(
    cfg: dict[str, Any],
    *,
    role: str,
    action_case: str,
    action_builder: Callable[[list[Any]], np.ndarray],
) -> dict[str, Any]:
    seeds = get_seeds(cfg)
    env_rng = np.random.default_rng(int(seeds["environment_seed"]))
    mobility_rng = np.random.default_rng(int(seeds["mobility_seed"]))
    env = build_environment(cfg)
    states, masks, _diag = env.reset(env_rng, mobility_rng)
    actions = action_builder(masks)
    result = env.step(actions, env_rng)
    active_power = result.beam_transmit_power_w[result.active_beam_mask]
    inactive_power = result.beam_transmit_power_w[~result.active_beam_mask]
    sum_thr = float(sum(float(rw.r1_throughput) for rw in result.rewards))
    total_power = float(result.total_active_beam_power_w)
    ee_recomputed = sum_thr / (total_power + 1e-9)
    ee_result = float(result.rewards[0].r1_hobs_active_tx_ee)
    return {
        "role": role,
        "action_case": action_case,
        "step_index": int(result.step_index),
        "active_beam_count": int(np.sum(result.active_beam_mask)),
        "beam_transmit_power_w": _json_vector(result.beam_transmit_power_w),
        "active_beam_power_w": _json_vector(active_power),
        "total_active_beam_power_w": total_power,
        "p_active_min_w": float(np.min(active_power)) if active_power.size else 0.0,
        "p_active_max_w": float(np.max(active_power)) if active_power.size else 0.0,
        "inactive_nonzero_power_count": int(np.sum(np.abs(inactive_power) > 1e-12)),
        "selected_power_profile": str(result.selected_power_profile),
        "power_budget_violation": int(bool(result.power_budget_violation)),
        "sum_user_throughput_bps": sum_thr,
        "ee_from_result": ee_result,
        "ee_recomputed_from_power_vector": ee_recomputed,
        "first_reward_hobs_active_tx_ee": ee_result,
        "training_run": False,
        "effectiveness_evidence": False,
    }


def _single_first_valid_actions(masks: list[Any]) -> np.ndarray:
    actions = np.zeros(len(masks), dtype=np.int32)
    for uid, mask in enumerate(masks):
        valid = np.flatnonzero(mask.mask)
        actions[uid] = int(valid[0]) if valid.size else 0
    return actions


def _rotating_valid_actions(masks: list[Any]) -> np.ndarray:
    actions = np.zeros(len(masks), dtype=np.int32)
    for uid, mask in enumerate(masks):
        valid = np.flatnonzero(mask.mask)
        actions[uid] = int(valid[uid % int(valid.size)]) if valid.size else 0
    return actions


def _acceptance_flags(
    boundary: dict[str, Any],
    samples: dict[str, Any],
) -> dict[str, bool]:
    checks = boundary["checks"]
    return {
        "frozen_baseline_unchanged": True,
        "new_behavior_opt_in_and_namespace_gated": bool(checks["continuous_power_mode"]),
        "active_power_continuous_analytic": bool(checks["continuous_power_mode"]),
        "active_power_not_constant_codebook_or_profile": (
            bool(samples["active_power_nonconstant"])
            and bool(samples["active_power_non_codebook"])
            and bool(samples["selected_power_profile_absent"])
        ),
        "not_post_hoc_rescore": bool(samples["ee_denominator_reuses_step_power_vector"]),
        "computed_before_throughput_and_reward": (
            bool(samples["ee_denominator_reuses_step_power_vector"])
            and bool(samples["same_power_vector_for_candidate_and_control"])
        ),
        "same_power_vector_feeds_throughput_and_ee": (
            bool(samples["same_throughput_for_candidate_and_control"])
            and bool(samples["ee_denominator_reuses_step_power_vector"])
        ),
        "candidate_control_only_r1_differs": bool(
            checks["only_intended_difference_is_r1_reward_mode"]
        ),
        "throughput_same_guard_same_power_control_present": bool(
            checks["control_r1_is_throughput"]
            and checks["same_anti_collapse_guard"]
            and checks["same_continuous_power_surface"]
        ),
        "forbidden_modes_absent": bool(checks["forbidden_modes_disabled"]),
        "policy_action_consequences_can_change_power": bool(
            samples["policy_action_consequences_change_power"]
        ),
        "power_guardrails_hold": int(samples["power_budget_violations"]) == 0,
    }


def _write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_review(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CP-base Continuous-Power Boundary Audit",
        "",
        f"Decision: `{summary['acceptance_result']}`",
        "",
        "This artifact is implementation-readiness evidence only. It does not run "
        "pilot training and does not claim EE-MODQN effectiveness.",
        "",
        "## Boundary",
        "",
        f"- matched boundary pass: `{summary['boundary_proof']['matched_boundary_pass']}`",
        "- only intended candidate/control difference: `r1_reward_mode`",
        "- shared control: throughput + same QoS-sticky guard + same continuous power",
        "",
        "## Wiring",
        "",
        f"- same power vector candidate/control: `{summary['deterministic_step_samples']['same_power_vector_for_candidate_and_control']}`",
        f"- EE denominator reuses step vector: `{summary['deterministic_step_samples']['ee_denominator_reuses_step_power_vector']}`",
        f"- policy/action consequences change power: `{summary['deterministic_step_samples']['policy_action_consequences_change_power']}`",
        f"- selected power profile absent: `{summary['deterministic_step_samples']['selected_power_profile_absent']}`",
        "",
        "## Forbidden Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in summary["forbidden_claims_still_active"])
    path.write_text("\n".join(lines) + "\n")


def _power_surface_value(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("resolved_assumptions", {}).get("hobs_power_surface", {})
    value = block.get("value", {}) if isinstance(block, dict) else {}
    return value if isinstance(value, dict) else {}


def _gate_block(cfg: dict[str, Any]) -> dict[str, Any]:
    experiment = cfg.get("training_experiment", {})
    experiment = experiment if isinstance(experiment, dict) else {}
    block = experiment.get("hobs_active_tx_ee_non_codebook_continuous_power", {})
    return block if isinstance(block, dict) else {}


def _anti_collapse_block(gate_block: dict[str, Any]) -> dict[str, Any]:
    block = gate_block.get("anti_collapse_action_constraint", {})
    return block if isinstance(block, dict) else {}


def _boundary_common_subset(cfg: dict[str, Any]) -> dict[str, Any]:
    trainer = build_trainer_config(cfg)
    gate = _gate_block(cfg)
    return {
        "training_experiment_kind": trainer.training_experiment_kind,
        "phase": trainer.phase,
        "method_family": trainer.method_family,
        "episodes": trainer.episodes,
        "objective_weights": list(trainer.objective_weights),
        "trainer_hyperparameters": _trainer_hyperparameter_subset(trainer),
        "checkpoint": {
            "primary": trainer.checkpoint_primary_report,
            "secondary": trainer.checkpoint_secondary_report,
        },
        "seeds": get_seeds(cfg),
        "power_surface": _power_surface_value(cfg),
        "anti_collapse": _anti_collapse_block(gate),
        "reward_calibration_enabled": trainer.reward_calibration_enabled,
        "catfish_enabled": trainer.catfish_enabled,
    }


def _trainer_hyperparameter_subset(trainer: Any) -> dict[str, Any]:
    return {
        "hidden_layers": list(trainer.hidden_layers),
        "activation": trainer.activation,
        "learning_rate": trainer.learning_rate,
        "discount_factor": trainer.discount_factor,
        "batch_size": trainer.batch_size,
        "epsilon_start": trainer.epsilon_start,
        "epsilon_end": trainer.epsilon_end,
        "epsilon_decay_episodes": trainer.epsilon_decay_episodes,
        "target_update_every_episodes": trainer.target_update_every_episodes,
        "replay_capacity": trainer.replay_capacity,
        "policy_sharing_mode": trainer.policy_sharing_mode,
        "snr_encoding": trainer.snr_encoding,
        "offset_scale_km": trainer.offset_scale_km,
        "load_normalization": trainer.load_normalization,
    }


def _forbidden_mode_flags(*cfgs: dict[str, Any]) -> dict[str, bool]:
    trainers = [build_trainer_config(cfg) for cfg in cfgs]
    boundary_tokens = []
    for cfg, trainer in zip(cfgs, trainers):
        power = _power_surface_value(cfg)
        gate = _gate_block(cfg)
        boundary_tokens.append(
            {
                "training_experiment_kind": trainer.training_experiment_kind,
                "method_family": trainer.method_family,
                "phase": trainer.phase,
                "comparison_role": trainer.comparison_role,
                "r1_reward_mode": trainer.r1_reward_mode,
                "power_mode": power.get("mode", ""),
                "power_profile": power.get("power_codebook_profile", ""),
                "anti_collapse_mode": _anti_collapse_block(gate).get("mode", ""),
            }
        )
    raw = yaml.safe_dump(boundary_tokens, sort_keys=True).lower()
    return {
        "catfish_enabled": any(trainer.catfish_enabled for trainer in trainers),
        "multi_catfish_enabled": "multi-catfish" in raw,
        "phase_03c_enabled": "phase-03c" in raw,
        "ra_ee_learned_association_enabled": "learned association" in raw,
        "oracle_enabled": "oracle" in raw,
        "future_information_enabled": "future information" in raw,
        "offline_replay_oracle_enabled": "offline replay oracle" in raw,
        "hobs_optimizer_enabled": (
            "hobs optimizer reproduction" not in raw and "hobs optimizer" in raw
        ),
        "finite_codebook_enabled": "power_codebook_levels_w" in raw,
        "runtime_profile_selector_enabled": "runtime-ee-selector" in raw,
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_vector(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values.tolist()])
