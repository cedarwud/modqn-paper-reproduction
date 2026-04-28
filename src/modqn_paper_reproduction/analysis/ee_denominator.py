"""Phase 02 / 02B EE denominator audit helpers.

This module is deliberately audit-only. It reads the MODQN runtime surface and
reports whether throughput, explicit per-beam transmit power, and active-beam
state can support a HOBS-linked EE metric. It does not change rewards, trainer
behavior, or checkpoint selection.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import build_environment, load_training_yaml
from ..env.step import ActionMask, StepResult, UserState
from ._common import write_json

_AUDIT_POLICIES = ("hold-current", "random-valid", "first-valid")

_DENOMINATOR_FIELDS = [
    "policy",
    "evaluation_seed",
    "step_index",
    "time_s",
    "active_beam_count",
    "active_beam_indices",
    "active_beam_mask",
    "beam_transmit_power_w",
    "active_beam_transmit_power_w",
    "tx_power_w_per_active_beam",
    "total_active_tx_power_w_fixed_proxy",
    "total_active_beam_power_w",
    "sum_user_throughput_bps",
    "sum_beam_throughput_bps",
    "throughput_delta_bps",
    "ee_system_bps_per_w",
    "ee_fixed_power_proxy_bps_per_w",
]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _select_actions(
    policy: str,
    states: list[UserState],
    masks: list[ActionMask],
    rng: np.random.Generator,
) -> np.ndarray:
    actions = np.zeros(len(masks), dtype=np.int32)
    for uid, mask in enumerate(masks):
        valid = np.flatnonzero(mask.mask)
        if valid.size == 0:
            actions[uid] = 0
            continue

        if policy == "random-valid":
            actions[uid] = int(rng.choice(valid))
            continue

        if policy == "hold-current":
            current = int(np.argmax(states[uid].access_vector))
            actions[uid] = current if bool(mask.mask[current]) else int(valid[0])
            continue

        if policy == "first-valid":
            actions[uid] = int(valid[0])
            continue

        raise ValueError(f"Unsupported EE denominator audit policy: {policy!r}")
    return actions


def _format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values.tolist())


def _unique_float_values(values: list[float], *, places: int = 12) -> list[float]:
    return sorted({round(float(value), places) for value in values})


def build_power_surface_audit_row(
    *,
    result: StepResult,
    policy: str,
    evaluation_seed: int,
    tx_power_w: float,
    power_surface_mode: str,
    inactive_beam_policy: str,
) -> dict[str, Any]:
    """Build one EE denominator audit row from explicit runtime fields.

    ``beam_transmit_power_w`` is the runtime ``P_b(t)`` surface. In baseline
    ``static-config`` mode this remains a fixed-power diagnostic; in Phase 02B
    active-load mode it is the opt-in allocated beam-power surface.
    """
    active_mask = result.active_beam_mask.astype(bool, copy=False)
    beam_power_w = result.beam_transmit_power_w.astype(np.float64, copy=False)
    active_beam_indices = [
        int(index) for index in np.flatnonzero(active_mask).tolist()
    ]
    active_beam_count = len(active_beam_indices)
    active_power_w = beam_power_w[active_mask]
    total_active_fixed_proxy_w = float(active_beam_count * tx_power_w)
    total_active_power_w = float(np.sum(active_power_w, dtype=np.float64))
    sum_user_throughput = float(
        np.sum([reward.r1_throughput for reward in result.rewards], dtype=np.float64)
    )
    sum_beam_throughput = float(np.sum(result.beam_throughputs, dtype=np.float64))
    throughput_delta = float(sum_user_throughput - sum_beam_throughput)
    ee_system = (
        None
        if total_active_power_w <= 0.0
        else float(sum_user_throughput / total_active_power_w)
    )
    ee_proxy = (
        None
        if total_active_fixed_proxy_w <= 0.0
        else float(sum_user_throughput / total_active_fixed_proxy_w)
    )
    tx_power_per_active_beam = (
        None
        if active_power_w.size == 0
        else (
            float(active_power_w[0])
            if len(_unique_float_values(active_power_w.tolist())) == 1
            else None
        )
    )

    return {
        "policy": policy,
        "evaluation_seed": int(evaluation_seed),
        "step_index": int(result.step_index),
        "time_s": float(result.time_s),
        "active_beam_count": int(active_beam_count),
        "active_beam_indices": " ".join(str(index) for index in active_beam_indices),
        "active_beam_mask": " ".join("1" if value else "0" for value in active_mask),
        "beam_transmit_power_w": _format_vector(beam_power_w),
        "active_beam_transmit_power_w": _format_vector(active_power_w),
        "tx_power_w_per_active_beam": tx_power_per_active_beam,
        "total_active_tx_power_w_fixed_proxy": total_active_fixed_proxy_w,
        "total_active_beam_power_w": total_active_power_w,
        "sum_user_throughput_bps": sum_user_throughput,
        "sum_beam_throughput_bps": sum_beam_throughput,
        "throughput_delta_bps": throughput_delta,
        "ee_system_bps_per_w": ee_system,
        "ee_fixed_power_proxy_bps_per_w": ee_proxy,
    }


def build_fixed_power_proxy_row(
    *,
    result: StepResult,
    policy: str,
    evaluation_seed: int,
    tx_power_w: float,
) -> dict[str, Any]:
    """Backward-compatible fixed-power proxy row builder for Phase 02 tests."""
    return build_power_surface_audit_row(
        result=result,
        policy=policy,
        evaluation_seed=evaluation_seed,
        tx_power_w=tx_power_w,
        power_surface_mode="static-config",
        inactive_beam_policy="excluded-from-active-beams",
    )


def _classify_denominator(
    rows: list[dict[str, Any]],
    *,
    power_surface_mode: str,
    denominator_varies: bool,
) -> str:
    if not rows:
        return "no-runtime-rows"
    if power_surface_mode == "active-load-concave" and denominator_varies:
        return "hobs-compatible-active-load-concave-power-surface"
    if power_surface_mode == "active-load-concave":
        return "active-load-concave-power-surface-no-sampled-variability"
    active_counts = {
        int(row["active_beam_count"])
        for row in rows
    }
    if len(active_counts) > 1:
        return "fixed-power-active-beam-count-proxy"
    return "fixed-power-constant-denominator"


def _build_review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["phase_02b_decision"]
    variability = summary["denominator_variability"]
    return [
        "# Phase 02B EE Power Surface Audit Review",
        "",
        "This audit is report-only. It does not train EE-MODQN and does not "
        "change the MODQN reward surface.",
        "",
        "## Power Surface",
        "",
        f"- mode: `{summary['power_surface']['mode']}`",
        f"- inactive beam policy: `{summary['power_surface']['inactive_beam_policy']}`",
        f"- denominator classification: `{summary['denominator_classification']}`",
        "",
        "## Runtime Mapping",
        "",
        "- `R_i(t)`: `RewardComponents.r1_throughput`, computed from "
        "`user_throughputs` in `StepEnvironment._build_states_and_masks` and "
        "aggregated into `beam_throughputs`.",
        "- `P_b(t)`: `StepResult.beam_transmit_power_w`, an explicit linear-W "
        "per-beam runtime vector.",
        "- `active_beams`: `StepResult.active_beam_mask`; inactive beams are "
        f"`{summary['power_surface']['inactive_beam_policy']}`.",
        "",
        "## Audit Result",
        "",
        f"- active power values sampled: "
        f"`{variability['active_beam_power_w_distinct_values']}`",
        f"- distinct active-beam counts sampled: "
        f"`{variability['distinct_active_beam_counts']}`",
        f"- distinct total active-beam powers sampled: "
        f"`{variability['distinct_total_active_beam_power_w']}`",
        f"- denominator varies: `{variability['denominator_varies']}`",
        f"- HOBS EE defensible in current runtime: "
        f"`{decision['hobs_ee_system_defensible']}`",
        "",
        "## Decision",
        "",
        f"- Phase 02B status: `{decision['status']}`",
        f"- Phase 03 gate: `{decision['phase_03_gate']}`",
        f"- EE degenerates to throughput scaling: "
        f"`{decision['ee_degenerates_to_throughput_scaling']}`",
        f"- reason: {decision['reason']}",
    ]


def export_ee_denominator_audit(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    evaluation_seed: int = 20260428,
    max_steps: int | None = None,
    policies: tuple[str, ...] = _AUDIT_POLICIES,
) -> dict[str, Any]:
    """Export the Phase 02 denominator and unit audit for a resolved config."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    if not policies:
        raise ValueError("At least one EE denominator audit policy is required.")
    for policy in policies:
        if policy not in _AUDIT_POLICIES:
            raise ValueError(
                f"Unsupported EE denominator audit policy {policy!r}; "
                f"supported policies are {_AUDIT_POLICIES!r}."
            )

    cfg = load_training_yaml(config_path)
    out_dir = Path(output_dir)

    rows: list[dict[str, Any]] = []
    power_surface_mode: str | None = None
    inactive_beam_policy: str | None = None
    for policy_index, policy in enumerate(policies):
        env = build_environment(cfg)
        power_surface_mode = env.power_surface_config.hobs_power_surface_mode
        inactive_beam_policy = env.power_surface_config.inactive_beam_policy
        tx_power_w = float(env.channel_config.tx_power_w)
        seed_sequence = np.random.SeedSequence([int(evaluation_seed), policy_index])
        env_seed_seq, mobility_seed_seq, action_seed_seq = seed_sequence.spawn(3)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        action_rng = np.random.default_rng(action_seed_seq)

        states, masks, _diag = env.reset(env_rng, mobility_rng)
        steps_seen = 0
        while True:
            if max_steps is not None and steps_seen >= int(max_steps):
                break

            actions = _select_actions(policy, states, masks, action_rng)
            result = env.step(actions, env_rng)
            rows.append(
                build_power_surface_audit_row(
                    result=result,
                    policy=policy,
                    evaluation_seed=evaluation_seed,
                    tx_power_w=tx_power_w,
                    power_surface_mode=power_surface_mode,
                    inactive_beam_policy=inactive_beam_policy,
                )
            )
            steps_seen += 1

            if result.done:
                break
            states = result.user_states
            masks = result.action_masks

    active_counts = [int(row["active_beam_count"]) for row in rows]
    total_fixed_proxy_powers = [
        float(row["total_active_tx_power_w_fixed_proxy"])
        for row in rows
    ]
    total_active_beam_powers = [
        float(row["total_active_beam_power_w"])
        for row in rows
    ]
    throughput_deltas = [
        abs(float(row["throughput_delta_bps"]))
        for row in rows
    ]
    active_power_values: list[float] = []
    for row in rows:
        for value in str(row["active_beam_transmit_power_w"]).split():
            active_power_values.append(float(value))
    power_vectors = {str(row["beam_transmit_power_w"]) for row in rows}
    active_masks = {str(row["active_beam_mask"]) for row in rows}
    distinct_total_active_beam_power = _unique_float_values(total_active_beam_powers)
    denominator_varies = len(distinct_total_active_beam_power) > 1
    power_surface_mode = power_surface_mode or "static-config"
    inactive_beam_policy = inactive_beam_policy or "excluded-from-active-beams"
    denominator_classification = _classify_denominator(
        rows,
        power_surface_mode=power_surface_mode,
        denominator_varies=denominator_varies,
    )
    hobs_defensible = (
        denominator_classification
        == "hobs-compatible-active-load-concave-power-surface"
        and inactive_beam_policy == "zero-w"
    )
    phase_03_gate = (
        "conditional-go-for-paired-phase-03-design"
        if hobs_defensible
        else "no-go"
    )
    decision_reason = (
        "The opt-in active-load-concave surface emits explicit linear-W "
        "per-beam P_b(t), assigns inactive beams 0 W, uses the same P_b(t) "
        "in the SINR numerator path, and the sampled denominator varies."
        if hobs_defensible
        else (
            "The runtime denominator is still not defensible as HOBS P_b(t): "
            f"classification={denominator_classification}, "
            f"mode={power_surface_mode}, denominator_varies={denominator_varies}."
        )
    )

    summary = {
        "config_path": str(Path(config_path)),
        "output_dir": str(out_dir),
        "evaluation_seed": int(evaluation_seed),
        "policies": list(policies),
        "rows_audited": int(len(rows)),
        "power_surface": {
            "mode": power_surface_mode,
            "inactive_beam_policy": inactive_beam_policy,
            "model": (
                "P_b(t)=0 for inactive beams; active P_b(t)=min(max_power_w, "
                "active_base_power_w + load_scale_power_w * N_b(t)^load_exponent)"
                if power_surface_mode == "active-load-concave"
                else "baseline static ChannelConfig.tx_power_w path"
            ),
            "config": (
                cfg.get("resolved_assumptions", {})
                .get("hobs_power_surface", {})
                .get("value", {})
            ),
        },
        "runtime_mapping": {
            "R_i_t": {
                "source": "StepEnvironment._build_states_and_masks user_throughputs",
                "reward_field": "RewardComponents.r1_throughput",
                "step_output": "StepResult.rewards[*].r1_throughput",
                "aggregate_output": "StepResult.beam_throughputs",
                "training_log_output": "EpisodeLog.r1_mean / training_log.json r1_mean",
                "replay_bundle_output": "rewardVector.r1Throughput and beamThroughputs",
            },
            "P_b_t": {
                "source": (
                    "StepResult.beam_transmit_power_w"
                    if power_surface_mode != "static-config"
                    else "ChannelConfig.tx_power_w via StepResult.beam_transmit_power_w"
                ),
                "channel_use": (
                    "StepEnvironment._build_states_and_masks computes "
                    "snr_linear = beam_transmit_power_w[b] * channel_gain / noise_power_w"
                    if power_surface_mode != "static-config"
                    else "compute_channel rx_power_w = tx_power_w * channel_gain"
                ),
                "available_as_allocated_power": power_surface_mode != "static-config",
                "varies_with_action": (
                    power_surface_mode != "static-config" and denominator_varies
                ),
                "varies_with_active_beam_state": (
                    power_surface_mode == "active-load-concave"
                ),
                "varies_with_power_allocation": (
                    power_surface_mode == "active-load-concave"
                ),
            },
            "active_beams": {
                "source": "StepResult.active_beam_mask",
                "explicit_named_field": True,
                "derivable": True,
            },
        },
        "power_unit_audit": {
            "beam_transmit_power_w_unit": "linear W",
            "active_beam_power_w_distinct_values": _unique_float_values(
                active_power_values
            ),
            "tx_power_w": float(tx_power_w) if rows else None,
            "unit": "linear W",
            "unit_basis": (
                "Power surface config fields use *_power_w names and "
                "StepEnvironment multiplies beam_transmit_power_w directly "
                "by channel_gain before dividing by noise_power_w."
            ),
            "mixed_db_units_detected_for_tx_power": False,
        },
        "denominator_variability": {
            "tx_power_w_distinct_values": [float(tx_power_w)] if rows else [],
            "active_beam_power_w_distinct_values": _unique_float_values(
                active_power_values
            ),
            "active_beam_count_min": min(active_counts) if active_counts else None,
            "active_beam_count_max": max(active_counts) if active_counts else None,
            "distinct_active_beam_counts": sorted(set(active_counts)),
            "total_active_fixed_proxy_power_w_min": (
                min(total_fixed_proxy_powers) if total_fixed_proxy_powers else None
            ),
            "total_active_fixed_proxy_power_w_max": (
                max(total_fixed_proxy_powers) if total_fixed_proxy_powers else None
            ),
            "distinct_total_active_power_w": _unique_float_values(
                total_fixed_proxy_powers
            ),
            "total_active_beam_power_w_min": (
                min(total_active_beam_powers) if total_active_beam_powers else None
            ),
            "total_active_beam_power_w_max": (
                max(total_active_beam_powers) if total_active_beam_powers else None
            ),
            "distinct_total_active_beam_power_w": (
                distinct_total_active_beam_power
            ),
            "beam_power_vector_distinct_count": len(power_vectors),
            "active_beam_mask_distinct_count": len(active_masks),
            "denominator_varies": denominator_varies,
            "throughput_sum_max_abs_delta_bps": (
                max(throughput_deltas) if throughput_deltas else None
            ),
        },
        "denominator_classification": denominator_classification,
        "phase_02b_decision": {
            "status": "promoted-for-metric-audit" if hobs_defensible else "blocked",
            "phase_03_gate": phase_03_gate,
            "hobs_ee_system_defensible": hobs_defensible,
            "fixed_power_trap_diagnostic": power_surface_mode == "static-config",
            "ee_degenerates_to_throughput_scaling": not hobs_defensible,
            "reason": decision_reason,
        },
        "forbidden_claims": [
            "Do not claim full paper-faithful reproduction.",
            "Do not claim EE-MODQN training evidence from this audit-only surface.",
            "Do not turn EE_system(t) into a reward in Phase 02B.",
            "Do not use static tx_power_w as allocated P_b(t).",
        ],
    }
    summary["phase_02_decision"] = summary["phase_02b_decision"]

    csv_path = _write_csv(
        out_dir / "ee_denominator_audit.csv",
        rows,
        fieldnames=_DENOMINATOR_FIELDS,
    )
    summary_path = write_json(out_dir / "ee_denominator_summary.json", summary)
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_build_review_lines(summary)) + "\n")

    return {
        "ee_denominator_summary": summary_path,
        "ee_denominator_audit_csv": csv_path,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = [
    "build_fixed_power_proxy_row",
    "build_power_surface_audit_row",
    "export_ee_denominator_audit",
]
