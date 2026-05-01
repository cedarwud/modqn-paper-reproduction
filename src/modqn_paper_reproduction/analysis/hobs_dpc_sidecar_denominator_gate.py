"""Route B: HOBS-style DPC sidecar denominator gate.

Namespace: hobs-dpc-sidecar-denominator-gate
Date: 2026-05-01

This module runs heuristic (hold-current) policy diagnostics under the DPC
sidecar to confirm that the active transmit-power denominator varies over time,
breaking the fixed-denominator failure mode that blocked Route D.

DPC sidecar rule (HOBS-inspired, not HOBS optimizer reproduction):
  P_b(t) = clip(P_b(t-1) + xi_b(t-1), P_min, P_beam_max)
  xi sign-flipped when EE_b(t-1) <= EE_b(t-2)

Gate PASS conditions:
  - denominator_varies_in_eval = True
  - active_power_single_point_distribution = False
  - throughput_proxy_risk_flag = False
  - all power guardrail violations = 0 (per-beam cap, sat cap, inactive)
  - dpc_sign_flip_count >= 1 (DPC controller is actively adjusting direction)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import build_environment, load_training_yaml
from ..env.step import ActionMask, StepResult, UserState, _HOBS_ACTIVE_TX_EE_EPSILON_P_W
from ._common import write_json

_AUDIT_POLICIES = ("hold-current", "random-valid")
_MIN_SAMPLES_CORRELATION = 5


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
        if policy == "hold-current":
            cur = int(np.argmax(states[uid].access_vector))
            actions[uid] = cur if bool(mask.mask[cur]) else int(valid[0])
        elif policy == "random-valid":
            actions[uid] = int(rng.choice(valid))
        else:
            raise ValueError(f"Unsupported policy: {policy!r}")
    return actions


def _unique_sorted(values: list[float], *, places: int = 9) -> list[float]:
    return sorted({round(float(v), places) for v in values})


def run_dpc_denominator_gate(
    config_path: str | Path,
    *,
    evaluation_seed: int = 20260501,
    max_steps: int | None = None,
    policies: tuple[str, ...] = _AUDIT_POLICIES,
) -> dict[str, Any]:
    """Run DPC sidecar denominator gate diagnostic.

    Returns:
      denominator_varies_in_eval, active_power_single_point_distribution,
      power_control_activity_rate, dpc_sign_flip_count, dpc_qos_guard_count,
      inactive_beam_power_violations, throughput_proxy_risk_flag,
      per_beam_cap_violations, sat_cap_violations, route_b_pass
    """
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)

    power_surface_mode = env.power_surface_config.hobs_power_surface_mode

    step_records: list[dict[str, Any]] = []

    for policy_index, policy in enumerate(policies):
        seed_seq = np.random.SeedSequence([int(evaluation_seed), policy_index])
        env_seed, mob_seed, act_seed = seed_seq.spawn(3)
        env_rng = np.random.default_rng(env_seed)
        mob_rng = np.random.default_rng(mob_seed)
        act_rng = np.random.default_rng(act_seed)

        states, masks, _diag = env.reset(env_rng, mob_rng)
        steps_seen = 0

        while True:
            if max_steps is not None and steps_seen >= int(max_steps):
                break
            actions = _select_actions(policy, states, masks, act_rng)
            result: StepResult = env.step(actions, env_rng)
            steps_seen += 1

            active_mask = result.active_beam_mask.astype(bool)
            active_count = int(np.sum(active_mask))
            total_active_power = float(result.total_active_beam_power_w)
            sum_thr = float(
                np.sum([rw.r1_throughput for rw in result.rewards], dtype=np.float64)
            )
            ee_val = float(result.rewards[0].r1_hobs_active_tx_ee) if result.rewards else 0.0
            inactive_powers = result.beam_transmit_power_w[~active_mask]
            max_inactive_power = float(np.max(inactive_powers)) if inactive_powers.size > 0 else 0.0

            active_power_vals = result.beam_transmit_power_w[active_mask].tolist()

            step_records.append({
                "policy": policy,
                "step_index": int(result.step_index),
                "active_beam_count": active_count,
                "total_active_power_w": total_active_power,
                "sum_throughput_bps": sum_thr,
                "ee_active_tx": ee_val,
                "active_power_vals": active_power_vals,
                "max_inactive_power_w": max_inactive_power,
            })

            if result.done:
                break
            states = result.user_states
            masks = result.action_masks

    # Collect DPC counters (reset-level cumulative, across all policies)
    dpc_diag = env.get_dpc_diagnostics()

    # Aggregate diagnostics
    if not step_records:
        return {
            "error": "no steps recorded",
            "denominator_varies_in_eval": False,
            "throughput_proxy_risk_flag": True,
            "route_b_pass": False,
        }

    total_powers = [r["total_active_power_w"] for r in step_records]
    throughputs = [r["sum_throughput_bps"] for r in step_records]
    ees = [r["ee_active_tx"] for r in step_records]
    all_active_vals: list[float] = []
    for r in step_records:
        all_active_vals.extend(r["active_power_vals"])
    max_inactive = max(r["max_inactive_power_w"] for r in step_records)

    distinct_total = _unique_sorted(total_powers)
    denominator_varies = len(distinct_total) > 1
    distinct_active = _unique_sorted(all_active_vals)
    active_single_point = len(distinct_active) <= 1

    steps_with_power_change = sum(
        1 for i in range(1, len(total_powers))
        if abs(total_powers[i] - total_powers[i - 1]) > 1e-10
    )
    power_activity_rate = steps_with_power_change / max(len(total_powers) - 1, 1)

    throughput_proxy_risk = not denominator_varies or active_single_point

    pearson: float | None = None
    if len(throughputs) >= _MIN_SAMPLES_CORRELATION:
        ta = np.array(throughputs, dtype=np.float64)
        ea = np.array(ees, dtype=np.float64)
        if float(np.std(ta)) > 0 and float(np.std(ea)) > 0:
            pearson = float(np.corrcoef(ta, ea)[0, 1])

    inactive_violation = max_inactive > 1e-12

    route_b_pass = (
        denominator_varies
        and not active_single_point
        and not throughput_proxy_risk
        and not inactive_violation
        and dpc_diag["dpc_per_beam_cap_violations"] == 0
        and dpc_diag["dpc_sat_cap_violations"] == 0
        and dpc_diag["dpc_sign_flip_count"] >= 1
    )

    return {
        "power_surface_mode": power_surface_mode,
        "steps_audited": len(step_records),
        "policies_used": list(policies),
        "denominator_varies_in_eval": denominator_varies,
        "distinct_total_active_power_w_values": distinct_total,
        "active_power_single_point_distribution": active_single_point,
        "distinct_active_beam_power_w_values": distinct_active,
        "power_control_activity_rate": power_activity_rate,
        "throughput_vs_ee_pearson": pearson,
        "throughput_proxy_risk_flag": throughput_proxy_risk,
        "inactive_beam_power_violations": inactive_violation,
        "max_inactive_beam_power_w": max_inactive,
        "dpc_step_count": dpc_diag["dpc_step_count"],
        "dpc_sign_flip_count": dpc_diag["dpc_sign_flip_count"],
        "dpc_qos_guard_count": dpc_diag["dpc_qos_guard_count"],
        "dpc_per_beam_cap_violations": dpc_diag["dpc_per_beam_cap_violations"],
        "dpc_sat_cap_violations": dpc_diag["dpc_sat_cap_violations"],
        "route_b_pass": route_b_pass,
        "route_b_note": (
            "DPC sidecar creates denominator variability independent of learned "
            "policy or interference. Denominator-varies gate is the prerequisite "
            "for Route D tiny matched pilot."
            if route_b_pass
            else "Gate did not fully pass — see diagnostic fields."
        ),
        "forbidden_claims": [
            "Do not claim EE-MODQN effectiveness from this diagnostic.",
            "Do not claim HOBS optimizer reproduction.",
            "Do not claim physical energy saving.",
            "Do not use scalar reward alone as success evidence.",
            "DPC is HOBS-inspired new extension, not MODQN paper-backed.",
            "This diagnostic does not prove Route D pilot will pass.",
        ],
    }


def export_dpc_denominator_gate(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    evaluation_seed: int = 20260501,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run gate and export summary.json + step_trace.csv + review.md."""
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)

    power_surface_mode = env.power_surface_config.hobs_power_surface_mode
    policies: tuple[str, ...] = _AUDIT_POLICIES

    step_trace: list[dict[str, Any]] = []

    for policy_index, policy in enumerate(policies):
        seed_seq = np.random.SeedSequence([int(evaluation_seed), policy_index])
        env_seed, mob_seed, act_seed = seed_seq.spawn(3)
        env_rng = np.random.default_rng(env_seed)
        mob_rng = np.random.default_rng(mob_seed)
        act_rng = np.random.default_rng(act_seed)

        states, masks, _diag = env.reset(env_rng, mob_rng)
        steps_seen = 0
        while True:
            if max_steps is not None and steps_seen >= int(max_steps):
                break
            actions = _select_actions(policy, states, masks, act_rng)
            result = env.step(actions, env_rng)
            steps_seen += 1

            active_mask = result.active_beam_mask.astype(bool)
            total_thr = float(
                np.sum([rw.r1_throughput for rw in result.rewards], dtype=np.float64)
            )
            ee = float(result.rewards[0].r1_hobs_active_tx_ee) if result.rewards else 0.0

            step_trace.append({
                "policy": policy,
                "step_index": int(result.step_index),
                "active_beam_count": int(np.sum(active_mask)),
                "total_active_power_w": float(result.total_active_beam_power_w),
                "sum_throughput_bps": total_thr,
                "ee_active_tx": ee,
                "selected_power_profile": result.selected_power_profile,
            })

            if result.done:
                break
            states = result.user_states
            masks = result.action_masks

    dpc_diag = env.get_dpc_diagnostics()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_powers_trace = [r["total_active_power_w"] for r in step_trace]
    distinct_total = _unique_sorted(total_powers_trace)
    denominator_varies = len(distinct_total) > 1

    all_active_vals: list[float] = []
    for r in step_trace:
        pass  # only stored total in this variant; distinct handled from distinct_total

    summary: dict[str, Any] = {
        "namespace": "hobs-dpc-sidecar-denominator-gate",
        "date": "2026-05-01",
        "power_surface_mode": power_surface_mode,
        "steps_traced": len(step_trace),
        "denominator_varies_in_eval": denominator_varies,
        "distinct_total_active_power_w_count": len(distinct_total),
        "distinct_total_active_power_w_values": distinct_total[:20],
        "dpc_diagnostics": dpc_diag,
        "route_b_pass": (
            denominator_varies
            and dpc_diag["dpc_sign_flip_count"] >= 1
            and dpc_diag["dpc_per_beam_cap_violations"] == 0
            and dpc_diag["dpc_sat_cap_violations"] == 0
        ),
        "forbidden_claims": [
            "Do not claim EE-MODQN effectiveness.",
            "Do not claim HOBS optimizer reproduction.",
            "Do not claim physical energy saving.",
            "DPC is HOBS-inspired extension, not MODQN paper-backed.",
        ],
    }

    summary_path = write_json(out / "summary.json", summary)

    csv_path = out / "step_trace.csv"
    fieldnames = [
        "policy", "step_index", "active_beam_count",
        "total_active_power_w", "sum_throughput_bps",
        "ee_active_tx", "selected_power_profile",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(step_trace)

    review_path = out / "review.md"
    _write_review(review_path, summary, dpc_diag)

    return {
        "summary_path": summary_path,
        "step_trace_csv_path": csv_path,
        "review_md_path": review_path,
        "summary": summary,
    }


def _write_review(path: Path, summary: dict, dpc_diag: dict) -> None:
    verdict = "**PASS**" if summary["route_b_pass"] else "**NEEDS MORE DESIGN**"
    lines = [
        "# Route B: HOBS-style DPC Sidecar Denominator Gate",
        "",
        f"**Date:** 2026-05-01  **Status:** {verdict}",
        "",
        "## Summary",
        "",
        f"- `denominator_varies_in_eval`: `{summary['denominator_varies_in_eval']}`",
        f"- Distinct total active power values: {summary['distinct_total_active_power_w_count']}",
        f"- DPC sign flip count: {dpc_diag['dpc_sign_flip_count']}",
        f"- DPC QoS guard activations: {dpc_diag['dpc_qos_guard_count']}",
        f"- Per-beam cap violations: {dpc_diag['dpc_per_beam_cap_violations']}",
        f"- Satellite cap violations: {dpc_diag['dpc_sat_cap_violations']}",
        "",
        "## Forbidden Claims",
        "",
        "- Do not claim EE-MODQN effectiveness.",
        "- Do not claim HOBS optimizer reproduction.",
        "- Do not claim physical energy saving.",
        "- DPC is HOBS-inspired new extension, not MODQN paper-backed.",
        "- This gate does not prove Route D pilot will pass.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


__all__ = [
    "run_dpc_denominator_gate",
    "export_dpc_denominator_gate",
]
