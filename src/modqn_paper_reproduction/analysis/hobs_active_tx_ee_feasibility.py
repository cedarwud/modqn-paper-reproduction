"""HOBS-style active-TX EE feasibility gate diagnostics.

Namespace: hobs-active-tx-ee-modqn-feasibility
Date: 2026-05-01

This module is audit-only. It runs fixed heuristic policies over the
environment and reports whether the HOBS-style active-TX EE formula:

    eta_EE,active-TX = sum_u(R_u^t) / (sum_b z_{s,b}^t p_{s,b}^t + eps)

can distinguish energy-aware behavior from throughput-proxy behavior.

The diagnostics check all failure conditions from Phase 03 / 03B / 03C-C:

    - denominator_varies_in_eval (False → throughput-proxy risk)
    - active beam count distribution (all 1 → collapse risk)
    - active transmit power distribution (single-point → fixed denominator)
    - power-control activity / distinct power values
    - throughput-vs-EE Pearson correlation (near 1 → EE ≈ throughput rescaling)
    - throughput-proxy risk flag (denominator effectively fixed)

No training is performed. No frozen baseline is mutated.

Slice B Design Gate Findings (2026-05-01)
=========================================

Mathematical equivalence under SNR-only channel + active-load-concave:

    hobs-active-tx-ee = bw * log2(1+SNR) / P_b_effective

    per-user-ee-credit = bw * log2(1+SNR) / P_b_beam

Under the current channel model (all beams of a satellite share the same SNR),
and under active-load-concave with uniform load per beam:

    hobs-active-tx-ee ≡ per-user-ee-credit  (numerically identical)

This is because SNR is per-satellite (not per-beam), so the active-TX EE
formula reduces to the same beam-power-normalized throughput value as the
old per-user EE credit.

Phase 03B already showed that per-user-ee-credit + active-load-concave
collapses under greedy evaluation (denominator_varies_in_eval=false,
one-active-beam on every step). Therefore:

    Route D (tiny pilot with current config) is PREDICTED TO BLOCK.
    Running D now would reproduce Phase 03 failure without new information.

Recommended next route (Slice B → Route A):

    Route A: SINR interference audit and minimum wiring.
    Add intra-satellite beam interference to break the SNR=per-satellite
    assumption. Under SINR, different active beam patterns produce different
    per-beam signal quality — the active-TX EE formula then becomes genuinely
    distinct from a throughput rescaling.

    Minimum SINR design:
    - Add I_intra,u^t = sum_{b'!=b, b' active on s} P_{s,b'} * h_{u<-s,b'}
    - Use same channel gain h for off-axis approximation (conservative)
    - New SINR: gamma = P_b * h / (I_intra + N0)
    - New R_u = bw/N_b * log2(1 + gamma) — now per-beam, not per-satellite
    - New system EE: numerator changes because beams have different SINR
    - Mathematical equivalence is BROKEN → genuine per-beam differentiation

    This is the prerequisite for a meaningful Route D pilot.

Gate stop conditions for any future D pilot (with or without SINR):
    - denominator_varies_in_eval=false under greedy eval → STOP
    - all_evaluated_steps_one_active_beam=true → STOP
    - throughput_vs_ee_pearson > 0.95 → STOP
    - active_power_single_point_distribution=true → STOP
    - EE improvement paired with throughput collapse → STOP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import build_environment, load_training_yaml
from ..env.step import ActionMask, StepResult, UserState
from ._common import write_json

_AUDIT_POLICIES = ("hold-current", "random-valid", "first-valid")
_MIN_CORRELATION_SAMPLES = 5


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
        raise ValueError(f"Unsupported policy: {policy!r}")
    return actions


def _unique_sorted(values: list[float], *, places: int = 9) -> list[float]:
    return sorted({round(float(v), places) for v in values})


def compute_hobs_active_tx_ee_diagnostics(
    config_path: str | Path,
    *,
    evaluation_seed: int = 20260501,
    max_steps: int | None = None,
    policies: tuple[str, ...] = _AUDIT_POLICIES,
) -> dict[str, Any]:
    """Run HOBS-style active-TX EE diagnostics over fixed heuristic policies.

    Returns a summary dict with:
      - denominator_varies_in_eval
      - active_beam_count_distribution
      - active_tx_power_w_distribution
      - power_surface_mode
      - throughput_proxy_risk_flag
      - throughput_vs_ee_pearson (if enough samples)
      - distinct_total_active_power_w_values
      - formula_verified  (bool — formula gives expected value on sampled steps)
    """
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")

    cfg = load_training_yaml(config_path)

    step_records: list[dict[str, Any]] = []
    power_surface_mode: str = "static-config"

    for policy_index, policy in enumerate(policies):
        env = build_environment(cfg)
        power_surface_mode = env.power_surface_config.hobs_power_surface_mode
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
            total_active_power_w = float(result.total_active_beam_power_w)
            sum_thr = float(np.sum(
                [rw.r1_throughput for rw in result.rewards], dtype=np.float64
            ))

            # Verify formula: r1_hobs_active_tx_ee should equal the raw ratio
            active_power_from_vector = float(
                np.sum(result.beam_transmit_power_w[active_mask], dtype=np.float64)
            )
            formula_ok = (
                active_count == 0  # degenerate: all inactive, ee = sum_thr / eps ≈ large
                or abs(
                    result.rewards[0].r1_hobs_active_tx_ee
                    - sum_thr / (active_power_from_vector + 1e-9)
                ) < 1e-6
            )

            # Distinct power values on active beams
            active_power_values = (
                result.beam_transmit_power_w[active_mask].tolist()
                if active_count > 0 else []
            )

            step_records.append({
                "policy": policy,
                "step_index": int(result.step_index),
                "active_beam_count": active_count,
                "total_active_power_w": total_active_power_w,
                "sum_throughput_bps": sum_thr,
                "ee_active_tx": float(result.rewards[0].r1_hobs_active_tx_ee),
                "active_power_values": active_power_values,
                "formula_ok": formula_ok,
            })

            if result.done:
                break
            states = result.user_states
            masks = result.action_masks

    if not step_records:
        return {
            "error": "no steps recorded",
            "denominator_varies_in_eval": False,
            "throughput_proxy_risk_flag": True,
        }

    active_counts = [r["active_beam_count"] for r in step_records]
    total_powers = [r["total_active_power_w"] for r in step_records]
    throughputs = [r["sum_throughput_bps"] for r in step_records]
    ees = [r["ee_active_tx"] for r in step_records]
    all_active_power_vals: list[float] = []
    for r in step_records:
        all_active_power_vals.extend(r["active_power_values"])

    distinct_total_powers = _unique_sorted(total_powers)
    denominator_varies = len(distinct_total_powers) > 1

    # Active beam count distribution
    count_freq: dict[int, int] = {}
    for c in active_counts:
        count_freq[c] = count_freq.get(c, 0) + 1
    active_beam_count_dist = {str(k): v for k, v in sorted(count_freq.items())}
    all_one_beam = all(c <= 1 for c in active_counts)

    # Active power distribution
    distinct_active_power_vals = _unique_sorted(all_active_power_vals)
    active_power_single_point = len(distinct_active_power_vals) <= 1

    # Throughput-proxy risk: denominator fixed, or single-point power
    throughput_proxy_risk = not denominator_varies or active_power_single_point

    # Throughput-vs-EE Pearson correlation (if enough samples)
    pearson_thr_ee: float | None = None
    if len(throughputs) >= _MIN_CORRELATION_SAMPLES:
        thr_arr = np.array(throughputs, dtype=np.float64)
        ee_arr = np.array(ees, dtype=np.float64)
        thr_std = float(np.std(thr_arr))
        ee_std = float(np.std(ee_arr))
        if thr_std > 0 and ee_std > 0:
            pearson_thr_ee = float(np.corrcoef(thr_arr, ee_arr)[0, 1])
        else:
            pearson_thr_ee = 1.0 if thr_std == 0 and ee_std == 0 else None

    formula_verified = all(r["formula_ok"] for r in step_records)

    return {
        "power_surface_mode": power_surface_mode,
        "steps_audited": len(step_records),
        "denominator_varies_in_eval": denominator_varies,
        "distinct_total_active_power_w_values": distinct_total_powers,
        "active_beam_count_distribution": active_beam_count_dist,
        "all_evaluated_steps_one_active_beam": all_one_beam,
        "distinct_active_beam_power_w_values": distinct_active_power_vals,
        "active_power_single_point_distribution": active_power_single_point,
        "throughput_proxy_risk_flag": throughput_proxy_risk,
        "throughput_vs_ee_pearson": pearson_thr_ee,
        "formula_verified": formula_verified,
        "forbidden_claims": [
            "Do not claim EE-MODQN effectiveness from this diagnostics run.",
            "Do not use scalar reward alone as a success flag.",
            "Do not claim physical spacecraft energy saving.",
            "Do not claim RA-EE-MODQN or learned association effectiveness.",
            "Throughput-proxy risk flag True means denominator is effectively fixed.",
        ],
    }


def export_hobs_active_tx_ee_feasibility_diagnostics(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    evaluation_seed: int = 20260501,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Export diagnostics to output_dir and return summary."""
    summary = compute_hobs_active_tx_ee_diagnostics(
        config_path,
        evaluation_seed=evaluation_seed,
        max_steps=max_steps,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(out / "hobs_active_tx_ee_feasibility_summary.json", summary)
    return {"summary_path": summary_path, "summary": summary}


def check_snr_per_satellite_assumption(config_path: str | Path) -> dict[str, Any]:
    """Check whether the channel model assigns per-satellite SNR to all beams.

    Under the SNR-only model, all beams of a satellite share the same
    channel SNR (snr_arr[sat*K : (sat+1)*K] = same value). This makes
    hobs-active-tx-ee numerically equivalent to per-user-ee-credit under
    active-load-concave, because R_u = bw/N_b * log2(1+SNR_sat) and
    the EE formula reduces to bw*log2(1+SNR_sat)/P_b_beam.

    Breaking this equivalence requires SINR with intra-beam interference
    (Route A), which makes different active beam patterns produce different
    per-beam signal quality.

    Returns a dict with:
      - all_beams_same_snr_per_sat: True if all beams share per-satellite SNR
      - has_per_beam_interference: True if SINR interference model is present
      - equivalence_holds: True if hobs-active-tx-ee ≡ per-user-ee-credit
      - route_a_needed: True if SINR interference must be added before D pilot
    """
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    K = env.beam_pattern.num_beams

    rng = np.random.default_rng(20260501)
    mob_rng = np.random.default_rng(99)
    states, masks, _diag = env.reset(rng, mob_rng)
    actions = np.zeros(env.config.num_users, dtype=np.int32)
    for uid, mask in enumerate(masks):
        valid = np.flatnonzero(mask.mask)
        if valid.size > 0:
            actions[uid] = int(valid[0])
    result = env.step(actions, rng)

    # Check whether channel gain (SNR / P_b) is constant across beams per satellite.
    # Under active-load-concave, SNR varies per beam (P_b differs), but the
    # channel gain h = SNR/P_b should be the same for all beams of a satellite
    # (computed from slant range, not per-beam interference).
    # Per-beam interference (Route A) would make h differ across beams.
    channel_qual = result.user_states[0].channel_quality  # shape (L*K,), SNR linear
    beam_pow = result.beam_transmit_power_w  # shape (L*K,)
    L = len(channel_qual) // K
    all_same_channel_gain_per_sat = True
    for s in range(L):
        block_snr = channel_qual[s * K: (s + 1) * K]
        block_pow = beam_pow[s * K: (s + 1) * K]
        active = (block_pow > 1e-12) & (block_snr > 0)
        if np.sum(active) > 1:
            ratios = block_snr[active] / block_pow[active]  # = h / noise per beam
            if float(np.max(ratios) - np.min(ratios)) > 1e-6:
                all_same_channel_gain_per_sat = False
                break

    # has_per_beam_interference: True if channel gain differs across beams (Route A done)
    has_per_beam_interference = not all_same_channel_gain_per_sat

    equivalence_holds = all_same_channel_gain_per_sat and (
        env.power_surface_config.hobs_power_surface_mode
        in ("active-load-concave", "phase-03c-b-power-codebook")
    )

    return {
        "all_beams_same_snr_per_satellite": all_same_channel_gain_per_sat,
        "has_per_beam_interference": has_per_beam_interference,
        "hobs_active_tx_ee_equiv_per_user_credit": equivalence_holds,
        "route_a_sinr_needed_before_d_pilot": equivalence_holds,
        "power_surface_mode": env.power_surface_config.hobs_power_surface_mode,
        "note": (
            "hobs-active-tx-ee ≡ per-user-ee-credit under SNR-only channel + "
            "active-load-concave. Phase 03B already showed collapse. "
            "Route A (SINR) must break this equivalence before Route D pilot."
            if equivalence_holds
            else "Per-beam differentiation detected — equivalence may not hold."
        ),
    }


def evaluate_greedy_policy_denominator_stats(
    trainer: Any,
    eval_seeds: tuple[int, ...],
    *,
    max_steps_per_episode: int | None = None,
) -> dict[str, Any]:
    """Collect per-step denominator stats under a trained greedy policy.

    Used by the Route D tiny pilot to check whether the learned policy
    avoids one-beam collapse. Call AFTER training with any config.

    Returns denominator_varies_in_eval, active_beam_count_distribution,
    all_one_beam, active_power_distribution, throughput_vs_ee_pearson.
    """
    step_records: list[dict[str, Any]] = []

    for seed in eval_seeds:
        env_seed_seq, mob_seed_seq = np.random.SeedSequence(seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mob_rng = np.random.default_rng(mob_seed_seq)

        states, masks, _diag = trainer.env.reset(env_rng, mob_rng)
        encoded = trainer._encode_states(states)

        steps_this = 0
        while True:
            if max_steps_per_episode and steps_this >= max_steps_per_episode:
                break
            actions = trainer.select_actions(encoded, masks, eps=0.0,
                                             objective_weights=trainer.config.objective_weights)
            result = trainer.env.step(actions, env_rng)
            steps_this += 1

            active_mask = result.active_beam_mask.astype(bool)
            total_thr = float(np.sum([rw.r1_throughput for rw in result.rewards], dtype=np.float64))
            ee = float(result.rewards[0].r1_hobs_active_tx_ee)

            step_records.append({
                "eval_seed": seed,
                "active_beam_count": int(np.sum(active_mask)),
                "total_active_power_w": float(result.total_active_beam_power_w),
                "sum_throughput_bps": total_thr,
                "ee_active_tx": ee,
                "active_power_vals": result.beam_transmit_power_w[active_mask].tolist(),
            })

            if result.done:
                break
            encoded = trainer._encode_states(result.user_states)
            masks = result.action_masks

    if not step_records:
        return {"error": "no steps", "denominator_varies_in_eval": False,
                "throughput_proxy_risk_flag": True}

    active_counts = [r["active_beam_count"] for r in step_records]
    total_powers = [r["total_active_power_w"] for r in step_records]
    throughputs = [r["sum_throughput_bps"] for r in step_records]
    ees = [r["ee_active_tx"] for r in step_records]
    all_power_vals: list[float] = []
    for r in step_records:
        all_power_vals.extend(r["active_power_vals"])

    distinct_powers = _unique_sorted(total_powers)
    denominator_varies = len(distinct_powers) > 1
    count_freq: dict[int, int] = {}
    for c in active_counts:
        count_freq[c] = count_freq.get(c, 0) + 1
    all_one = all(c <= 1 for c in active_counts)
    distinct_power_vals = _unique_sorted(all_power_vals)
    single_point = len(distinct_power_vals) <= 1

    pearson: float | None = None
    if len(throughputs) >= 5:
        ta = np.array(throughputs, dtype=np.float64)
        ea = np.array(ees, dtype=np.float64)
        if float(np.std(ta)) > 0 and float(np.std(ea)) > 0:
            pearson = float(np.corrcoef(ta, ea)[0, 1])
        else:
            pearson = 1.0

    return {
        "denominator_varies_in_eval": denominator_varies,
        "distinct_total_active_power_w_values": distinct_powers,
        "active_beam_count_distribution": {str(k): v for k, v in sorted(count_freq.items())},
        "all_evaluated_steps_one_active_beam": all_one,
        "distinct_active_power_w_values": distinct_power_vals,
        "active_power_single_point_distribution": single_point,
        "throughput_proxy_risk_flag": not denominator_varies or single_point,
        "throughput_vs_ee_pearson": pearson,
        "steps_evaluated": len(step_records),
        "gate_result": (
            "BLOCK" if all_one or single_point
            else ("PASS" if denominator_varies and (pearson is None or pearson < 0.95)
                  else "NEEDS-MORE-EVIDENCE")
        ),
    }


__all__ = [
    "check_snr_per_satellite_assumption",
    "compute_hobs_active_tx_ee_diagnostics",
    "evaluate_greedy_policy_denominator_stats",
    "export_hobs_active_tx_ee_feasibility_diagnostics",
]
