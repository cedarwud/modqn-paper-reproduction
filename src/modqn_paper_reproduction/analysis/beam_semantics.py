"""Beam-semantic audit helpers for the Phase 01E reopen slice."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import RunArtifactPaths, read_run_metadata
from ..artifacts.compat import (
    _select_timeline_seed,
    resolve_training_config_snapshot,
    select_replay_checkpoint,
)
from ..config_loader import build_environment, build_trainer_config
from ._common import write_json

_BEAM_TIE_FIELDS = [
    "evaluation_seed",
    "decision_step_index",
    "decision_time_s",
    "user_index",
    "satellite_index",
    "valid_beam_count",
    "beams_per_satellite",
    "all_beams_valid",
    "unique_valid_channel_count",
    "all_valid_channels_tied",
    "rss_action",
    "first_valid_action",
    "rss_is_first_valid",
    "rss_tie_width",
    "modqn_selected_action",
    "modqn_selected_action_in_block",
    "runner_up_action",
    "runner_up_action_in_block",
    "scalarized_margin_to_runner_up",
]

_DECISION_MARGIN_FIELDS = [
    "evaluation_seed",
    "decision_step_index",
    "decision_time_s",
    "user_index",
    "valid_action_count",
    "visible_satellite_count",
    "full_valid_satellite_block_count",
    "all_visible_blocks_full_valid",
    "channel_collapsed_visible_block_count",
    "any_channel_collapse",
    "all_visible_blocks_channel_collapsed",
    "rss_action",
    "first_valid_action",
    "rss_is_first_valid",
    "rss_tie_width",
    "modqn_selected_action",
    "modqn_equals_rss",
    "runner_up_action",
    "selected_satellite_index",
    "runner_up_satellite_index",
    "top1_top2_same_satellite",
    "selected_scalarized_q",
    "runner_up_scalarized_q",
    "scalarized_margin_to_runner_up",
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


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _classify_fraction(value: float) -> str:
    if value >= 0.75:
        return "pervasive"
    if value >= 0.25:
        return "common"
    if value > 0.0:
        return "rare"
    return "absent"


def _margin_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
            "lt_1_rate": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "lt_1_rate": float(np.mean(arr < 1.0)),
    }


def _rss_selection_metrics(
    channel_quality: np.ndarray,
    mask: np.ndarray,
) -> tuple[int | None, int | None, int, bool]:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return (None, None, 0, False)

    valid_channels = channel_quality[valid].astype(np.float64, copy=False)
    max_channel = float(np.max(valid_channels))
    tie_mask = np.isclose(valid_channels, max_channel, rtol=1e-6, atol=1e-12)
    tie_candidates = valid[tie_mask]
    rss_action = int(tie_candidates[0])
    first_valid_action = int(valid[0])
    return (
        rss_action,
        first_valid_action,
        int(tie_candidates.size),
        rss_action == first_valid_action,
    )


def _action_satellite_index(action: int | None, beams_per_satellite: int) -> int | None:
    if action is None:
        return None
    return int(action // beams_per_satellite)


def _action_in_block(action: int | None, start: int, stop: int) -> bool:
    if action is None:
        return False
    return start <= action < stop


def _build_review_lines(summary: dict[str, Any]) -> list[str]:
    surface = summary["effective_audit_surface"]
    decision_margins = summary["scalarized_margin_stats"]
    lines = [
        "# Beam Semantics Audit Review",
        "",
        "This audit replays a preserved checkpoint without retraining.",
        "",
        "## Surface",
        "",
        f"- checkpoint: `{summary['checkpoint_kind']}`",
        f"- evaluation seed: `{summary['evaluation_seed']}` "
        f"({summary['evaluation_seed_source']})",
        f"- bounded steps audited: `{surface['steps_audited']}`",
        f"- bounded users audited per step: `{surface['users_audited_per_step']}` "
        f"of `{surface['users_total']}`",
        "",
        "## Classification",
        "",
        f"- valid-mask collapse: `{summary['valid_mask_collapse_classification']}` "
        f"({summary['fraction_user_steps_all_visible_blocks_full_valid']:.1%} of "
        "user-steps exposed every visible satellite as a full beam block)",
        f"- channel-value collapse: `{summary['channel_value_collapse_classification']}` "
        f"({summary['fraction_visible_blocks_channel_collapsed']:.1%} of visible "
        "satellite blocks had one distinct channel value)",
        f"- comparator degeneration: "
        f"`{summary['comparator_degeneration_classification']}` "
        f"(`RSS_max` tied on {summary['rss_tie_rate']:.1%} of audited "
        f"user-steps and fell back to the first valid beam on "
        f"{summary['rss_first_valid_beam_rate']:.1%})",
        "",
        "## Decision Margins",
        "",
        f"- scalarized top-1 minus top-2 median: `{decision_margins['median']}`",
        f"- scalarized top-1 minus top-2 p90: `{decision_margins['p90']}`",
        f"- small-margin rate (`margin < 1.0`): "
        f"`{decision_margins['lt_1_rate']}`",
        f"- top-1 / top-2 same-satellite rate: "
        f"`{summary['top1_top2_same_satellite_rate']}`",
        "",
        "## Interpretation",
        "",
        "- This helper quantifies baseline semantics only; it does not change "
        "the frozen runtime contract.",
        "- If valid-mask, channel-value, and comparator collapse all remain "
        "high here, `01E2` is justified as the only allowed next slice.",
    ]
    return lines


def export_beam_semantics_audit(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    evaluation_seed: int | None = None,
    max_steps: int | None = None,
    max_users: int | None = None,
) -> dict[str, Any]:
    """Replay a preserved checkpoint and quantify beam-semantic collapse."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    if max_users is not None and max_users < 1:
        raise ValueError(f"max_users must be >= 1 when provided, got {max_users!r}")

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    metadata = read_run_metadata(RunArtifactPaths(in_dir).run_metadata_json)
    cfg = resolve_training_config_snapshot(metadata, artifact_dir=in_dir)
    trainer_cfg = build_trainer_config(cfg)
    seeds = metadata.seeds.to_dict()

    env = build_environment(copy.deepcopy(cfg))
    trainer = MODQNTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )

    checkpoint_path, checkpoint_kind = select_replay_checkpoint(
        metadata,
        artifact_dir=in_dir,
    )
    checkpoint_payload = trainer.load_checkpoint(
        checkpoint_path,
        load_optimizers=False,
    )
    if evaluation_seed is None:
        selected_seed, selected_seed_source = _select_timeline_seed(
            metadata,
            checkpoint_payload,
            cfg=cfg,
        )
    else:
        selected_seed = int(evaluation_seed)
        selected_seed_source = "cli.override"

    objective_weights = tuple(
        float(value)
        for value in checkpoint_payload.get(
            "trainer_config",
            {},
        ).get("objective_weights", trainer.config.objective_weights)
    )
    beams_per_satellite = int(trainer.env.beam_pattern.num_beams)
    num_satellites = int(trainer.env.orbit.num_satellites)
    users_total = int(trainer.num_users)
    users_to_audit = users_total if max_users is None else min(int(max_users), users_total)

    env_seed_seq, mobility_seed_seq = np.random.SeedSequence(selected_seed).spawn(2)
    env_rng = np.random.default_rng(env_seed_seq)
    mobility_rng = np.random.default_rng(mobility_seed_seq)
    states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)
    encoded = trainer.encode_states(states)

    decision_rows: list[dict[str, Any]] = []
    beam_rows: list[dict[str, Any]] = []
    steps_audited = 0

    while True:
        if max_steps is not None and steps_audited >= int(max_steps):
            break

        decision_step_index = int(trainer.env.step_index)
        decision_time_s = float(trainer.env.time_s)
        actions, diagnostics = trainer.select_actions_with_diagnostics(
            encoded,
            masks,
            objective_weights=objective_weights,
            top_k=2,
        )

        for uid in range(users_to_audit):
            mask = masks[uid].mask
            valid = np.flatnonzero(mask)
            visible_satellite_count = 0
            full_valid_satellite_block_count = 0
            channel_collapsed_visible_block_count = 0

            diag = diagnostics[uid]
            selected_action = int(actions[uid]) if valid.size > 0 else None
            selected_scalarized_q = (
                None
                if diag is None
                else float(diag["selectedScalarizedQ"])
            )
            runner_up_action = (
                None
                if diag is None or diag["runnerUpAction"] is None
                else int(diag["runnerUpAction"])
            )
            runner_up_scalarized_q = (
                None
                if diag is None or diag["runnerUpScalarizedQ"] is None
                else float(diag["runnerUpScalarizedQ"])
            )
            scalarized_margin = (
                None
                if diag is None or diag["scalarizedMarginToRunnerUp"] is None
                else float(diag["scalarizedMarginToRunnerUp"])
            )
            rss_action, first_valid_action, rss_tie_width, rss_is_first_valid = (
                _rss_selection_metrics(states[uid].channel_quality, mask)
            )

            for sat_index in range(num_satellites):
                start = sat_index * beams_per_satellite
                stop = start + beams_per_satellite
                block_mask = mask[start:stop]
                valid_beam_count = int(np.sum(block_mask))
                if valid_beam_count == 0:
                    continue

                visible_satellite_count += 1
                all_beams_valid = valid_beam_count == beams_per_satellite
                if all_beams_valid:
                    full_valid_satellite_block_count += 1

                valid_channels = states[uid].channel_quality[start:stop][block_mask]
                unique_valid_channel_count = int(np.unique(valid_channels).size)
                all_valid_channels_tied = (
                    valid_beam_count > 1 and unique_valid_channel_count == 1
                )
                if all_valid_channels_tied:
                    channel_collapsed_visible_block_count += 1

                beam_rows.append(
                    {
                        "evaluation_seed": int(selected_seed),
                        "decision_step_index": decision_step_index,
                        "decision_time_s": decision_time_s,
                        "user_index": int(uid),
                        "satellite_index": int(sat_index),
                        "valid_beam_count": valid_beam_count,
                        "beams_per_satellite": beams_per_satellite,
                        "all_beams_valid": bool(all_beams_valid),
                        "unique_valid_channel_count": unique_valid_channel_count,
                        "all_valid_channels_tied": bool(all_valid_channels_tied),
                        "rss_action": rss_action,
                        "first_valid_action": first_valid_action,
                        "rss_is_first_valid": bool(rss_is_first_valid),
                        "rss_tie_width": int(rss_tie_width),
                        "modqn_selected_action": selected_action,
                        "modqn_selected_action_in_block": bool(
                            _action_in_block(selected_action, start, stop)
                        ),
                        "runner_up_action": runner_up_action,
                        "runner_up_action_in_block": bool(
                            _action_in_block(runner_up_action, start, stop)
                        ),
                        "scalarized_margin_to_runner_up": scalarized_margin,
                    }
                )

            all_visible_blocks_full_valid = (
                visible_satellite_count > 0
                and full_valid_satellite_block_count == visible_satellite_count
            )
            any_channel_collapse = channel_collapsed_visible_block_count > 0
            all_visible_blocks_channel_collapsed = (
                visible_satellite_count > 0
                and channel_collapsed_visible_block_count == visible_satellite_count
            )
            selected_satellite_index = _action_satellite_index(
                selected_action,
                beams_per_satellite,
            )
            runner_up_satellite_index = _action_satellite_index(
                runner_up_action,
                beams_per_satellite,
            )

            decision_rows.append(
                {
                    "evaluation_seed": int(selected_seed),
                    "decision_step_index": decision_step_index,
                    "decision_time_s": decision_time_s,
                    "user_index": int(uid),
                    "valid_action_count": int(valid.size),
                    "visible_satellite_count": visible_satellite_count,
                    "full_valid_satellite_block_count": full_valid_satellite_block_count,
                    "all_visible_blocks_full_valid": bool(all_visible_blocks_full_valid),
                    "channel_collapsed_visible_block_count": (
                        channel_collapsed_visible_block_count
                    ),
                    "any_channel_collapse": bool(any_channel_collapse),
                    "all_visible_blocks_channel_collapsed": bool(
                        all_visible_blocks_channel_collapsed
                    ),
                    "rss_action": rss_action,
                    "first_valid_action": first_valid_action,
                    "rss_is_first_valid": bool(rss_is_first_valid),
                    "rss_tie_width": int(rss_tie_width),
                    "modqn_selected_action": selected_action,
                    "modqn_equals_rss": (
                        None
                        if rss_action is None or selected_action is None
                        else bool(selected_action == rss_action)
                    ),
                    "runner_up_action": runner_up_action,
                    "selected_satellite_index": selected_satellite_index,
                    "runner_up_satellite_index": runner_up_satellite_index,
                    "top1_top2_same_satellite": (
                        None
                        if runner_up_satellite_index is None
                        else bool(selected_satellite_index == runner_up_satellite_index)
                    ),
                    "selected_scalarized_q": selected_scalarized_q,
                    "runner_up_scalarized_q": runner_up_scalarized_q,
                    "scalarized_margin_to_runner_up": scalarized_margin,
                }
            )

        steps_audited += 1
        if max_steps is not None and steps_audited >= int(max_steps):
            break

        result = trainer.env.step(actions, env_rng)
        if result.done:
            break

        states = result.user_states
        masks = result.action_masks
        encoded = trainer.encode_states(states)

    decision_count = len(decision_rows)
    beam_count = len(beam_rows)
    margin_values = [
        float(row["scalarized_margin_to_runner_up"])
        for row in decision_rows
        if row["scalarized_margin_to_runner_up"] is not None
    ]
    same_satellite_rows = [
        bool(row["top1_top2_same_satellite"])
        for row in decision_rows
        if row["top1_top2_same_satellite"] is not None
    ]
    modqn_equals_rss_rows = [
        bool(row["modqn_equals_rss"])
        for row in decision_rows
        if row["modqn_equals_rss"] is not None
    ]

    fraction_user_steps_all_visible_blocks_full_valid = float(
        np.mean([row["all_visible_blocks_full_valid"] for row in decision_rows])
    ) if decision_rows else 0.0
    fraction_visible_blocks_channel_collapsed = float(
        np.mean([row["all_valid_channels_tied"] for row in beam_rows])
    ) if beam_rows else 0.0
    fraction_user_steps_any_channel_collapse = float(
        np.mean([row["any_channel_collapse"] for row in decision_rows])
    ) if decision_rows else 0.0
    fraction_user_steps_all_visible_blocks_channel_collapsed = float(
        np.mean([row["all_visible_blocks_channel_collapsed"] for row in decision_rows])
    ) if decision_rows else 0.0
    rss_tie_rate = float(
        np.mean([row["rss_tie_width"] > 1 for row in decision_rows])
    ) if decision_rows else 0.0
    rss_first_valid_beam_rate = float(
        np.mean([row["rss_is_first_valid"] for row in decision_rows])
    ) if decision_rows else 0.0
    comparator_degeneration_fraction = min(rss_tie_rate, rss_first_valid_beam_rate)

    summary = {
        "paper_id": metadata.paper_id,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_kind": checkpoint_kind,
        "policy_episode": int(checkpoint_payload.get("episode", -1)),
        "evaluation_seed": int(selected_seed),
        "evaluation_seed_source": selected_seed_source,
        "objective_weights": [float(value) for value in objective_weights],
        "audit_limits": {
            "max_steps": None if max_steps is None else int(max_steps),
            "max_users": None if max_users is None else int(max_users),
        },
        "effective_audit_surface": {
            "steps_audited": int(steps_audited),
            "users_audited_per_step": int(users_to_audit),
            "users_total": int(users_total),
            "num_satellites": int(num_satellites),
            "beams_per_satellite": int(beams_per_satellite),
        },
        "decision_rows_audited": int(decision_count),
        "beam_block_rows_audited": int(beam_count),
        "mean_valid_action_count": (
            float(np.mean([row["valid_action_count"] for row in decision_rows]))
            if decision_rows else 0.0
        ),
        "mean_visible_satellite_count": (
            float(np.mean([row["visible_satellite_count"] for row in decision_rows]))
            if decision_rows else 0.0
        ),
        "fraction_user_steps_all_visible_blocks_full_valid": (
            fraction_user_steps_all_visible_blocks_full_valid
        ),
        "fraction_visible_blocks_channel_collapsed": (
            fraction_visible_blocks_channel_collapsed
        ),
        "fraction_user_steps_any_channel_collapse": (
            fraction_user_steps_any_channel_collapse
        ),
        "fraction_user_steps_all_visible_blocks_channel_collapsed": (
            fraction_user_steps_all_visible_blocks_channel_collapsed
        ),
        "rss_tie_rate": rss_tie_rate,
        "rss_first_valid_beam_rate": rss_first_valid_beam_rate,
        "modqn_equals_rss_rate": (
            float(np.mean(modqn_equals_rss_rows))
            if modqn_equals_rss_rows else None
        ),
        "top1_top2_same_satellite_rate": (
            float(np.mean(same_satellite_rows))
            if same_satellite_rows else None
        ),
        "scalarized_margin_stats": _margin_stats(margin_values),
        "valid_mask_collapse_classification": _classify_fraction(
            fraction_user_steps_all_visible_blocks_full_valid
        ),
        "channel_value_collapse_classification": _classify_fraction(
            fraction_user_steps_all_visible_blocks_channel_collapsed
        ),
        "comparator_degeneration_classification": _classify_fraction(
            comparator_degeneration_fraction
        ),
        "comparator_degeneration_fraction": comparator_degeneration_fraction,
    }

    summary_path = write_json(out_dir / "beam_semantics_summary.json", summary)
    beam_csv_path = _write_csv(
        out_dir / "beam_tie_metrics.csv",
        beam_rows,
        fieldnames=_BEAM_TIE_FIELDS,
    )
    decision_csv_path = _write_csv(
        out_dir / "decision_margin_metrics.csv",
        decision_rows,
        fieldnames=_DECISION_MARGIN_FIELDS,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_build_review_lines(summary)) + "\n")

    return {
        "beam_semantics_summary": summary_path,
        "beam_tie_metrics_csv": beam_csv_path,
        "decision_margin_metrics_csv": decision_csv_path,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = ["export_beam_semantics_audit"]
