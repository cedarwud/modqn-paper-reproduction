"""Phase 05A objective-specific Catfish buffer diagnostics.

This module is analysis-only. It reuses the Phase 04-B single-Catfish trainer
surface to collect bounded transition reward vectors, then evaluates whether
objective-wise high-value buffers would select distinct samples. It does not
create objective-specialist learners or change MODQN reward semantics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..algorithms.catfish_modqn import CatfishMODQNTrainer
from ..config_loader import (
    ConfigValidationError,
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from ..runtime.catfish_replay import (
    component_distribution_summary,
    distribution_summary,
    quality_score,
)
from ..runtime.trainer_spec import (
    PHASE_04_B_SINGLE_CATFISH_KIND,
    R1_REWARD_MODE_THROUGHPUT,
    TrainerConfig,
)


PHASE_05A_ANALYSIS_KIND = "phase-05a-multi-buffer-validation"
OBJECTIVE_NAMES = ("r1", "r2", "r3")
SCALAR_BUFFER_NAME = "scalar_phase04_high_value"
DUPLICATION_JACCARD_WARNING_THRESHOLD = 0.80
MIN_DISTINCT_SHARE_FOR_INTERVENTION = 0.10
DEGENERATE_TOP_BUFFER_SHARE_THRESHOLD = 0.50


@dataclass(frozen=True)
class Phase05ASample:
    """Transition-level objective record used only for Phase 05A analysis."""

    sample_id: int
    r1: float
    r2: float
    r3: float
    scalar_quality: float
    scalar_phase04_admitted: bool


class Phase05ADiagnosticTrainer(CatfishMODQNTrainer):
    """Single-Catfish trainer variant that records raw objective vectors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.phase05a_samples: list[Phase05ASample] = []

    def _route_catfish_replay(self, **kwargs: Any) -> bool:
        reward_raw = np.asarray(kwargs["reward_raw"], dtype=np.float64)
        scalar_quality = quality_score(reward_raw, self.config.catfish_quality_weights)
        admitted = super()._route_catfish_replay(**kwargs)
        self.phase05a_samples.append(
            Phase05ASample(
                sample_id=len(self.phase05a_samples),
                r1=float(reward_raw[0]),
                r2=float(reward_raw[1]),
                r3=float(reward_raw[2]),
                scalar_quality=float(scalar_quality),
                scalar_phase04_admitted=bool(admitted),
            )
        )
        return admitted


def validate_phase05a_analysis_config(
    cfg: dict[str, Any],
    trainer_cfg: TrainerConfig,
) -> dict[str, Any]:
    """Validate the opt-in Phase 05A analysis boundary."""
    analysis_block = cfg.get("analysis_experiment")
    if not isinstance(analysis_block, dict):
        raise ConfigValidationError(
            "Phase 05A requires analysis_experiment metadata."
        )
    kind = str(analysis_block.get("kind", "")).strip()
    if kind != PHASE_05A_ANALYSIS_KIND:
        raise ConfigValidationError(
            f"analysis_experiment.kind must be {PHASE_05A_ANALYSIS_KIND!r}, "
            f"got {kind!r}."
        )

    phase_block = analysis_block.get("phase_05a_multi_buffer")
    if not isinstance(phase_block, dict):
        raise ConfigValidationError(
            "analysis_experiment.phase_05a_multi_buffer must be a mapping."
        )
    if not _is_enabled(phase_block.get("enabled", False)):
        raise ConfigValidationError("Phase 05A analysis must be explicitly enabled.")

    if _is_enabled(phase_block.get("ee_reward", False)):
        raise ConfigValidationError("Phase 05A must not enable EE reward/objective.")
    if _is_enabled(phase_block.get("full_multi_catfish_agents", False)):
        raise ConfigValidationError(
            "Phase 05A must not enable full multi-Catfish agents."
        )

    analysis_method = str(analysis_block.get("method_family", "")).strip()
    if analysis_method in {"EE-MODQN", "Multi-Catfish-MODQN", "Catfish-EE-MODQN"}:
        raise ConfigValidationError(
            f"Phase 05A analysis cannot use method_family={analysis_method!r}."
        )
    if trainer_cfg.training_experiment_kind != PHASE_04_B_SINGLE_CATFISH_KIND:
        raise ConfigValidationError(
            "Phase 05A reuses only the Phase 04-B single-Catfish trainer surface."
        )
    if trainer_cfg.method_family != "Catfish-MODQN" or not trainer_cfg.catfish_enabled:
        raise ConfigValidationError(
            "Phase 05A needs the single Catfish-MODQN scalar high-value replay "
            "surface for comparison."
        )
    if trainer_cfg.r1_reward_mode != R1_REWARD_MODE_THROUGHPUT:
        raise ConfigValidationError(
            "Phase 05A keeps fixed reward r1='throughput'."
        )
    if trainer_cfg.reward_calibration_enabled:
        raise ConfigValidationError("Phase 05A must keep reward calibration disabled.")

    fixed_reward = phase_block.get("fixed_reward", {})
    expected = {"r1": "throughput", "r2": "handover penalty", "r3": "load balance"}
    if fixed_reward != expected:
        raise ConfigValidationError(
            "Phase 05A fixed_reward must be exactly "
            f"{expected!r}, got {fixed_reward!r}."
        )

    quantile = float(phase_block.get("objective_top_quantile", 0.80))
    if not 0.0 < quantile < 1.0:
        raise ConfigValidationError(
            f"objective_top_quantile must be between 0 and 1, got {quantile}."
        )

    return phase_block


def analyze_phase05a_samples(
    samples: Iterable[Phase05ASample],
    *,
    objective_top_quantile: float = 0.80,
    duplication_jaccard_warning_threshold: float = (
        DUPLICATION_JACCARD_WARNING_THRESHOLD
    ),
) -> dict[str, Any]:
    """Build objective buffers and overlap diagnostics from reward samples."""
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Phase 05A analysis requires at least one sample.")
    if not 0.0 < objective_top_quantile < 1.0:
        raise ValueError(
            "objective_top_quantile must be between 0 and 1, "
            f"got {objective_top_quantile}."
        )

    sample_ids = np.asarray([sample.sample_id for sample in sample_list], dtype=np.int64)
    rewards = np.asarray(
        [[sample.r1, sample.r2, sample.r3] for sample in sample_list],
        dtype=np.float64,
    )
    scalar_quality = np.asarray(
        [sample.scalar_quality for sample in sample_list],
        dtype=np.float64,
    )
    scalar_admitted_mask = np.asarray(
        [sample.scalar_phase04_admitted for sample in sample_list],
        dtype=bool,
    )

    objective_sets: dict[str, set[int]] = {}
    buffer_summaries: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    for objective in OBJECTIVE_NAMES:
        scores = objective_scores(rewards, objective)
        threshold = float(np.quantile(scores, objective_top_quantile))
        mask = scores >= threshold
        ids = {int(sample_id) for sample_id in sample_ids[mask].tolist()}
        objective_sets[objective] = ids
        thresholds[objective] = _threshold_payload(
            objective=objective,
            score_threshold=threshold,
            objective_top_quantile=objective_top_quantile,
        )
        buffer_summaries[objective] = _buffer_payload(
            objective=objective,
            sample_ids=ids,
            rewards=rewards,
            all_sample_ids=sample_ids,
            objective_top_quantile=objective_top_quantile,
        )

    scalar_ids = {
        int(sample_id) for sample_id in sample_ids[scalar_admitted_mask].tolist()
    }
    if not scalar_ids:
        scalar_threshold = float(np.quantile(scalar_quality, objective_top_quantile))
        scalar_ids = {
            int(sample_id)
            for sample_id in sample_ids[scalar_quality >= scalar_threshold].tolist()
        }
        scalar_selection_mode = "fallback-global-scalar-quantile"
    else:
        scalar_threshold = None
        scalar_selection_mode = "phase04-rolling-scalar-high-value-admission"

    all_sets = {**objective_sets, SCALAR_BUFFER_NAME: scalar_ids}
    overlaps = {
        "objective_jaccard": _pairwise_jaccard(objective_sets),
        "against_scalar_phase04_high_value": {
            name: jaccard(ids, scalar_ids)
            for name, ids in objective_sets.items()
        },
    }

    for objective in OBJECTIVE_NAMES:
        ids = objective_sets[objective]
        buffer_summaries[objective]["distinct_sample_contribution"] = (
            _distinct_contribution(objective, ids, objective_sets, scalar_ids)
        )

    warnings = _warnings(
        objective_sets=objective_sets,
        scalar_ids=scalar_ids,
        buffer_summaries=buffer_summaries,
        duplication_threshold=duplication_jaccard_warning_threshold,
    )
    stop_conditions = _stop_conditions(
        rewards=rewards,
        objective_sets=objective_sets,
        buffer_summaries=buffer_summaries,
        warnings=warnings,
    )
    verdict = _distinctness_verdict(warnings, stop_conditions, buffer_summaries)

    return {
        "phase": "05A",
        "analysis_kind": PHASE_05A_ANALYSIS_KIND,
        "claim_boundary": {
            "multi_buffer_validation_only": True,
            "full_multi_catfish_agents_started": False,
            "ee_reward_or_objective_introduced": False,
            "effectiveness_claim_allowed": False,
            "scalar_reward_alone_used_as_evidence": False,
        },
        "fixed_reward": {
            "r1": "throughput",
            "r2": "handover penalty",
            "r3": "load balance",
        },
        "sample_count": int(len(sample_list)),
        "objective_top_quantile": float(objective_top_quantile),
        "objective_thresholds": thresholds,
        "buffers": buffer_summaries,
        "scalar_phase04_high_value_buffer": {
            "selection_mode": scalar_selection_mode,
            "threshold": scalar_threshold,
            "size": int(len(scalar_ids)),
            "distribution": _component_distribution_for_ids(
                scalar_ids,
                rewards,
                sample_ids,
            ),
        },
        "overlap": overlaps,
        "sample_admission_counts": {
            **{name: int(len(ids)) for name, ids in objective_sets.items()},
            SCALAR_BUFFER_NAME: int(len(scalar_ids)),
        },
        "warnings": warnings,
        "stop_conditions": stop_conditions,
        "distinctness_verdict": verdict,
        "phase05b_recommendation": (
            "ALLOW_PHASE_05B_PLANNING" if verdict["may_plan_phase05b"] else "BLOCK_PHASE_05B"
        ),
    }


def objective_scores(rewards: np.ndarray, objective: str) -> np.ndarray:
    """Return rank scores where larger is better for the requested objective."""
    if rewards.ndim != 2 or rewards.shape[1] != 3:
        raise ValueError(f"rewards must have shape (n, 3), got {rewards.shape}.")
    if objective == "r1":
        return rewards[:, 0]
    if objective == "r2":
        return -np.abs(rewards[:, 1])
    if objective == "r3":
        return -np.abs(rewards[:, 2])
    raise ValueError(f"unsupported objective {objective!r}.")


def jaccard(left: set[int], right: set[int]) -> float:
    """Return Jaccard overlap for two sample-id sets."""
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def run_phase05a_multi_buffer_validation(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    episodes: int | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Run the bounded Phase 05A diagnostic and write JSON artifacts."""
    cfg = load_training_yaml(config_path)
    trainer_cfg = build_trainer_config(cfg)
    phase_block = validate_phase05a_analysis_config(cfg, trainer_cfg)
    if episodes is not None:
        trainer_cfg = _trainer_with_episode_override(trainer_cfg, episodes)

    seeds = get_seeds(cfg)
    env = build_environment(cfg)
    trainer = Phase05ADiagnosticTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=seeds["train_seed"],
        env_seed=seeds["environment_seed"],
        mobility_seed=seeds["mobility_seed"],
    )
    logs = trainer.train(
        progress_every=progress_every,
        evaluation_seed_set=(),
        evaluation_every_episodes=trainer_cfg.target_update_every_episodes,
    )

    diagnostics = analyze_phase05a_samples(
        trainer.phase05a_samples,
        objective_top_quantile=float(phase_block.get("objective_top_quantile", 0.80)),
    )
    diagnostics["inputs"] = {
        "config_path": str(config_path),
        "episodes_requested": int(trainer_cfg.episodes),
        "seeds": seeds,
        "source_surface": "Phase 04-B single Catfish-MODQN bounded trainer",
        "reused_existing_phase04c_artifacts": False,
        "existing_phase04c_artifact_blocker": (
            "Phase 04C artifacts contain only aggregate training and Catfish "
            "diagnostics; transition-level sample ids and reward vectors needed "
            "for objective top-set Jaccard are not present."
        ),
    }
    diagnostics["training_log_summary"] = {
        "episodes_completed": int(len(logs)),
        "final": asdict(logs[-1]) if logs else None,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "phase05a_transition_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for sample in trainer.phase05a_samples:
            f.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
    diagnostics_path = out_dir / "phase05a_multi_buffer_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    training_log_path = out_dir / "training_log.json"
    training_log_path.write_text(
        json.dumps([asdict(log) for log in logs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata_path = out_dir / "phase05a_run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "analysis_kind": PHASE_05A_ANALYSIS_KIND,
                "config_path": str(config_path),
                "output_dir": str(out_dir),
                "claim_boundary": diagnostics["claim_boundary"],
                "fixed_reward": diagnostics["fixed_reward"],
                "trainer_config": asdict(trainer_cfg),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    diagnostics["artifact_paths"] = {
        "diagnostics_json": str(diagnostics_path),
        "transition_samples_jsonl": str(samples_path),
        "training_log_json": str(training_log_path),
        "metadata_json": str(metadata_path),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return diagnostics


def _trainer_with_episode_override(
    trainer_cfg: TrainerConfig,
    episodes: int,
) -> TrainerConfig:
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}.")
    kwargs = {field.name: getattr(trainer_cfg, field.name) for field in dc_fields(trainer_cfg)}
    kwargs["episodes"] = int(episodes)
    return TrainerConfig(**kwargs)


def _is_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"enabled", "true", "yes", "1"}


def _threshold_payload(
    *,
    objective: str,
    score_threshold: float,
    objective_top_quantile: float,
) -> dict[str, Any]:
    if objective == "r1":
        return {
            "criterion": "r1 high throughput; larger is better",
            "percentile": float(objective_top_quantile),
            "score_threshold": float(score_threshold),
            "objective_threshold": float(score_threshold),
        }
    return {
        "criterion": f"{objective} penalty closer to zero; smaller absolute value is better",
        "percentile": float(objective_top_quantile),
        "score_threshold": float(score_threshold),
        "max_abs_penalty_for_admission": float(-score_threshold),
    }


def _buffer_payload(
    *,
    objective: str,
    sample_ids: set[int],
    rewards: np.ndarray,
    all_sample_ids: np.ndarray,
    objective_top_quantile: float,
) -> dict[str, Any]:
    admission_share = _safe_ratio(len(sample_ids), len(all_sample_ids))
    return {
        "objective": objective,
        "size": int(len(sample_ids)),
        "admission_share": admission_share,
        "expected_top_share_without_tie_expansion": float(1.0 - objective_top_quantile),
        "selection_rule": (
            "objective-wise percentile/rank; scalar reward is not used for "
            "objective buffer admission"
        ),
        "distribution": _component_distribution_for_ids(
            sample_ids,
            rewards,
            all_sample_ids,
        ),
        "admission_count": int(len(sample_ids)),
    }


def _component_distribution_for_ids(
    ids: set[int],
    rewards: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, Any]:
    if not ids:
        return component_distribution_summary([])
    id_mask = np.isin(sample_ids, list(ids))
    vectors = [tuple(row.tolist()) for row in rewards[id_mask]]
    return component_distribution_summary(vectors)


def _pairwise_jaccard(sets: dict[str, set[int]]) -> dict[str, float]:
    names = tuple(sets)
    result: dict[str, float] = {}
    for idx, left_name in enumerate(names):
        for right_name in names[idx + 1 :]:
            result[f"{left_name}_vs_{right_name}"] = jaccard(
                sets[left_name],
                sets[right_name],
            )
    return result


def _distinct_contribution(
    objective: str,
    ids: set[int],
    objective_sets: dict[str, set[int]],
    scalar_ids: set[int],
) -> dict[str, Any]:
    other_objective_ids: set[int] = set()
    for name, other_ids in objective_sets.items():
        if name != objective:
            other_objective_ids |= other_ids
    distinct_vs_other_objectives = ids - other_objective_ids
    distinct_vs_scalar = ids - scalar_ids
    distinct_vs_all_others = ids - (other_objective_ids | scalar_ids)
    share = _safe_ratio(len(distinct_vs_all_others), len(ids))
    return {
        "distinct_vs_other_objectives_count": int(len(distinct_vs_other_objectives)),
        "distinct_vs_scalar_count": int(len(distinct_vs_scalar)),
        "distinct_vs_all_other_buffers_count": int(len(distinct_vs_all_others)),
        "distinct_vs_all_other_buffers_share": share,
        "would_contribute_distinct_intervention_samples": bool(
            share is not None and share >= MIN_DISTINCT_SHARE_FOR_INTERVENTION
        ),
    }


def _warnings(
    *,
    objective_sets: dict[str, set[int]],
    scalar_ids: set[int],
    buffer_summaries: dict[str, Any],
    duplication_threshold: float,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for objective, ids in objective_sets.items():
        scalar_overlap = jaccard(ids, scalar_ids)
        if scalar_overlap >= duplication_threshold:
            warnings.append(
                {
                    "code": "duplicates-scalar-high-value-buffer",
                    "buffer": objective,
                    "jaccard": scalar_overlap,
                    "threshold": duplication_threshold,
                }
            )
        if objective != "r1":
            r1_overlap = jaccard(ids, objective_sets["r1"])
            if r1_overlap >= duplication_threshold:
                warnings.append(
                    {
                        "code": "duplicates-high-throughput-buffer",
                        "buffer": objective,
                        "jaccard": r1_overlap,
                        "threshold": duplication_threshold,
                    }
                )
        contribution = buffer_summaries[objective]["distinct_sample_contribution"]
        admission_share = buffer_summaries[objective]["admission_share"]
        if (
            admission_share is not None
            and admission_share > DEGENERATE_TOP_BUFFER_SHARE_THRESHOLD
        ):
            warnings.append(
                {
                    "code": "degenerate-objective-tie-admission",
                    "buffer": objective,
                    "admission_share": admission_share,
                    "threshold": DEGENERATE_TOP_BUFFER_SHARE_THRESHOLD,
                    "detail": (
                        "Objective percentile threshold admitted more than half "
                        "of all samples, usually because the objective has too "
                        "few distinct values for a bounded top buffer."
                    ),
                }
            )
        if not contribution["would_contribute_distinct_intervention_samples"]:
            warnings.append(
                {
                    "code": "low-distinct-intervention-contribution",
                    "buffer": objective,
                    "distinct_share": contribution[
                        "distinct_vs_all_other_buffers_share"
                    ],
                    "threshold": MIN_DISTINCT_SHARE_FOR_INTERVENTION,
                }
            )
    return warnings


def _stop_conditions(
    *,
    rewards: np.ndarray,
    objective_sets: dict[str, set[int]],
    buffer_summaries: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stops: list[dict[str, Any]] = []
    all_r1_summary = distribution_summary(rewards[:, 0].tolist())
    r3_summary = buffer_summaries["r3"]["distribution"]["r1"]
    all_r1_p25 = all_r1_summary["p25"]
    r3_r1_mean = r3_summary["mean"]
    if (
        all_r1_p25 is not None
        and r3_r1_mean is not None
        and float(r3_r1_mean) < float(all_r1_p25)
    ):
        stops.append(
            {
                "code": "r3-buffer-low-throughput-load-balancing",
                "detail": (
                    "r3 buffer r1 mean is below the all-sample r1 p25; "
                    "the load-balance buffer may be capturing low-throughput "
                    "balance samples."
                ),
                "r3_buffer_r1_mean": float(r3_r1_mean),
                "all_sample_r1_p25": float(all_r1_p25),
            }
        )

    if any(warning["code"].startswith("duplicates-") for warning in warnings):
        stops.append(
            {
                "code": "diagnostics-cannot-prove-distinct-sample-types",
                "detail": "At least one objective buffer duplicates scalar or r1 top samples.",
            }
        )
    if any(
        warning["code"] == "degenerate-objective-tie-admission"
        for warning in warnings
    ):
        stops.append(
            {
                "code": "objective-percentile-degenerated",
                "detail": (
                    "At least one objective-wise top buffer admitted a majority "
                    "of samples, so it is not a bounded high-value subset."
                ),
            }
        )

    empty_buffers = [name for name, ids in objective_sets.items() if not ids]
    if empty_buffers:
        stops.append(
            {
                "code": "objective-buffer-empty",
                "buffers": empty_buffers,
            }
        )
    return stops


def _distinctness_verdict(
    warnings: list[dict[str, Any]],
    stop_conditions: list[dict[str, Any]],
    buffer_summaries: dict[str, Any],
) -> dict[str, Any]:
    contribution = {
        name: payload["distinct_sample_contribution"][
            "would_contribute_distinct_intervention_samples"
        ]
        for name, payload in buffer_summaries.items()
    }
    all_contribute = all(contribution.values())
    may_plan = bool(all_contribute and not stop_conditions)
    return {
        "meaningfully_distinct": may_plan,
        "may_plan_phase05b": may_plan,
        "rationale": (
            "All objective buffers contribute distinct samples and no stop "
            "condition was triggered."
            if may_plan
            else "Phase 05A cannot promote full multi-agent validation because "
            "one or more buffers failed distinctness or triggered a stop condition."
        ),
        "per_buffer_distinct_intervention_contribution": contribution,
        "warning_count": int(len(warnings)),
        "stop_condition_count": int(len(stop_conditions)),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)
