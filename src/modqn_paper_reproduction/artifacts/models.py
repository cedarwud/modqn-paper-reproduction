"""Typed training-artifact models for the Phase 04B seam.

These models preserve the current training-side artifact contract:
they make the boundary explicit without changing JSON keys, payload
semantics, or checkpoint envelope contents.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _copy_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(payload)


def _json_ready_value(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _json_ready_value(value)
            for key, value in payload.items()
        }
    if isinstance(payload, tuple):
        return [_json_ready_value(value) for value in payload]
    if isinstance(payload, list):
        return [_json_ready_value(value) for value in payload]
    return copy.deepcopy(payload)


def _json_ready_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return _json_ready_value(payload)


def _copy_optional_mapping(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return _copy_mapping(payload)


def _json_ready_optional_mapping(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    return _json_ready_mapping(payload)


def _copy_sequence(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(payload)


def _copy_optional_sequence(
    payload: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if payload is None:
        return None
    return _copy_sequence(payload)


@dataclass(frozen=True)
class SeedsBlock:
    train_seed: int
    environment_seed: int
    mobility_seed: int
    evaluation_seed_set: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_seed": int(self.train_seed),
            "environment_seed": int(self.environment_seed),
            "mobility_seed": int(self.mobility_seed),
            "evaluation_seed_set": [
                int(seed) for seed in self.evaluation_seed_set
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SeedsBlock:
        return cls(
            train_seed=int(payload["train_seed"]),
            environment_seed=int(payload["environment_seed"]),
            mobility_seed=int(payload["mobility_seed"]),
            evaluation_seed_set=tuple(
                int(seed) for seed in payload.get("evaluation_seed_set", [])
            ),
        )


@dataclass(frozen=True)
class CheckpointRuleV1:
    assumption_id: str
    primary_report: str
    secondary_report: str
    secondary_implemented: bool
    secondary_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "primary_report": self.primary_report,
            "secondary_report": self.secondary_report,
            "secondary_implemented": bool(self.secondary_implemented),
            "secondary_status": self.secondary_status,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointRuleV1:
        return cls(
            assumption_id=str(payload["assumption_id"]),
            primary_report=str(payload["primary_report"]),
            secondary_report=str(payload["secondary_report"]),
            secondary_implemented=bool(payload["secondary_implemented"]),
            secondary_status=str(payload["secondary_status"]),
        )


@dataclass(frozen=True)
class CheckpointFilesV1:
    primary_final: str
    secondary_best_eval: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_final": self.primary_final,
            "secondary_best_eval": self.secondary_best_eval,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointFilesV1:
        secondary = payload.get("secondary_best_eval")
        return cls(
            primary_final=str(payload["primary_final"]),
            secondary_best_eval=None if secondary is None else str(secondary),
        )


@dataclass(frozen=True)
class RewardCalibrationV1:
    enabled: bool
    mode: str
    source: str
    scales: tuple[float, float, float]
    training_experiment_kind: str
    training_experiment_id: str
    evaluation_metrics: str
    checkpoint_selection_metric: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "source": self.source,
            "scales": [float(scale) for scale in self.scales],
            "training_experiment_kind": self.training_experiment_kind,
            "training_experiment_id": self.training_experiment_id,
            "evaluation_metrics": self.evaluation_metrics,
            "checkpoint_selection_metric": self.checkpoint_selection_metric,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RewardCalibrationV1:
        return cls(
            enabled=bool(payload["enabled"]),
            mode=str(payload["mode"]),
            source=str(payload["source"]),
            scales=tuple(float(scale) for scale in payload["scales"]),
            training_experiment_kind=str(payload["training_experiment_kind"]),
            training_experiment_id=str(payload["training_experiment_id"]),
            evaluation_metrics=str(payload["evaluation_metrics"]),
            checkpoint_selection_metric=str(payload["checkpoint_selection_metric"]),
        )


@dataclass(frozen=True)
class RuntimeEnvironmentV1:
    num_users: int
    num_satellites: int
    beams_per_satellite: int
    user_lat_deg: float
    user_lon_deg: float
    r3_gap_scope: str
    r3_empty_beam_throughput: float
    user_heading_stride_rad: float
    user_scatter_radius_km: float
    user_scatter_distribution: str
    user_area_width_km: float
    user_area_height_km: float
    mobility_model: str
    random_wandering_max_turn_rad: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_users": int(self.num_users),
            "num_satellites": int(self.num_satellites),
            "beams_per_satellite": int(self.beams_per_satellite),
            "user_lat_deg": float(self.user_lat_deg),
            "user_lon_deg": float(self.user_lon_deg),
            "r3_gap_scope": self.r3_gap_scope,
            "r3_empty_beam_throughput": float(self.r3_empty_beam_throughput),
            "user_heading_stride_rad": float(self.user_heading_stride_rad),
            "user_scatter_radius_km": float(self.user_scatter_radius_km),
            "user_scatter_distribution": self.user_scatter_distribution,
            "user_area_width_km": float(self.user_area_width_km),
            "user_area_height_km": float(self.user_area_height_km),
            "mobility_model": self.mobility_model,
            "random_wandering_max_turn_rad": float(
                self.random_wandering_max_turn_rad
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeEnvironmentV1:
        return cls(
            num_users=int(payload["num_users"]),
            num_satellites=int(payload["num_satellites"]),
            beams_per_satellite=int(payload["beams_per_satellite"]),
            user_lat_deg=float(payload["user_lat_deg"]),
            user_lon_deg=float(payload["user_lon_deg"]),
            r3_gap_scope=str(payload["r3_gap_scope"]),
            r3_empty_beam_throughput=float(payload["r3_empty_beam_throughput"]),
            user_heading_stride_rad=float(payload["user_heading_stride_rad"]),
            user_scatter_radius_km=float(payload["user_scatter_radius_km"]),
            user_scatter_distribution=str(payload["user_scatter_distribution"]),
            user_area_width_km=float(payload["user_area_width_km"]),
            user_area_height_km=float(payload["user_area_height_km"]),
            mobility_model=str(payload["mobility_model"]),
            random_wandering_max_turn_rad=float(
                payload["random_wandering_max_turn_rad"]
            ),
        )


@dataclass(frozen=True)
class TrainingSummaryV1:
    episodes_requested: int
    episodes_completed: int
    elapsed_s: float
    final_episode_index: int
    final_scalar_reward: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes_requested": int(self.episodes_requested),
            "episodes_completed": int(self.episodes_completed),
            "elapsed_s": float(self.elapsed_s),
            "final_episode_index": int(self.final_episode_index),
            "final_scalar_reward": (
                None
                if self.final_scalar_reward is None
                else float(self.final_scalar_reward)
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingSummaryV1:
        final_scalar_reward = payload.get("final_scalar_reward")
        return cls(
            episodes_requested=int(payload["episodes_requested"]),
            episodes_completed=int(payload["episodes_completed"]),
            elapsed_s=float(payload["elapsed_s"]),
            final_episode_index=int(payload["final_episode_index"]),
            final_scalar_reward=(
                None if final_scalar_reward is None else float(final_scalar_reward)
            ),
        )


@dataclass(frozen=True)
class ResumeFromV1:
    path: str
    checkpoint_kind: str | None
    episode: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "checkpoint_kind": self.checkpoint_kind,
            "episode": self.episode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResumeFromV1:
        checkpoint_kind = payload.get("checkpoint_kind")
        episode = payload.get("episode")
        return cls(
            path=str(payload["path"]),
            checkpoint_kind=(
                None if checkpoint_kind is None else str(checkpoint_kind)
            ),
            episode=None if episode is None else int(episode),
        )


@dataclass(frozen=True)
class RunMetadataV1:
    """Current run-metadata contract with Phase 04B typing.

    ``config_path`` preserves the argv path passed to ``train_main`` at
    write time. Relative-vs-absolute form is intentionally preserved.
    """

    paper_id: str
    package_version: str
    config_path: str
    config_role: str | None
    resolved_config_snapshot: dict[str, Any]
    training_experiment: dict[str, Any] | None
    seeds: SeedsBlock
    checkpoint_rule: CheckpointRuleV1
    reward_calibration: RewardCalibrationV1
    checkpoint_files: CheckpointFilesV1
    resolved_assumptions: dict[str, Any]
    runtime_environment: RuntimeEnvironmentV1
    trainer_config: dict[str, Any]
    best_eval_summary: dict[str, Any] | None
    resume_from: ResumeFromV1 | None
    training_summary: TrainingSummaryV1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resolved_config_snapshot",
            _json_ready_mapping(self.resolved_config_snapshot),
        )
        object.__setattr__(
            self,
            "training_experiment",
            _json_ready_optional_mapping(self.training_experiment),
        )
        object.__setattr__(
            self,
            "resolved_assumptions",
            _json_ready_mapping(self.resolved_assumptions),
        )
        object.__setattr__(
            self,
            "trainer_config",
            _json_ready_mapping(self.trainer_config),
        )
        object.__setattr__(
            self,
            "best_eval_summary",
            _json_ready_optional_mapping(self.best_eval_summary),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "package_version": self.package_version,
            "config_path": self.config_path,
            "config_role": self.config_role,
            "resolved_config_snapshot": _json_ready_mapping(
                self.resolved_config_snapshot
            ),
            "training_experiment": _json_ready_optional_mapping(
                self.training_experiment
            ),
            "seeds": self.seeds.to_dict(),
            "checkpoint_rule": self.checkpoint_rule.to_dict(),
            "reward_calibration": self.reward_calibration.to_dict(),
            "checkpoint_files": self.checkpoint_files.to_dict(),
            "resolved_assumptions": _json_ready_mapping(self.resolved_assumptions),
            "runtime_environment": self.runtime_environment.to_dict(),
            "trainer_config": _json_ready_mapping(self.trainer_config),
            "best_eval_summary": _json_ready_optional_mapping(
                self.best_eval_summary
            ),
            "resume_from": (
                None if self.resume_from is None else self.resume_from.to_dict()
            ),
            "training_summary": self.training_summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunMetadataV1:
        config_role = payload.get("config_role")
        training_experiment = payload.get("training_experiment")
        best_eval_summary = payload.get("best_eval_summary")
        resume_from = payload.get("resume_from")
        return cls(
            paper_id=str(payload["paper_id"]),
            package_version=str(payload["package_version"]),
            config_path=str(payload["config_path"]),
            config_role=None if config_role is None else str(config_role),
            resolved_config_snapshot=_json_ready_mapping(
                payload["resolved_config_snapshot"]
            ),
            training_experiment=_json_ready_optional_mapping(training_experiment),
            seeds=SeedsBlock.from_dict(payload["seeds"]),
            checkpoint_rule=CheckpointRuleV1.from_dict(
                payload["checkpoint_rule"]
            ),
            reward_calibration=RewardCalibrationV1.from_dict(
                payload["reward_calibration"]
            ),
            checkpoint_files=CheckpointFilesV1.from_dict(
                payload["checkpoint_files"]
            ),
            resolved_assumptions=_json_ready_mapping(
                payload["resolved_assumptions"]
            ),
            runtime_environment=RuntimeEnvironmentV1.from_dict(
                payload["runtime_environment"]
            ),
            trainer_config=_json_ready_mapping(payload["trainer_config"]),
            best_eval_summary=_json_ready_optional_mapping(best_eval_summary),
            resume_from=(
                None
                if resume_from is None
                else ResumeFromV1.from_dict(resume_from)
            ),
            training_summary=TrainingSummaryV1.from_dict(
                payload["training_summary"]
            ),
        )


@dataclass(frozen=True)
class TrainingLogRow:
    episode: int
    epsilon: float
    r1_mean: float
    r2_mean: float
    r3_mean: float
    scalar_reward: float
    total_handovers: int
    replay_size: int
    losses: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": int(self.episode),
            "epsilon": float(self.epsilon),
            "r1_mean": float(self.r1_mean),
            "r2_mean": float(self.r2_mean),
            "r3_mean": float(self.r3_mean),
            "scalar_reward": float(self.scalar_reward),
            "total_handovers": int(self.total_handovers),
            "replay_size": int(self.replay_size),
            "losses": [float(value) for value in self.losses],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingLogRow:
        return cls(
            episode=int(payload["episode"]),
            epsilon=float(payload["epsilon"]),
            r1_mean=float(payload["r1_mean"]),
            r2_mean=float(payload["r2_mean"]),
            r3_mean=float(payload["r3_mean"]),
            scalar_reward=float(payload["scalar_reward"]),
            total_handovers=int(payload["total_handovers"]),
            replay_size=int(payload["replay_size"]),
            losses=tuple(float(value) for value in payload["losses"]),
        )


@dataclass(frozen=True)
class CheckpointPayloadV1:
    format_version: int
    checkpoint_kind: str
    episode: int
    train_seed: int
    env_seed: int
    mobility_seed: int
    state_dim: int
    action_dim: int
    trainer_config: dict[str, Any]
    checkpoint_rule: CheckpointRuleV1
    q_networks: list[dict[str, Any]]
    target_networks: list[dict[str, Any]]
    optimizers: list[dict[str, Any]] | None = None
    last_episode_log: dict[str, Any] | None = None
    evaluation_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": int(self.format_version),
            "checkpoint_kind": self.checkpoint_kind,
            "episode": int(self.episode),
            "train_seed": int(self.train_seed),
            "env_seed": int(self.env_seed),
            "mobility_seed": int(self.mobility_seed),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "trainer_config": _copy_mapping(self.trainer_config),
            "checkpoint_rule": self.checkpoint_rule.to_dict(),
            "q_networks": _copy_sequence(self.q_networks),
            "target_networks": _copy_sequence(self.target_networks),
        }
        if self.optimizers is not None:
            payload["optimizers"] = _copy_sequence(self.optimizers)
        if self.last_episode_log is not None:
            payload["last_episode_log"] = _copy_mapping(self.last_episode_log)
        if self.evaluation_summary is not None:
            payload["evaluation_summary"] = _copy_mapping(
                self.evaluation_summary
            )
        return payload

    def __getitem__(self, key: str) -> Any:
        if key == "checkpoint_rule":
            return self.checkpoint_rule.to_dict()
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        payload = self.to_dict()
        return payload.get(key, default)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointPayloadV1:
        optimizers = payload.get("optimizers")
        last_episode_log = payload.get("last_episode_log")
        evaluation_summary = payload.get("evaluation_summary")
        return cls(
            format_version=int(payload["format_version"]),
            checkpoint_kind=str(payload["checkpoint_kind"]),
            episode=int(payload["episode"]),
            train_seed=int(payload["train_seed"]),
            env_seed=int(payload["env_seed"]),
            mobility_seed=int(payload["mobility_seed"]),
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            trainer_config=_copy_mapping(payload["trainer_config"]),
            checkpoint_rule=CheckpointRuleV1.from_dict(
                payload["checkpoint_rule"]
            ),
            q_networks=_copy_sequence(payload["q_networks"]),
            target_networks=_copy_sequence(payload["target_networks"]),
            optimizers=_copy_optional_sequence(optimizers),
            last_episode_log=_copy_optional_mapping(last_episode_log),
            evaluation_summary=_copy_optional_mapping(evaluation_summary),
        )


@dataclass(frozen=True)
class CheckpointCatalog:
    """Paths to the two checkpoint files produced by one training run."""

    primary_final: Path
    secondary_best_eval: Path | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary_final", Path(self.primary_final))
        if self.secondary_best_eval is not None:
            object.__setattr__(
                self,
                "secondary_best_eval",
                Path(self.secondary_best_eval),
            )

    def to_v1(self) -> CheckpointFilesV1:
        return CheckpointFilesV1(
            primary_final=str(self.primary_final),
            secondary_best_eval=(
                None
                if self.secondary_best_eval is None
                else str(self.secondary_best_eval)
            ),
        )
