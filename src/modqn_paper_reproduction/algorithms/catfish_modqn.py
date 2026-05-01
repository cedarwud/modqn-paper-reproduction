"""Phase 04-B Catfish-MODQN sibling trainer.

This opt-in trainer preserves the original MODQN task surface and adds only
dual-agent replay/intervention mechanics.
"""

from __future__ import annotations

import copy
import resource
from collections import deque
from dataclasses import asdict
from typing import Any

import numpy as np
import torch.nn as nn
import torch.optim as optim

from ..env.step import StepEnvironment
from ..runtime.catfish_replay import (
    component_distribution_summary,
    distribution_summary,
    quality_score,
    sample_equal_budget_random_replay_batch,
    sample_mixed_replay_batch,
    sample_r2_guarded_mixed_replay_batch,
    target_catfish_sample_count,
)
from ..runtime.objective_math import apply_reward_calibration, scalarize_objectives
from ..runtime.replay_buffer import ReplayBuffer
from ..runtime.trainer_spec import EpisodeLog, EvalSummary, TrainerConfig
from .modqn import MODQNTrainer


class CatfishMODQNTrainer(MODQNTrainer):
    """MODQN trainer with Phase 04-B Catfish replay/intervention enabled."""

    def __init__(
        self,
        env: StepEnvironment,
        config: TrainerConfig,
        train_seed: int = 42,
        env_seed: int = 1337,
        mobility_seed: int = 7,
        device: str = "cpu",
    ) -> None:
        if not config.catfish_enabled:
            raise ValueError("CatfishMODQNTrainer requires catfish_enabled=True.")
        super().__init__(env, config, train_seed, env_seed, mobility_seed, device)

        self.catfish_challenger_enabled = bool(config.catfish_challenger_enabled)
        if self.catfish_challenger_enabled:
            self.catfish_q_nets = nn.ModuleList(
                [copy.deepcopy(net).to(self.device) for net in self.q_nets]
            )
            self.catfish_target_nets = nn.ModuleList(
                [copy.deepcopy(net).to(self.device) for net in self.catfish_q_nets]
            )
            for target in self.catfish_target_nets:
                target.eval()
            self.catfish_optimizers = [
                optim.Adam(net.parameters(), lr=config.learning_rate)
                for net in self.catfish_q_nets
            ]
        else:
            self.catfish_q_nets = nn.ModuleList()
            self.catfish_target_nets = nn.ModuleList()
            self.catfish_optimizers = []
        self.catfish_replay = ReplayBuffer(config.catfish_replay_capacity)

        self._quality_history: deque[float] = deque(
            maxlen=config.catfish_quality_threshold_window
        )
        self._catfish_quality_scores: deque[float] = deque(
            maxlen=config.catfish_replay_capacity
        )
        self._catfish_reward_components: deque[tuple[float, float, float]] = deque(
            maxlen=config.catfish_replay_capacity
        )
        self._episode_diagnostics: list[dict[str, Any]] = []
        self._main_update_count = 0
        self._catfish_update_count = 0
        self._intervention_trigger_count = 0
        self._intervention_skip_count = 0
        self._actual_catfish_samples_used = 0
        self._actual_random_control_samples_used = 0
        self._actual_main_samples_used_in_mixed = 0
        self._catfish_replay_training_warmup_skips = 0
        self._intervention_warmup_skip_count = 0
        self._main_replay_starved_updates = 0
        self._catfish_replay_starved_training = 0
        self._catfish_replay_starved_intervention = 0
        self._nan_detected = False
        self._sample_id_counter = 0
        self._catfish_lineage_records: deque[dict[str, Any]] = deque(maxlen=2_000)
        self._intervention_windows: deque[dict[str, Any]] = deque(maxlen=2_000)
        self._r2_admission_guard_pass_count = 0
        self._r2_admission_guard_skip_count = 0
        self._r2_intervention_guard_pass_count = 0
        self._r2_intervention_guard_skip_count = 0
        self._r2_guard_skip_reasons: dict[str, int] = {}
        self._r2_guard_batch_records: deque[dict[str, Any]] = deque(maxlen=2_000)
        self._recent_intervention_handover_rates: deque[float] = deque(
            maxlen=max(1, config.catfish_handover_spike_window)
        )

    def sync_targets(self) -> None:
        """Sync both the main and Catfish target networks."""
        super().sync_targets()
        if not self.catfish_challenger_enabled:
            return
        for i in range(3):
            self.catfish_target_nets[i].load_state_dict(
                self.catfish_q_nets[i].state_dict()
            )

    def _current_quality_threshold(self) -> float | None:
        cfg = self.config
        if cfg.catfish_quality_threshold_mode == "fixed":
            return cfg.catfish_quality_fixed_threshold
        if not self._quality_history:
            return None
        return float(
            np.quantile(
                np.asarray(self._quality_history, dtype=np.float64),
                cfg.catfish_quality_quantile,
            )
        )

    def _route_catfish_replay(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward_raw: np.ndarray,
        reward_train: np.ndarray,
        next_state: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
        episode_quality_scores: list[float],
        episode_thresholds: list[float],
    ) -> bool:
        cfg = self.config
        sample_id = self._sample_id_counter
        self._sample_id_counter += 1
        quality = quality_score(reward_raw, cfg.catfish_quality_weights)
        self._quality_history.append(quality)
        episode_quality_scores.append(quality)
        threshold = self._current_quality_threshold()
        if threshold is not None:
            episode_thresholds.append(float(threshold))

        warmed = len(self.replay) >= cfg.catfish_warmup_transitions
        high_value = (
            warmed
            and threshold is not None
            and quality >= threshold
        )
        if not high_value:
            return False
        if cfg.catfish_intervention_source_mode == "random-main-replay":
            return False
        if cfg.catfish_r2_admission_guard_enabled and float(reward_raw[1]) < 0.0:
            self._r2_admission_guard_skip_count += 1
            self._record_r2_guard_skip("r2-admission-negative-sample")
            return False

        self.catfish_replay.push(
            state,
            action,
            reward_train.astype(np.float32),
            next_state,
            mask,
            next_mask,
            done,
        )
        self._catfish_quality_scores.append(quality)
        self._catfish_reward_components.append(
            (float(reward_raw[0]), float(reward_raw[1]), float(reward_raw[2]))
        )
        if cfg.catfish_lineage_tracking_enabled:
            history = np.asarray(self._quality_history, dtype=np.float64)
            quality_quantile = (
                float(np.mean(history <= quality))
                if history.size
                else None
            )
            self._catfish_lineage_records.append(
                {
                    "sample_id": int(sample_id),
                    "source_buffer": "catfish",
                    "accepted_at_main_update": int(self._main_update_count),
                    "quality_score": float(quality),
                    "quality_threshold": (
                        None if threshold is None else float(threshold)
                    ),
                    "quality_quantile": quality_quantile,
                    "age_updates_at_accept": 0,
                    "use_count_observed": None,
                    "stale_or_delayed": False,
                }
            )
        if cfg.catfish_r2_admission_guard_enabled:
            self._r2_admission_guard_pass_count += 1
        return True

    def _record_r2_guard_skip(self, reason: str) -> None:
        self._r2_guard_skip_reasons[reason] = (
            self._r2_guard_skip_reasons.get(reason, 0) + 1
        )

    def _record_r2_guard_batch(self, composition: dict[str, Any]) -> None:
        self._r2_guard_batch_records.append(
            {
                "main_update_index": int(self._main_update_count + 1),
                "source_mode": composition.get("source_mode"),
                "source_counts": copy.deepcopy(composition.get("source_counts", {})),
                "matched_main_batch_r2_distribution": copy.deepcopy(
                    composition.get("matched_main_batch_r2_distribution", {})
                ),
                "injected_batch_r2_distribution": copy.deepcopy(
                    composition.get("injected_batch_r2_distribution", {})
                ),
                "r2_guard": copy.deepcopy(composition.get("r2_guard", {})),
            }
        )

    def _should_skip_for_handover_spike(
        self,
        current_episode_handover_rate: float | None,
    ) -> tuple[bool, dict[str, Any]]:
        cfg = self.config
        if not cfg.catfish_handover_spike_guard_enabled:
            return False, {
                "enabled": False,
                "reason": "handover-spike-guard-disabled",
            }
        if current_episode_handover_rate is None:
            return False, {
                "enabled": True,
                "reason": "current-handover-rate-unavailable",
            }
        recent = list(self._recent_intervention_handover_rates)
        if len(recent) < cfg.catfish_handover_spike_min_windows:
            return False, {
                "enabled": True,
                "reason": "insufficient-recent-intervention-windows",
                "current_episode_handover_rate": float(
                    current_episode_handover_rate
                ),
                "recent_window_count": int(len(recent)),
                "min_recent_windows": int(cfg.catfish_handover_spike_min_windows),
            }
        recent_mean = float(np.mean(recent))
        threshold = recent_mean + float(cfg.catfish_handover_spike_margin)
        skip = bool(float(current_episode_handover_rate) > threshold)
        return skip, {
            "enabled": True,
            "reason": (
                "recent-handover-spike"
                if skip
                else "recent-handover-rate-within-guard"
            ),
            "current_episode_handover_rate": float(current_episode_handover_rate),
            "recent_mean_handover_rate": recent_mean,
            "threshold": threshold,
            "recent_window_count": int(len(recent)),
            "margin": float(cfg.catfish_handover_spike_margin),
        }

    def _update_catfish_agent(self) -> tuple[tuple[float, float, float], dict[str, Any]]:
        cfg = self.config
        if not self.catfish_challenger_enabled:
            return (
                (0.0, 0.0, 0.0),
                {
                    "executed": False,
                    "reason": "challenger-disabled",
                    "q_abs_max": None,
                    "target_abs_max": None,
                    "nan_detected": False,
                },
            )
        if len(self.catfish_replay) < cfg.batch_size:
            if len(self.replay) >= cfg.catfish_warmup_transitions and len(
                self.catfish_replay
            ) == 0:
                self._catfish_replay_starved_training += 1
            else:
                self._catfish_replay_training_warmup_skips += 1
            return (
                (0.0, 0.0, 0.0),
                {
                    "executed": False,
                    "reason": "catfish-replay-below-batch-size",
                    "q_abs_max": None,
                    "target_abs_max": None,
                    "nan_detected": False,
                },
            )

        (
            states,
            actions,
            rewards,
            next_states,
            _masks,
            next_masks,
            dones,
        ) = self.catfish_replay.sample(cfg.batch_size, self._train_rng)
        losses, diagnostics = self._update_from_arrays(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            next_masks=next_masks,
            dones=dones,
            discount_factor=cfg.catfish_discount_factor,
            q_nets=self.catfish_q_nets,
            target_nets=self.catfish_target_nets,
            optimizers_=self.catfish_optimizers,
        )
        diagnostics["executed"] = True
        self._catfish_update_count += 1
        self._nan_detected = self._nan_detected or bool(
            diagnostics.get("nan_detected")
        )
        return losses, diagnostics

    def _sample_main_update_batch(
        self,
        *,
        current_episode_handover_rate: float | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        cfg = self.config
        if len(self.replay) < cfg.batch_size:
            self._main_replay_starved_updates += 1
            return None, None

        next_update_index = self._main_update_count + 1
        intervention_due = (
            cfg.catfish_intervention_enabled
            and next_update_index % cfg.catfish_intervention_period_updates == 0
        )
        if intervention_due:
            if cfg.catfish_intervention_source_mode == "random-main-replay":
                mixed = sample_equal_budget_random_replay_batch(
                    main_replay=self.replay,
                    batch_size=cfg.batch_size,
                    injected_ratio=cfg.catfish_intervention_catfish_ratio,
                    rng=self._train_rng,
                )
                mixed.composition["pre_update_main_replay_size"] = int(
                    len(self.replay)
                )
                mixed.composition["pre_update_catfish_replay_size"] = int(
                    len(self.catfish_replay)
                )
                self._intervention_trigger_count += 1
                self._actual_random_control_samples_used += int(
                    mixed.composition["actual_random_control_sample_count"]
                )
                self._actual_main_samples_used_in_mixed += int(
                    mixed.composition["actual_main_sample_count"]
                )
                self._record_r2_guard_batch(mixed.composition)
                return mixed, mixed.composition

            target_catfish_count = target_catfish_sample_count(
                cfg.batch_size,
                cfg.catfish_intervention_catfish_ratio,
            )
            required_catfish = max(
                cfg.catfish_min_catfish_replay_size,
                target_catfish_count,
            )
            if len(self.catfish_replay) >= required_catfish:
                if cfg.catfish_r2_intervention_guard_enabled:
                    skip_for_spike, spike_guard = self._should_skip_for_handover_spike(
                        current_episode_handover_rate
                    )
                    if skip_for_spike:
                        self._intervention_skip_count += 1
                        self._r2_intervention_guard_skip_count += 1
                        self._record_r2_guard_skip("recent-handover-spike")
                        return self._plain_main_replay_batch(), None

                    mixed, guard = sample_r2_guarded_mixed_replay_batch(
                        main_replay=self.replay,
                        catfish_replay=self.catfish_replay,
                        batch_size=cfg.batch_size,
                        catfish_ratio=cfg.catfish_intervention_catfish_ratio,
                        rng=self._train_rng,
                        max_attempts=cfg.catfish_r2_guard_max_batch_attempts,
                        strict_no_handover_samples=(
                            cfg.catfish_r2_strict_no_handover_sample_guard
                        ),
                    )
                    if mixed is None:
                        self._intervention_skip_count += 1
                        self._r2_intervention_guard_skip_count += 1
                        self._record_r2_guard_skip(
                            str(guard.get("reason", "r2-guard-skip"))
                        )
                        return self._plain_main_replay_batch(), None
                    mixed.composition["handover_spike_guard"] = spike_guard
                    self._r2_intervention_guard_pass_count += 1
                else:
                    mixed = sample_mixed_replay_batch(
                        main_replay=self.replay,
                        catfish_replay=self.catfish_replay,
                        batch_size=cfg.batch_size,
                        catfish_ratio=cfg.catfish_intervention_catfish_ratio,
                        rng=self._train_rng,
                    )
                mixed.composition["source_mode"] = "catfish-replay"
                mixed.composition["pre_update_main_replay_size"] = int(
                    len(self.replay)
                )
                mixed.composition["pre_update_catfish_replay_size"] = int(
                    len(self.catfish_replay)
                )
                self._intervention_trigger_count += 1
                self._actual_catfish_samples_used += int(
                    mixed.composition["actual_catfish_sample_count"]
                )
                self._actual_main_samples_used_in_mixed += int(
                    mixed.composition["actual_main_sample_count"]
                )
                if current_episode_handover_rate is not None:
                    mixed.composition["current_episode_handover_rate"] = float(
                        current_episode_handover_rate
                    )
                    self._recent_intervention_handover_rates.append(
                        float(current_episode_handover_rate)
                    )
                self._record_r2_guard_batch(mixed.composition)
                return mixed, mixed.composition

            self._intervention_skip_count += 1
            if len(self.replay) >= cfg.catfish_warmup_transitions and len(
                self.catfish_replay
            ) == 0:
                self._catfish_replay_starved_intervention += 1
            else:
                self._intervention_warmup_skip_count += 1

        (
            states,
            actions,
            rewards,
            next_states,
            masks,
            next_masks,
            dones,
        ) = self.replay.sample(cfg.batch_size, self._train_rng)
        composition = {
            "batch_size": int(cfg.batch_size),
            "configured_catfish_ratio": 0.0,
            "target_catfish_sample_count": 0,
            "actual_catfish_sample_count": 0,
            "actual_random_control_sample_count": 0,
            "actual_main_sample_count": int(cfg.batch_size),
            "actual_catfish_ratio": 0.0,
            "source_mode": "main-only",
            "source_counts": {
                "main": int(cfg.batch_size),
                "catfish": 0,
                "random-control": 0,
            },
        }
        batch = type(
            "MainReplayBatch",
            (),
            {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "masks": masks,
                "next_masks": next_masks,
                "dones": dones,
                "composition": composition,
            },
        )()
        return batch, None

    def _plain_main_replay_batch(self) -> Any:
        cfg = self.config
        (
            states,
            actions,
            rewards,
            next_states,
            masks,
            next_masks,
            dones,
        ) = self.replay.sample(cfg.batch_size, self._train_rng)
        composition = {
            "batch_size": int(cfg.batch_size),
            "configured_catfish_ratio": 0.0,
            "target_catfish_sample_count": 0,
            "actual_catfish_sample_count": 0,
            "actual_random_control_sample_count": 0,
            "actual_main_sample_count": int(cfg.batch_size),
            "actual_catfish_ratio": 0.0,
            "source_mode": "main-only",
            "source_counts": {
                "main": int(cfg.batch_size),
                "catfish": 0,
                "random-control": 0,
            },
        }
        return type(
            "MainReplayBatch",
            (),
            {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "masks": masks,
                "next_masks": next_masks,
                "dones": dones,
                "composition": composition,
            },
        )()

    def _update_main_agent(
        self,
        *,
        current_episode_handover_rate: float | None = None,
    ) -> tuple[tuple[float, float, float], dict[str, Any], dict[str, Any] | None]:
        batch, mixed_composition = self._sample_main_update_batch(
            current_episode_handover_rate=current_episode_handover_rate
        )
        if batch is None:
            return (
                (0.0, 0.0, 0.0),
                {
                    "executed": False,
                    "reason": "main-replay-below-batch-size",
                    "q_abs_max": None,
                    "target_abs_max": None,
                    "nan_detected": False,
                },
                None,
            )
        losses, diagnostics = self._update_from_arrays(
            states=batch.states,
            actions=batch.actions,
            rewards=batch.rewards,
            next_states=batch.next_states,
            next_masks=batch.next_masks,
            dones=batch.dones,
            discount_factor=self.config.discount_factor,
        )
        diagnostics["executed"] = True
        self._main_update_count += 1
        self._nan_detected = self._nan_detected or bool(
            diagnostics.get("nan_detected")
        )
        return losses, diagnostics, mixed_composition

    def _episode_diagnostic_row(
        self,
        *,
        episode: int,
        episode_quality_scores: list[float],
        episode_thresholds: list[float],
        episode_catfish_accepts: int,
        episode_interventions: list[dict[str, Any]],
        main_loss_rows: list[tuple[float, float, float]],
        catfish_loss_rows: list[tuple[float, float, float]],
        main_q_abs_max: list[float],
        catfish_q_abs_max: list[float],
        episode_action_counts: np.ndarray,
        episode_nan_detected: bool,
    ) -> dict[str, Any]:
        latest_threshold = (
            float(episode_thresholds[-1])
            if episode_thresholds
            else self._current_quality_threshold()
        )
        return {
            "episode": int(episode),
            "main_replay_size": int(len(self.replay)),
            "catfish_replay_size": int(len(self.catfish_replay)),
            "quality_threshold": latest_threshold,
            "quality_threshold_mode": self.config.catfish_quality_threshold_mode,
            "quality_percentile": float(self.config.catfish_quality_quantile),
            "phase07b_variant": self.config.catfish_phase07b_variant,
            "phase07d_variant": self.config.catfish_phase07d_variant,
            "quality_score_distribution": distribution_summary(
                list(self._catfish_quality_scores)
            ),
            "episode_quality_score_distribution": distribution_summary(
                episode_quality_scores
            ),
            "catfish_replay_reward_component_distribution": (
                component_distribution_summary(
                    list(self._catfish_reward_components)
                )
            ),
            "episode_catfish_replay_accept_count": int(episode_catfish_accepts),
            "intervention": {
                "episode_trigger_count": len(episode_interventions),
                "cumulative_trigger_count": int(self._intervention_trigger_count),
                "configured_catfish_ratio": float(
                    self.config.catfish_intervention_catfish_ratio
                ),
                "source_mode": self.config.catfish_intervention_source_mode,
                "actual_catfish_samples_used_cumulative": int(
                    self._actual_catfish_samples_used
                ),
                "actual_random_control_samples_used_cumulative": int(
                    self._actual_random_control_samples_used
                ),
                "actual_main_samples_used_in_mixed_cumulative": int(
                    self._actual_main_samples_used_in_mixed
                ),
                "episode_mixed_batch_compositions": episode_interventions,
                "skip_count_cumulative": int(self._intervention_skip_count),
                "warmup_skip_count_cumulative": int(
                    self._intervention_warmup_skip_count
                ),
            },
            "r2_handover_guard": self._r2_guard_diagnostics(),
            "td_loss": {
                "main": _loss_distribution(main_loss_rows),
                "catfish": _loss_distribution(catfish_loss_rows),
            },
            "q_stability": {
                "main_q_abs_max": distribution_summary(main_q_abs_max),
                "catfish_q_abs_max": distribution_summary(catfish_q_abs_max),
                "nan_detected": bool(episode_nan_detected),
            },
            "action_diversity": _action_diversity_payload(episode_action_counts),
            "replay_starvation": {
                "main_replay_starved_updates_cumulative": int(
                    self._main_replay_starved_updates
                ),
                "catfish_replay_starved_training_cumulative": int(
                    self._catfish_replay_starved_training
                ),
                "catfish_replay_starved_intervention_cumulative": int(
                    self._catfish_replay_starved_intervention
                ),
                "catfish_replay_training_warmup_skips_cumulative": int(
                    self._catfish_replay_training_warmup_skips
                ),
                "intervention_warmup_skips_cumulative": int(
                    self._intervention_warmup_skip_count
                ),
                "catfish_replay_empty_after_warmup": bool(
                    len(self.replay) >= self.config.catfish_warmup_transitions
                    and len(self.catfish_replay) == 0
                ),
                "by_source": {
                    "main": int(self._main_replay_starved_updates),
                    "catfish-training": int(self._catfish_replay_starved_training),
                    "catfish-intervention": int(
                        self._catfish_replay_starved_intervention
                    ),
                    "random-control": 0,
                },
            },
        }

    def catfish_diagnostics(self) -> dict[str, Any]:
        """Return JSON-ready Phase 04-B diagnostics."""
        final = self._episode_diagnostics[-1] if self._episode_diagnostics else {}
        return {
            "method_family": self.config.method_family,
            "training_experiment_kind": self.config.training_experiment_kind,
            "training_experiment_id": self.config.training_experiment_id,
            "ablation": self.config.catfish_ablation,
            "phase07b_variant": self.config.catfish_phase07b_variant,
            "phase07d_variant": self.config.catfish_phase07d_variant,
            "reward_surface": {
                "r1": "throughput",
                "r2": "handover penalty",
                "r3": "load balance",
                "competitive_shaping_enabled": bool(
                    self.config.catfish_competitive_shaping_enabled
                ),
            },
            "config": {
                "main_discount_factor": float(self.config.discount_factor),
                "catfish_discount_factor": float(
                    self.config.catfish_discount_factor
                ),
                "main_replay_capacity": int(self.config.replay_capacity),
                "catfish_replay_capacity": int(self.config.catfish_replay_capacity),
                "quality_weights": [
                    float(value) for value in self.config.catfish_quality_weights
                ],
                "quality_threshold_mode": self.config.catfish_quality_threshold_mode,
                "quality_quantile": float(self.config.catfish_quality_quantile),
                "quality_fixed_threshold": self.config.catfish_quality_fixed_threshold,
                "quality_threshold_window": int(
                    self.config.catfish_quality_threshold_window
                ),
                "warmup_trigger": self.config.catfish_warmup_trigger,
                "warmup_transitions": int(self.config.catfish_warmup_transitions),
                "partition_mode": self.config.catfish_partition_mode,
                "intervention_enabled": bool(
                    self.config.catfish_intervention_enabled
                ),
                "intervention_period_updates": int(
                    self.config.catfish_intervention_period_updates
                ),
                "intervention_source_mode": (
                    self.config.catfish_intervention_source_mode
                ),
                "challenger_enabled": bool(self.catfish_challenger_enabled),
                "lineage_tracking_enabled": bool(
                    self.config.catfish_lineage_tracking_enabled
                ),
                "configured_catfish_ratio": float(
                    self.config.catfish_intervention_catfish_ratio
                ),
                "min_catfish_replay_size": int(
                    self.config.catfish_min_catfish_replay_size
                ),
                "agent_count": 1 + int(self.catfish_challenger_enabled),
                "r2_handover_guard": {
                    "enabled": bool(self.config.catfish_r2_guard_enabled),
                    "admission_guard_enabled": bool(
                        self.config.catfish_r2_admission_guard_enabled
                    ),
                    "intervention_guard_enabled": bool(
                        self.config.catfish_r2_intervention_guard_enabled
                    ),
                    "strict_no_handover_sample_guard": bool(
                        self.config.catfish_r2_strict_no_handover_sample_guard
                    ),
                    "max_batch_attempts": int(
                        self.config.catfish_r2_guard_max_batch_attempts
                    ),
                    "handover_spike_guard_enabled": bool(
                        self.config.catfish_handover_spike_guard_enabled
                    ),
                    "handover_spike_window": int(
                        self.config.catfish_handover_spike_window
                    ),
                    "handover_spike_margin": float(
                        self.config.catfish_handover_spike_margin
                    ),
                    "handover_spike_min_windows": int(
                        self.config.catfish_handover_spike_min_windows
                    ),
                },
            },
            "cumulative": {
                "main_update_count": int(self._main_update_count),
                "catfish_update_count": int(self._catfish_update_count),
                "intervention_trigger_count": int(self._intervention_trigger_count),
                "intervention_skip_count": int(self._intervention_skip_count),
                "configured_catfish_ratio": float(
                    self.config.catfish_intervention_catfish_ratio
                ),
                "actual_catfish_samples_used_in_main_updates": int(
                    self._actual_catfish_samples_used
                ),
                "actual_random_control_samples_used_in_main_updates": int(
                    self._actual_random_control_samples_used
                ),
                "actual_main_samples_used_in_mixed_updates": int(
                    self._actual_main_samples_used_in_mixed
                ),
                "actual_catfish_ratio_in_mixed_updates": _safe_ratio(
                    self._actual_catfish_samples_used,
                    self._actual_catfish_samples_used
                    + self._actual_main_samples_used_in_mixed,
                ),
                "actual_injected_ratio_in_mixed_updates": _safe_ratio(
                    self._actual_catfish_samples_used
                    + self._actual_random_control_samples_used,
                    self._actual_catfish_samples_used
                    + self._actual_random_control_samples_used
                    + self._actual_main_samples_used_in_mixed,
                ),
                "catfish_replay_training_warmup_skips": int(
                    self._catfish_replay_training_warmup_skips
                ),
                "intervention_warmup_skip_count": int(
                    self._intervention_warmup_skip_count
                ),
                "nan_detected": bool(self._nan_detected),
            },
            "r2_handover_guard": self._r2_guard_diagnostics(),
            "final_replay": {
                "main_replay_size": int(len(self.replay)),
                "catfish_replay_size": int(len(self.catfish_replay)),
                "quality_threshold": final.get("quality_threshold"),
                "quality_score_distribution": final.get(
                    "quality_score_distribution",
                    distribution_summary([]),
                ),
                "catfish_replay_reward_component_distribution": final.get(
                    "catfish_replay_reward_component_distribution",
                    component_distribution_summary([]),
                ),
                "sample_lineage_summary": self._sample_lineage_summary(),
            },
            "intervention_utility": {
                "window_count": int(len(self._intervention_windows)),
                "windows": copy.deepcopy(list(self._intervention_windows)),
                "latest_window": (
                    copy.deepcopy(self._intervention_windows[-1])
                    if self._intervention_windows
                    else None
                ),
            },
            "runtime_cost": {
                "agent_count": 1 + int(self.catfish_challenger_enabled),
                "extra_learner_count": int(self.catfish_challenger_enabled),
                "max_rss_kb": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            },
            "replay_starvation": final.get("replay_starvation", {}),
            "episode_diagnostics": copy.deepcopy(self._episode_diagnostics),
        }

    def _r2_guard_diagnostics(self) -> dict[str, Any]:
        records = list(self._r2_guard_batch_records)
        guarded_records = [
            record
            for record in records
            if record.get("r2_guard", {}).get("enabled") is True
        ]
        guard_passed = [
            bool(record.get("r2_guard", {}).get("passed"))
            for record in guarded_records
        ]
        injected_negative_shares = [
            record.get("injected_batch_r2_distribution", {}).get("negative_share")
            for record in records
            if record.get("injected_batch_r2_distribution", {}).get("negative_share")
            is not None
        ]
        matched_main_negative_shares = [
            record.get("matched_main_batch_r2_distribution", {}).get("negative_share")
            for record in records
            if record.get("matched_main_batch_r2_distribution", {}).get(
                "negative_share"
            )
            is not None
        ]
        violations = [
            record
            for record in guarded_records
            if (
                record.get("injected_batch_r2_distribution", {}).get(
                    "negative_share"
                )
                is not None
                and record.get("matched_main_batch_r2_distribution", {}).get(
                    "negative_share"
                )
                is not None
                and float(
                    record["injected_batch_r2_distribution"]["negative_share"]
                )
                > float(
                    record["matched_main_batch_r2_distribution"]["negative_share"]
                )
                + 1e-12
            )
        ]
        return {
            "enabled": bool(self.config.catfish_r2_guard_enabled),
            "admission_guard_enabled": bool(
                self.config.catfish_r2_admission_guard_enabled
            ),
            "intervention_guard_enabled": bool(
                self.config.catfish_r2_intervention_guard_enabled
            ),
            "strict_no_handover_sample_guard": bool(
                self.config.catfish_r2_strict_no_handover_sample_guard
            ),
            "admission_guard_pass_count": int(self._r2_admission_guard_pass_count),
            "admission_guard_skip_count": int(self._r2_admission_guard_skip_count),
            "intervention_guard_pass_count": int(
                self._r2_intervention_guard_pass_count
            ),
            "intervention_guard_skip_count": int(
                self._r2_intervention_guard_skip_count
            ),
            "skip_reasons": dict(sorted(self._r2_guard_skip_reasons.items())),
            "guarded_batch_count": int(len(guarded_records)),
            "guarded_batch_pass_count": int(sum(guard_passed)),
            "guarded_batch_violation_count": int(len(violations)),
            "injected_batch_r2_negative_share_distribution": distribution_summary(
                [float(value) for value in injected_negative_shares]
            ),
            "matched_main_batch_r2_negative_share_distribution": distribution_summary(
                [float(value) for value in matched_main_negative_shares]
            ),
            "recent_batch_records": copy.deepcopy(records[-50:]),
        }

    def _sample_lineage_summary(self) -> dict[str, Any]:
        records = list(self._catfish_lineage_records)
        estimated_use_count_mean = _safe_ratio(
            self._actual_catfish_samples_used,
            len(records),
        )
        return {
            "tracking_enabled": bool(self.config.catfish_lineage_tracking_enabled),
            "accepted_sample_count": int(len(records)),
            "emitted_record_count": int(len(records)),
            "estimated_use_count_mean": estimated_use_count_mean,
            "fields": [
                "sample_id",
                "source_buffer",
                "accepted_at_main_update",
                "quality_score",
                "quality_threshold",
                "quality_quantile",
                "age_updates_at_accept",
                "use_count_observed",
                "stale_or_delayed",
            ],
            "recent_records": copy.deepcopy(records[-50:]),
        }

    def train(
        self,
        progress_every: int = 100,
        *,
        evaluation_seed_set: tuple[int, ...] | None = None,
        evaluation_every_episodes: int | None = None,
    ) -> list[EpisodeLog]:
        """Train with main replay plus Catfish high-value replay/intervention."""
        cfg = self.config
        logs: list[EpisodeLog] = []
        eval_seeds = tuple(int(seed) for seed in (evaluation_seed_set or ()))
        eval_every = max(
            int(evaluation_every_episodes or cfg.target_update_every_episodes),
            1,
        )

        self._best_eval_summary = None
        self._best_eval_payload = None
        self._evaluation_every_episodes = eval_every if eval_seeds else None
        self._evaluation_seed_set = eval_seeds
        self._secondary_checkpoint_enabled = bool(eval_seeds)
        if eval_seeds:
            self._secondary_checkpoint_status = (
                "best-eval checkpoint captured from mean weighted reward over "
                f"{len(eval_seeds)} evaluation seeds; evaluated every "
                f"{eval_every} episodes and at the final episode"
            )
        else:
            self._secondary_checkpoint_status = (
                "not-yet-implemented: no eval loop / best-eval checkpoint in this training run"
            )

        for ep in range(cfg.episodes):
            eps = self.epsilon(ep)
            states, masks, _diag = self.env.reset(
                self._env_rng, self._mobility_rng
            )
            encoded = self._encode_states(states)

            ep_reward = np.zeros(3, dtype=np.float64)
            ep_handovers = 0
            ep_losses = np.zeros(3, dtype=np.float64)
            update_count = 0
            episode_quality_scores: list[float] = []
            episode_thresholds: list[float] = []
            episode_catfish_accepts = 0
            episode_interventions: list[dict[str, Any]] = []
            main_loss_rows: list[tuple[float, float, float]] = []
            catfish_loss_rows: list[tuple[float, float, float]] = []
            main_q_abs_max: list[float] = []
            catfish_q_abs_max: list[float] = []
            episode_action_counts = np.zeros(self.action_dim, dtype=np.int64)
            episode_nan_detected = False

            for _step_idx in range(self.env.config.steps_per_episode):
                actions = self.select_actions(encoded, masks, eps)
                episode_action_counts += np.bincount(
                    actions.astype(np.int64),
                    minlength=self.action_dim,
                )
                result = self.env.step(actions, self._env_rng)
                next_encoded = self._encode_states(result.user_states)

                for uid in range(self.num_users):
                    rw = result.rewards[uid]
                    reward_vec = self.reward_vector_from_step_result(result, uid)
                    reward_vec_train = apply_reward_calibration(reward_vec, cfg)
                    self.replay.push(
                        encoded[uid],
                        int(actions[uid]),
                        reward_vec_train.astype(np.float32),
                        next_encoded[uid],
                        masks[uid].mask.copy(),
                        result.action_masks[uid].mask.copy(),
                        result.done,
                    )
                    accepted = self._route_catfish_replay(
                        state=encoded[uid],
                        action=int(actions[uid]),
                        reward_raw=reward_vec,
                        reward_train=reward_vec_train,
                        next_state=next_encoded[uid],
                        mask=masks[uid].mask.copy(),
                        next_mask=result.action_masks[uid].mask.copy(),
                        done=result.done,
                        episode_quality_scores=episode_quality_scores,
                        episode_thresholds=episode_thresholds,
                    )
                    if accepted:
                        episode_catfish_accepts += 1
                    ep_reward += reward_vec
                    if rw.r2_handover < 0:
                        ep_handovers += 1

                catfish_losses, catfish_diag = self._update_catfish_agent()
                if catfish_diag.get("executed"):
                    catfish_loss_rows.append(catfish_losses)
                    if _finite_losses(catfish_losses):
                        q_abs_max = catfish_diag.get("q_abs_max")
                        if q_abs_max is not None and np.isfinite(q_abs_max):
                            catfish_q_abs_max.append(float(q_abs_max))

                current_handover_rate = ep_handovers / max(
                    (_step_idx + 1) * self.num_users,
                    1,
                )
                main_losses, main_diag, mixed_composition = self._update_main_agent(
                    current_episode_handover_rate=current_handover_rate
                )
                if main_diag.get("executed"):
                    main_loss_rows.append(main_losses)
                    if _finite_losses(main_losses):
                        ep_losses += np.asarray(main_losses, dtype=np.float64)
                        update_count += 1
                        q_abs_max = main_diag.get("q_abs_max")
                        if q_abs_max is not None and np.isfinite(q_abs_max):
                            main_q_abs_max.append(float(q_abs_max))
                if mixed_composition is not None:
                    window = {
                        "episode": int(ep),
                        "main_update_index": int(self._main_update_count),
                        "composition": copy.deepcopy(mixed_composition),
                        "pre_update_td_loss": None,
                        "post_update_td_loss": [
                            float(value) for value in main_losses
                        ],
                        "post_update_q_abs_max": main_diag.get("q_abs_max"),
                        "post_update_target_abs_max": main_diag.get(
                            "target_abs_max"
                        ),
                        "action_distribution_after_update": (
                            _action_diversity_payload(episode_action_counts)
                        ),
                        "component_return_mean_so_far": [
                            float(value)
                            for value in (
                                ep_reward / max(self.num_users, 1)
                            ).tolist()
                        ],
                        "current_episode_handover_rate": float(
                            current_handover_rate
                        ),
                        "r2_guard": copy.deepcopy(
                            mixed_composition.get("r2_guard", {})
                        ),
                        "handover_spike_guard": copy.deepcopy(
                            mixed_composition.get("handover_spike_guard", {})
                        ),
                        "nan_detected": bool(main_diag.get("nan_detected")),
                    }
                    episode_interventions.append(copy.deepcopy(window))
                    self._intervention_windows.append(copy.deepcopy(window))

                episode_nan_detected = episode_nan_detected or bool(
                    main_diag.get("nan_detected")
                    or catfish_diag.get("nan_detected")
                )

                encoded = next_encoded
                masks = result.action_masks
                if result.done:
                    break

            if (ep + 1) % cfg.target_update_every_episodes == 0:
                self.sync_targets()

            avg_reward = ep_reward / max(self.num_users, 1)
            scalar = scalarize_objectives(avg_reward, cfg.objective_weights)
            avg_losses = ep_losses / max(update_count, 1)

            log = EpisodeLog(
                episode=ep,
                epsilon=eps,
                r1_mean=float(avg_reward[0]),
                r2_mean=float(avg_reward[1]),
                r3_mean=float(avg_reward[2]),
                scalar_reward=float(scalar),
                total_handovers=ep_handovers,
                replay_size=len(self.replay),
                losses=(
                    float(avg_losses[0]),
                    float(avg_losses[1]),
                    float(avg_losses[2]),
                ),
            )
            logs.append(log)
            self._episode_diagnostics.append(
                self._episode_diagnostic_row(
                    episode=ep,
                    episode_quality_scores=episode_quality_scores,
                    episode_thresholds=episode_thresholds,
                    episode_catfish_accepts=episode_catfish_accepts,
                    episode_interventions=episode_interventions,
                    main_loss_rows=main_loss_rows,
                    catfish_loss_rows=catfish_loss_rows,
                    main_q_abs_max=main_q_abs_max,
                    catfish_q_abs_max=catfish_q_abs_max,
                    episode_action_counts=episode_action_counts,
                    episode_nan_detected=episode_nan_detected,
                )
            )

            if self._secondary_checkpoint_enabled:
                should_evaluate = ((ep + 1) % eval_every == 0) or (
                    ep == cfg.episodes - 1
                )
                if should_evaluate:
                    eval_summary = self.evaluate_policy(
                        eval_seeds,
                        episode=ep,
                        evaluation_every_episodes=eval_every,
                    )
                    is_best = (
                        self._best_eval_summary is None
                        or eval_summary.mean_scalar_reward
                        > self._best_eval_summary.mean_scalar_reward
                    )
                    if is_best:
                        self._best_eval_summary = eval_summary
                        self._best_eval_payload = copy.deepcopy(
                            self.build_checkpoint_payload(
                                episode=ep,
                                checkpoint_kind=cfg.checkpoint_secondary_report,
                                logs=logs,
                                include_optimizers=True,
                                evaluation_summary=asdict(eval_summary),
                            )
                        )
                        if progress_every > 0:
                            print(
                                f"[eval {ep+1:5d}/{cfg.episodes}] "
                                f"best-mean-scalar={eval_summary.mean_scalar_reward:.4e} "
                                f"std={eval_summary.std_scalar_reward:.4e} "
                                f"seeds={len(eval_seeds)}"
                            )

            if progress_every > 0 and (ep + 1) % progress_every == 0:
                print(
                    f"[ep {ep+1:5d}/{cfg.episodes}] "
                    f"eps={eps:.3f} "
                    f"scalar={scalar:.4e} "
                    f"r1={avg_reward[0]:.4e} "
                    f"r2={avg_reward[1]:.4f} "
                    f"r3={avg_reward[2]:.4e} "
                    f"ho={ep_handovers} "
                    f"main_buf={len(self.replay)} "
                    f"catfish_buf={len(self.catfish_replay)} "
                    f"interventions={self._intervention_trigger_count}"
                )

        return logs


def _finite_losses(losses: tuple[float, float, float]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(losses, dtype=np.float64))))


def _loss_distribution(
    losses: list[tuple[float, float, float]],
) -> dict[str, Any]:
    if not losses:
        empty = distribution_summary([])
        return {"r1": empty, "r2": empty, "r3": empty}
    arr = np.asarray(losses, dtype=np.float64)
    return {
        "r1": distribution_summary(arr[:, 0].tolist()),
        "r2": distribution_summary(arr[:, 1].tolist()),
        "r3": distribution_summary(arr[:, 2].tolist()),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _action_diversity_payload(counts: np.ndarray) -> dict[str, Any]:
    total = int(np.sum(counts))
    if total <= 0:
        return {
            "total_actions": 0,
            "unique_action_count": 0,
            "max_action_share": None,
            "action_collapse_detected": False,
        }
    nonzero = counts[counts > 0]
    max_share = float(np.max(counts) / total)
    return {
        "total_actions": total,
        "unique_action_count": int(nonzero.size),
        "max_action_share": max_share,
        "action_collapse_detected": bool(max_share >= 0.95),
    }
