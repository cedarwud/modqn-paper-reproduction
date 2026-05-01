"""Phase 05-B Multi-Catfish-MODQN bounded pilot trainer."""

from __future__ import annotations

import copy
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
    sample_multi_source_replay_batch,
    target_source_sample_counts,
)
from ..runtime.objective_math import apply_reward_calibration, scalarize_objectives
from ..runtime.replay_buffer import ReplayBuffer
from ..runtime.trainer_spec import (
    EpisodeLog,
    PHASE_05_B_MULTI_CATFISH_KIND,
    TrainerConfig,
)
from .catfish_modqn import (
    _action_diversity_payload,
    _finite_losses,
    _loss_distribution,
    _safe_ratio,
)
from .modqn import MODQNTrainer


OBJECTIVE_SOURCES = ("r1", "r2", "r3")


class MultiCatfishMODQNTrainer(MODQNTrainer):
    """Main MODQN learner with objective-specialized Catfish replay sources."""

    def __init__(
        self,
        env: StepEnvironment,
        config: TrainerConfig,
        train_seed: int = 42,
        env_seed: int = 1337,
        mobility_seed: int = 7,
        device: str = "cpu",
    ) -> None:
        if config.training_experiment_kind != PHASE_05_B_MULTI_CATFISH_KIND:
            raise ValueError("MultiCatfishMODQNTrainer is Phase 05-B only.")
        if not config.catfish_enabled:
            raise ValueError("MultiCatfishMODQNTrainer requires catfish_enabled=True.")
        super().__init__(env, config, train_seed, env_seed, mobility_seed, device)

        self.objective_replays = {
            name: ReplayBuffer(config.catfish_replay_capacity)
            for name in OBJECTIVE_SOURCES
        }
        self._source_ratios = {
            name: float(ratio)
            for name, ratio in zip(OBJECTIVE_SOURCES, config.catfish_source_ratios)
        }
        specialist_names = (
            ("single",)
            if config.catfish_specialist_mode == "single-learner"
            else OBJECTIVE_SOURCES
        )
        self.specialist_q_nets = {
            name: nn.ModuleList([copy.deepcopy(net).to(self.device) for net in self.q_nets])
            for name in specialist_names
        }
        self.specialist_target_nets = {
            name: nn.ModuleList(
                [copy.deepcopy(net).to(self.device) for net in nets]
            )
            for name, nets in self.specialist_q_nets.items()
        }
        for targets in self.specialist_target_nets.values():
            for target in targets:
                target.eval()
        self.specialist_optimizers = {
            name: [optim.Adam(net.parameters(), lr=config.learning_rate) for net in nets]
            for name, nets in self.specialist_q_nets.items()
        }

        self._quality_history: deque[float] = deque(
            maxlen=config.catfish_quality_threshold_window
        )
        self._records: list[dict[str, Any]] = []
        self._scalar_reference_ids: set[int] = set()
        self._source_sample_ids: dict[str, set[int]] = {
            name: set() for name in OBJECTIVE_SOURCES
        }
        self._source_reward_components: dict[str, deque[tuple[float, float, float]]] = {
            name: deque(maxlen=config.catfish_replay_capacity)
            for name in OBJECTIVE_SOURCES
        }
        self._episode_diagnostics: list[dict[str, Any]] = []
        self._main_update_count = 0
        self._specialist_update_counts = {
            name: 0 for name in self.specialist_q_nets
        }
        self._intervention_trigger_count = 0
        self._intervention_skip_count = 0
        self._actual_source_samples_used = {name: 0 for name in OBJECTIVE_SOURCES}
        self._actual_main_samples_used_in_mixed = 0
        self._main_replay_starved_updates = 0
        self._intervention_warmup_skip_count = 0
        self._source_replay_starved_intervention = {
            name: 0 for name in OBJECTIVE_SOURCES
        }
        self._source_replay_starved_training = {
            name: 0 for name in self.specialist_q_nets
        }
        self._specialist_training_warmup_skips = {
            name: 0 for name in self.specialist_q_nets
        }
        self._nan_detected = False

    def sync_targets(self) -> None:
        """Sync main and specialist target networks."""
        super().sync_targets()
        for name, nets in self.specialist_q_nets.items():
            targets = self.specialist_target_nets[name]
            for idx in range(3):
                targets[idx].load_state_dict(nets[idx].state_dict())

    def _current_scalar_threshold(self) -> float | None:
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

    def _route_objective_replays(
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
    ) -> list[str]:
        cfg = self.config
        sample_id = len(self._records)
        reward_tuple = (
            float(reward_raw[0]),
            float(reward_raw[1]),
            float(reward_raw[2]),
        )
        quality = quality_score(reward_raw, cfg.catfish_quality_weights)
        self._quality_history.append(quality)
        scalar_threshold = self._current_scalar_threshold()
        warmed = len(self.replay) >= cfg.catfish_warmup_transitions
        scalar_admitted = (
            warmed
            and scalar_threshold is not None
            and quality >= scalar_threshold
        )
        if scalar_admitted:
            self._scalar_reference_ids.add(sample_id)

        accepted: list[str] = []
        if warmed:
            if (
                cfg.catfish_objective_admission_rule
                == "guarded-residual-objective-admission"
            ):
                accepted = self._guarded_residual_sources(
                    reward_raw=reward_raw,
                    scalar_admitted=bool(scalar_admitted),
                )
            elif (
                cfg.catfish_objective_admission_rule
                == "random-uniform-buffer-control"
            ):
                accepted = [
                    name
                    for name in OBJECTIVE_SOURCES
                    if self._train_rng.random()
                    < cfg.catfish_random_buffer_admission_probability
                ]

        for source in accepted:
            self.objective_replays[source].push(
                state,
                action,
                reward_train.astype(np.float32),
                next_state,
                mask,
                next_mask,
                done,
            )
            self._source_sample_ids[source].add(sample_id)
            self._source_reward_components[source].append(reward_tuple)

        self._records.append(
            {
                "sample_id": int(sample_id),
                "r1": reward_tuple[0],
                "r2": reward_tuple[1],
                "r3": reward_tuple[2],
                "scalar_quality": float(quality),
                "scalar_reference_admitted": bool(scalar_admitted),
                "accepted_sources": list(accepted),
            }
        )
        return accepted

    def _guarded_residual_sources(
        self,
        *,
        reward_raw: np.ndarray,
        scalar_admitted: bool,
    ) -> list[str]:
        cfg = self.config
        r1, r2, r3 = (float(reward_raw[0]), float(reward_raw[1]), float(reward_raw[2]))
        accepted: list[str] = []
        if (
            not scalar_admitted
            and r1 >= cfg.catfish_r1_threshold
            and r3 >= cfg.catfish_r1_r3_guardrail
        ):
            accepted.append("r1")
        if np.isclose(r2, cfg.catfish_r2_best_value):
            accepted.append("r2")
        if r3 >= cfg.catfish_r3_threshold and r1 >= cfg.catfish_r3_r1_guardrail:
            accepted.append("r3")
        return accepted

    def _sample_specialist_training_batch(self) -> tuple[Any, dict[str, Any]] | None:
        base_count = self.config.batch_size // len(OBJECTIVE_SOURCES)
        counts = {name: base_count for name in OBJECTIVE_SOURCES}
        for name in OBJECTIVE_SOURCES[: self.config.batch_size - sum(counts.values())]:
            counts[name] += 1
        for source, count in counts.items():
            if len(self.objective_replays[source]) < count:
                return None

        parts = [
            self.objective_replays[source].sample(counts[source], self._train_rng)
            for source in OBJECTIVE_SOURCES
            if counts[source] > 0
        ]
        stacked = tuple(
            np.concatenate([part[idx] for part in parts], axis=0)
            for idx in range(7)
        )
        sources = np.array(
            [
                source
                for source in OBJECTIVE_SOURCES
                for _ in range(counts[source])
                if counts[source] > 0
            ],
            dtype=object,
        )
        order = self._train_rng.permutation(self.config.batch_size)
        batch = tuple(part[order] for part in stacked)
        sources = sources[order]
        composition = {
            "target_source_sample_counts": counts,
            "source_counts": {
                source: int(np.sum(sources == source))
                for source in OBJECTIVE_SOURCES
            },
        }
        return (
            type(
                "SpecialistTrainingBatch",
                (),
                {
                    "states": batch[0],
                    "actions": batch[1],
                    "rewards": batch[2],
                    "next_states": batch[3],
                    "masks": batch[4],
                    "next_masks": batch[5],
                    "dones": batch[6],
                },
            )(),
            composition,
        )

    def _update_specialists(self) -> dict[str, tuple[tuple[float, float, float], dict[str, Any]]]:
        rows: dict[str, tuple[tuple[float, float, float], dict[str, Any]]] = {}
        for name in self.specialist_q_nets:
            if name == "single":
                sampled = self._sample_specialist_training_batch()
                if sampled is None:
                    self._specialist_training_warmup_skips[name] += 1
                    rows[name] = _skipped_specialist_row("objective-replays-below-batch-size")
                    continue
                batch, composition = sampled
            else:
                if len(self.objective_replays[name]) < self.config.batch_size:
                    if len(self.replay) >= self.config.catfish_warmup_transitions:
                        self._source_replay_starved_training[name] += 1
                    else:
                        self._specialist_training_warmup_skips[name] += 1
                    rows[name] = _skipped_specialist_row(
                        f"{name}-replay-below-batch-size"
                    )
                    continue
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    _masks,
                    next_masks,
                    dones,
                ) = self.objective_replays[name].sample(
                    self.config.batch_size,
                    self._train_rng,
                )
                batch = type(
                    "ObjectiveReplayBatch",
                    (),
                    {
                        "states": states,
                        "actions": actions,
                        "rewards": rewards,
                        "next_states": next_states,
                        "next_masks": next_masks,
                        "dones": dones,
                    },
                )()
                composition = {"source_counts": {name: self.config.batch_size}}

            losses, diagnostics = self._update_from_arrays(
                states=batch.states,
                actions=batch.actions,
                rewards=batch.rewards,
                next_states=batch.next_states,
                next_masks=batch.next_masks,
                dones=batch.dones,
                discount_factor=self.config.catfish_discount_factor,
                q_nets=self.specialist_q_nets[name],
                target_nets=self.specialist_target_nets[name],
                optimizers_=self.specialist_optimizers[name],
            )
            diagnostics["executed"] = True
            diagnostics["training_sample_composition"] = composition
            self._specialist_update_counts[name] += 1
            self._nan_detected = self._nan_detected or bool(
                diagnostics.get("nan_detected")
            )
            rows[name] = (losses, diagnostics)
        return rows

    def _sample_main_update_batch(self) -> tuple[Any, dict[str, Any] | None]:
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
            source_counts = target_source_sample_counts(
                batch_size=cfg.batch_size,
                source_ratios=self._source_ratios,
            )
            enough_sources = all(
                len(self.objective_replays[name])
                >= max(cfg.catfish_min_catfish_replay_size, count)
                for name, count in source_counts.items()
            )
            if enough_sources:
                mixed = sample_multi_source_replay_batch(
                    main_replay=self.replay,
                    source_replays=self.objective_replays,
                    batch_size=cfg.batch_size,
                    source_ratios=self._source_ratios,
                    rng=self._train_rng,
                )
                self._intervention_trigger_count += 1
                self._actual_main_samples_used_in_mixed += int(
                    mixed.composition["actual_main_sample_count"]
                )
                for source in OBJECTIVE_SOURCES:
                    self._actual_source_samples_used[source] += int(
                        mixed.composition["source_counts"][source]
                    )
                return mixed, mixed.composition

            self._intervention_skip_count += 1
            if len(self.replay) >= cfg.catfish_warmup_transitions:
                for source, count in source_counts.items():
                    if len(self.objective_replays[source]) < max(
                        cfg.catfish_min_catfish_replay_size,
                        count,
                    ):
                        self._source_replay_starved_intervention[source] += 1
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
            "actual_main_sample_count": int(cfg.batch_size),
            "actual_catfish_ratio": 0.0,
            "source_counts": {"main": int(cfg.batch_size), "r1": 0, "r2": 0, "r3": 0},
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

    def _update_main_agent(
        self,
    ) -> tuple[tuple[float, float, float], dict[str, Any], dict[str, Any] | None]:
        batch, mixed_composition = self._sample_main_update_batch()
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

    def catfish_diagnostics(self) -> dict[str, Any]:
        """Return JSON-ready Phase 05-B diagnostics."""
        final = self._episode_diagnostics[-1] if self._episode_diagnostics else {}
        total_source_samples = int(sum(self._actual_source_samples_used.values()))
        total_mixed = total_source_samples + self._actual_main_samples_used_in_mixed
        return {
            "method_family": self.config.method_family,
            "training_experiment_kind": self.config.training_experiment_kind,
            "training_experiment_id": self.config.training_experiment_id,
            "variant": self.config.catfish_phase05b_variant,
            "ablation": self.config.catfish_ablation,
            "reward_surface": {
                "r1": "throughput",
                "r2": "handover penalty",
                "r3": "load balance",
                "competitive_shaping_enabled": bool(
                    self.config.catfish_competitive_shaping_enabled
                ),
                "ee_or_catfish_ee_introduced": False,
            },
            "config": {
                "admission_rule": self.config.catfish_objective_admission_rule,
                "tie_policy": self.config.catfish_objective_tie_policy,
                "source_ratios": self._source_ratios,
                "total_catfish_ratio": float(
                    self.config.catfish_total_intervention_ratio
                ),
                "warmup_transitions": int(self.config.catfish_warmup_transitions),
                "specialist_mode": self.config.catfish_specialist_mode,
                "agent_count": int(1 + len(self.specialist_q_nets)),
                "replay_buffer_count": int(1 + len(self.objective_replays)),
            },
            "cumulative": {
                "main_update_count": int(self._main_update_count),
                "specialist_update_counts": {
                    name: int(count)
                    for name, count in self._specialist_update_counts.items()
                },
                "intervention_trigger_count": int(self._intervention_trigger_count),
                "intervention_skip_count": int(self._intervention_skip_count),
                "configured_catfish_ratio": float(
                    self.config.catfish_total_intervention_ratio
                ),
                "actual_catfish_samples_used_in_main_updates": total_source_samples,
                "actual_source_samples_used_in_main_updates": {
                    name: int(count)
                    for name, count in self._actual_source_samples_used.items()
                },
                "actual_main_samples_used_in_mixed_updates": int(
                    self._actual_main_samples_used_in_mixed
                ),
                "actual_catfish_ratio_in_mixed_updates": _safe_ratio(
                    total_source_samples,
                    total_mixed,
                ),
                "nan_detected": bool(self._nan_detected),
            },
            "final_replay": self._buffer_summary(),
            "overlap": self._overlap_summary(),
            "non_target_objective_damage": self._non_target_damage_table(),
            "replay_starvation": final.get("replay_starvation", {}),
            "runtime_cost": {
                "agent_count": int(1 + len(self.specialist_q_nets)),
                "replay_buffer_count": int(1 + len(self.objective_replays)),
                "stored_main_transitions": int(len(self.replay)),
                "stored_specialist_transitions": {
                    name: int(len(buf)) for name, buf in self.objective_replays.items()
                },
            },
            "episode_diagnostics": copy.deepcopy(self._episode_diagnostics),
        }

    def train(
        self,
        progress_every: int = 100,
        *,
        evaluation_seed_set: tuple[int, ...] | None = None,
        evaluation_every_episodes: int | None = None,
    ) -> list[EpisodeLog]:
        """Train the bounded Phase 05-B objective-buffer Catfish pilot."""
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
                self._env_rng,
                self._mobility_rng,
            )
            encoded = self._encode_states(states)

            ep_reward = np.zeros(3, dtype=np.float64)
            ep_handovers = 0
            ep_losses = np.zeros(3, dtype=np.float64)
            update_count = 0
            episode_source_accepts = {name: 0 for name in OBJECTIVE_SOURCES}
            episode_interventions: list[dict[str, Any]] = []
            main_loss_rows: list[tuple[float, float, float]] = []
            specialist_loss_rows: dict[str, list[tuple[float, float, float]]] = {
                name: [] for name in self.specialist_q_nets
            }
            main_q_abs_max: list[float] = []
            specialist_q_abs_max: dict[str, list[float]] = {
                name: [] for name in self.specialist_q_nets
            }
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
                    accepted = self._route_objective_replays(
                        state=encoded[uid],
                        action=int(actions[uid]),
                        reward_raw=reward_vec,
                        reward_train=reward_vec_train,
                        next_state=next_encoded[uid],
                        mask=masks[uid].mask.copy(),
                        next_mask=result.action_masks[uid].mask.copy(),
                        done=result.done,
                    )
                    for source in accepted:
                        episode_source_accepts[source] += 1
                    ep_reward += reward_vec
                    if rw.r2_handover < 0:
                        ep_handovers += 1

                specialist_rows = self._update_specialists()
                for name, (losses, diag) in specialist_rows.items():
                    if diag.get("executed"):
                        specialist_loss_rows[name].append(losses)
                        if _finite_losses(losses):
                            q_abs_max = diag.get("q_abs_max")
                            if q_abs_max is not None and np.isfinite(q_abs_max):
                                specialist_q_abs_max[name].append(float(q_abs_max))
                    episode_nan_detected = episode_nan_detected or bool(
                        diag.get("nan_detected")
                    )

                main_losses, main_diag, mixed_composition = self._update_main_agent()
                if main_diag.get("executed"):
                    main_loss_rows.append(main_losses)
                    if _finite_losses(main_losses):
                        ep_losses += np.asarray(main_losses, dtype=np.float64)
                        update_count += 1
                        q_abs_max = main_diag.get("q_abs_max")
                        if q_abs_max is not None and np.isfinite(q_abs_max):
                            main_q_abs_max.append(float(q_abs_max))
                if mixed_composition is not None:
                    episode_interventions.append(copy.deepcopy(mixed_composition))

                episode_nan_detected = episode_nan_detected or bool(
                    main_diag.get("nan_detected")
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
                    episode_source_accepts=episode_source_accepts,
                    episode_interventions=episode_interventions,
                    main_loss_rows=main_loss_rows,
                    specialist_loss_rows=specialist_loss_rows,
                    main_q_abs_max=main_q_abs_max,
                    specialist_q_abs_max=specialist_q_abs_max,
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
                    f"eps={eps:.3f} scalar={scalar:.4e} "
                    f"r1={avg_reward[0]:.4e} r2={avg_reward[1]:.4f} "
                    f"r3={avg_reward[2]:.4e} ho={ep_handovers} "
                    f"main_buf={len(self.replay)} "
                    f"buffers={self._buffer_sizes()}"
                )

        return logs

    def _episode_diagnostic_row(
        self,
        *,
        episode: int,
        episode_source_accepts: dict[str, int],
        episode_interventions: list[dict[str, Any]],
        main_loss_rows: list[tuple[float, float, float]],
        specialist_loss_rows: dict[str, list[tuple[float, float, float]]],
        main_q_abs_max: list[float],
        specialist_q_abs_max: dict[str, list[float]],
        episode_action_counts: np.ndarray,
        episode_nan_detected: bool,
    ) -> dict[str, Any]:
        return {
            "episode": int(episode),
            "main_replay_size": int(len(self.replay)),
            "objective_replay_sizes": self._buffer_sizes(),
            "episode_source_accept_counts": {
                name: int(count) for name, count in episode_source_accepts.items()
            },
            "buffer_diagnostics": self._buffer_summary(),
            "intervention": {
                "episode_trigger_count": len(episode_interventions),
                "cumulative_trigger_count": int(self._intervention_trigger_count),
                "configured_source_ratios": self._source_ratios,
                "episode_mixed_batch_compositions": episode_interventions,
                "skip_count_cumulative": int(self._intervention_skip_count),
                "warmup_skip_count_cumulative": int(
                    self._intervention_warmup_skip_count
                ),
            },
            "td_loss": {
                "main": _loss_distribution(main_loss_rows),
                "specialists": {
                    name: _loss_distribution(rows)
                    for name, rows in specialist_loss_rows.items()
                },
            },
            "q_stability": {
                "main_q_abs_max": distribution_summary(main_q_abs_max),
                "specialist_q_abs_max": {
                    name: distribution_summary(rows)
                    for name, rows in specialist_q_abs_max.items()
                },
                "nan_detected": bool(episode_nan_detected),
            },
            "action_diversity": _action_diversity_payload(episode_action_counts),
            "replay_starvation": {
                "main_replay_starved_updates_cumulative": int(
                    self._main_replay_starved_updates
                ),
                "source_replay_starved_training_cumulative": {
                    name: int(count)
                    for name, count in self._source_replay_starved_training.items()
                },
                "source_replay_starved_intervention_cumulative": {
                    name: int(count)
                    for name, count in self._source_replay_starved_intervention.items()
                },
                "specialist_training_warmup_skips_cumulative": {
                    name: int(count)
                    for name, count in self._specialist_training_warmup_skips.items()
                },
                "intervention_warmup_skips_cumulative": int(
                    self._intervention_warmup_skip_count
                ),
                "any_objective_replay_empty_after_warmup": bool(
                    len(self.replay) >= self.config.catfish_warmup_transitions
                    and any(len(buf) == 0 for buf in self.objective_replays.values())
                ),
            },
        }

    def _buffer_sizes(self) -> dict[str, int]:
        return {name: int(len(buf)) for name, buf in self.objective_replays.items()}

    def _buffer_summary(self) -> dict[str, Any]:
        sample_count = max(len(self._records), 1)
        return {
            "main_replay_size": int(len(self.replay)),
            "objective_buffers": {
                name: {
                    "size": int(len(self.objective_replays[name])),
                    "share": float(len(self.objective_replays[name]) / sample_count),
                    "threshold": self._threshold_payload(name),
                    "tie_mass": self._tie_mass(name),
                    "reward_component_distribution": component_distribution_summary(
                        list(self._source_reward_components[name])
                    ),
                }
                for name in OBJECTIVE_SOURCES
            },
        }

    def _threshold_payload(self, source: str) -> dict[str, Any]:
        if self.config.catfish_objective_admission_rule == "random-uniform-buffer-control":
            return {
                "rule": "random-uniform-buffer-control",
                "admission_probability_per_buffer": float(
                    self.config.catfish_random_buffer_admission_probability
                ),
            }
        if source == "r1":
            return {
                "rule": "scalar-distinct high-throughput residual with r3 guardrail",
                "r1_threshold": float(self.config.catfish_r1_threshold),
                "r3_guardrail": float(self.config.catfish_r1_r3_guardrail),
                "scalar_phase04_admitted_required": False,
                "tie_policy": self.config.catfish_objective_tie_policy,
            }
        if source == "r2":
            return {
                "rule": "complete strict best-score tie group",
                "r2_best_value": float(self.config.catfish_r2_best_value),
                "tie_policy": self.config.catfish_objective_tie_policy,
            }
        return {
            "rule": "load-balance threshold with throughput guardrail",
            "r3_threshold": float(self.config.catfish_r3_threshold),
            "r1_guardrail": float(self.config.catfish_r3_r1_guardrail),
            "tie_policy": self.config.catfish_objective_tie_policy,
        }

    def _tie_mass(self, source: str) -> int | None:
        if self.config.catfish_objective_admission_rule == "random-uniform-buffer-control":
            return None
        ids = self._source_sample_ids[source]
        if source == "r1":
            threshold = self.config.catfish_r1_threshold
            return sum(
                1
                for row in self._records
                if row["sample_id"] in ids and np.isclose(row["r1"], threshold)
            )
        if source == "r2":
            threshold = self.config.catfish_r2_best_value
            return sum(
                1
                for row in self._records
                if row["sample_id"] in ids and np.isclose(row["r2"], threshold)
            )
        threshold = self.config.catfish_r3_threshold
        return sum(
            1
            for row in self._records
            if row["sample_id"] in ids and np.isclose(row["r3"], threshold)
        )

    def _overlap_summary(self) -> dict[str, Any]:
        return {
            "pairwise_objective_jaccard": _pairwise_jaccard(self._source_sample_ids),
            "jaccard_vs_scalar_phase04_high_value": {
                name: _jaccard(ids, self._scalar_reference_ids)
                for name, ids in self._source_sample_ids.items()
            },
        }

    def _non_target_damage_table(self) -> dict[str, Any]:
        if not self._records:
            return {"available": False}
        rewards = np.asarray(
            [[row["r1"], row["r2"], row["r3"]] for row in self._records],
            dtype=np.float64,
        )
        all_means = np.mean(rewards, axis=0)
        tolerances = np.abs(all_means) * 0.05
        table: dict[str, Any] = {
            "available": True,
            "baseline_all_sample_means": _means_payload(all_means),
            "non_target_damage_tolerance": _means_payload(tolerances),
            "buffers": {},
        }
        sample_ids = np.asarray([row["sample_id"] for row in self._records], dtype=np.int64)
        for idx, source in enumerate(OBJECTIVE_SOURCES):
            ids = self._source_sample_ids[source]
            if not ids:
                table["buffers"][source] = {"available": False, "size": 0}
                continue
            mask = np.isin(sample_ids, list(ids))
            means = np.mean(rewards[mask], axis=0)
            deltas = means - all_means
            non_targets = [pos for pos in range(3) if pos != idx]
            damage = {
                OBJECTIVE_SOURCES[pos]: bool(deltas[pos] < -tolerances[pos])
                for pos in non_targets
            }
            table["buffers"][source] = {
                "available": True,
                "size": int(len(ids)),
                "means": _means_payload(means),
                "deltas_vs_all_samples_larger_is_better": _means_payload(deltas),
                "target_lift_visible": bool(deltas[idx] > 0.0),
                "non_target_damage": damage,
                "has_significant_non_target_damage": bool(any(damage.values())),
            }
        return table


def _skipped_specialist_row(reason: str) -> tuple[tuple[float, float, float], dict[str, Any]]:
    return (
        (0.0, 0.0, 0.0),
        {
            "executed": False,
            "reason": reason,
            "q_abs_max": None,
            "target_abs_max": None,
            "nan_detected": False,
        },
    )


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _pairwise_jaccard(sets: dict[str, set[int]]) -> dict[str, float]:
    names = tuple(sets)
    return {
        f"{left}_vs_{right}": _jaccard(sets[left], sets[right])
        for idx, left in enumerate(names)
        for right in names[idx + 1 :]
    }


def _means_payload(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(values[idx])
        for idx, name in enumerate(OBJECTIVE_SOURCES)
    }
