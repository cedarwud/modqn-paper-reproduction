"""MODQN trainer for PAP-2024-MORL-MULTIBEAM reproduction.

Three parallel DQNs (one per objective: throughput, handover, load balance).
Scalarized action selection with weight row at decision time.
Epsilon-greedy with action masking (ASSUME-MODQN-REP-012).
Experience replay with periodic hard target-network sync.

All trainer hyperparameters must come from the resolved-run config.
No hidden defaults — every knob is surfaced in TrainerConfig.

Paper-backed (SDD §3.5):
    hidden_layers, activation, learning_rate, discount_factor,
    batch_size, episodes, objective_weights.

Reproduction-assumption:
    epsilon schedule (ASSUME-MODQN-REP-004),
    target update cadence (ASSUME-MODQN-REP-005),
    replay capacity (ASSUME-MODQN-REP-006),
    policy sharing mode (ASSUME-MODQN-REP-007),
    state encoding/normalization (ASSUME-MODQN-REP-013).
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..artifacts import (
    CheckpointPayloadV1,
    CheckpointRuleV1,
    read_checkpoint,
    write_checkpoint,
)
from ..env.step import (
    ActionMask,
    StepEnvironment,
    UserState,
)
from ..runtime.objective_math import (
    apply_reward_calibration,
    scalarize_objectives,
    select_r1_reward_value,
)
from ..runtime.q_network import DQNNetwork
from ..runtime.replay_buffer import ReplayBuffer
from ..runtime.state_encoding import encode_state, state_dim_for
from ..runtime.trainer_spec import (
    EpisodeLog,
    EvalSummary,
    TrainerConfig,
)

class MODQNTrainer:
    """Multi-Objective DQN trainer per PAP-2024-MORL-MULTIBEAM.

    Usage::

        env = StepEnvironment(...)
        trainer = MODQNTrainer(env, config, ...)
        logs = trainer.train()
    """

    def __init__(
        self,
        env: StepEnvironment,
        config: TrainerConfig,
        train_seed: int = 42,
        env_seed: int = 1337,
        mobility_seed: int = 7,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device)
        self.train_seed = train_seed
        self.env_seed = env_seed
        self.mobility_seed = mobility_seed

        self.num_beams = env.num_beams_total
        self.num_users = env.config.num_users
        self.state_dim = state_dim_for(self.num_beams)
        self.action_dim = self.num_beams

        # Deterministic seed path (ASSUME-MODQN-REP-018)
        torch.manual_seed(train_seed)
        self._train_rng = np.random.default_rng(train_seed)
        self._env_rng = np.random.default_rng(env_seed)
        self._mobility_rng = np.random.default_rng(mobility_seed)

        # 3 parallel DQNs: Q1 (throughput), Q2 (handover), Q3 (load balance)
        self.q_nets = nn.ModuleList([
            DQNNetwork(
                self.state_dim, self.action_dim,
                config.hidden_layers, config.activation,
            ).to(self.device)
            for _ in range(3)
        ])
        self.target_nets = nn.ModuleList([
            copy.deepcopy(q).to(self.device) for q in self.q_nets
        ])
        for t in self.target_nets:
            t.eval()

        # Per-objective optimizers
        self.optimizers = [
            optim.Adam(self.q_nets[i].parameters(), lr=config.learning_rate)
            for i in range(3)
        ]

        # Replay buffer (ASSUME-MODQN-REP-006)
        self.replay = ReplayBuffer(config.replay_capacity)

        self._loss_fn = nn.MSELoss()
        self._loaded_checkpoint_metadata: dict[str, Any] | None = None
        self._secondary_checkpoint_enabled: bool = False
        self._secondary_checkpoint_status: str = (
            "not-yet-implemented: no eval loop / best-eval checkpoint in this training run"
        )
        self._evaluation_every_episodes: int | None = None
        self._evaluation_seed_set: tuple[int, ...] = ()
        self._best_eval_summary: EvalSummary | None = None
        self._best_eval_payload: CheckpointPayloadV1 | None = None

    # -- epsilon schedule (ASSUME-MODQN-REP-004) ----------------------------

    def epsilon(self, episode: int) -> float:
        """Linear epsilon decay from start to end over decay_episodes."""
        cfg = self.config
        if episode >= cfg.epsilon_decay_episodes:
            return cfg.epsilon_end
        frac = episode / cfg.epsilon_decay_episodes
        return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac

    # -- action selection ---------------------------------------------------

    def _predict_objective_q_values(
        self,
        states_encoded: np.ndarray,
    ) -> list[np.ndarray]:
        """Run the three objective networks on a batch of encoded states."""
        with torch.no_grad():
            st = torch.tensor(
                states_encoded, dtype=torch.float32, device=self.device
            )
            return [self.q_nets[i](st).cpu().numpy() for i in range(3)]

    def _scalarize_q_values(
        self,
        q_values: list[np.ndarray],
        objective_weights: tuple[float, float, float],
    ) -> np.ndarray:
        """Scalarize the three objective-Q tables with one weight row."""
        return (
            objective_weights[0] * q_values[0]
            + objective_weights[1] * q_values[1]
            + objective_weights[2] * q_values[2]
        )

    def _select_masked_greedy_action(
        self,
        scalarized_row: np.ndarray,
        mask: np.ndarray,
    ) -> int | None:
        """Return the greedy masked action or ``None`` when no action is valid."""
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return None
        q_row = scalarized_row.copy()
        q_row[~mask] = -np.inf
        return int(np.argmax(q_row))

    def _rank_masked_actions(
        self,
        scalarized_row: np.ndarray,
        mask: np.ndarray,
    ) -> list[int]:
        """Return valid actions sorted by scalarized-Q then beam index."""
        valid = np.flatnonzero(mask)
        return sorted(
            (int(action) for action in valid.tolist()),
            key=lambda action: (-float(scalarized_row[action]), int(action)),
        )

    def select_actions(
        self,
        states_encoded: np.ndarray,
        masks: list[ActionMask],
        eps: float,
        *,
        objective_weights: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Epsilon-greedy masked action selection with scalarized Q.

        At each decision step:
        1. Compute Q1, Q2, Q3 for each user
        2. Scalarize: Q_scalar = w1*Q1 + w2*Q2 + w3*Q3
        3. Mask invalid actions to -inf
        4. epsilon-greedy over the masked scalarized Q

        Returns actions array shape (num_users,).
        """
        U = len(masks)
        actions = np.zeros(U, dtype=np.int32)
        w = objective_weights or self.config.objective_weights
        q_values = self._predict_objective_q_values(states_encoded)
        scalarized = self._scalarize_q_values(q_values, w)

        for uid in range(U):
            valid = np.where(masks[uid].mask)[0]
            if len(valid) == 0:
                actions[uid] = 0
                continue

            if self._train_rng.random() < eps:
                actions[uid] = self._train_rng.choice(valid)
            else:
                selected_action = self._select_masked_greedy_action(
                    scalarized[uid],
                    masks[uid].mask,
                )
                actions[uid] = 0 if selected_action is None else selected_action

        return actions

    def select_actions_with_diagnostics(
        self,
        states_encoded: np.ndarray,
        masks: list[ActionMask],
        *,
        objective_weights: tuple[float, float, float] | None = None,
        top_k: int = 3,
    ) -> tuple[np.ndarray, list[dict[str, Any] | None]]:
        """Greedy masked action selection plus exporter-owned diagnostics.

        This helper is intentionally separate from ``select_actions()`` so
        training/evaluation call sites keep their existing API and epsilon
        semantics. It is meant for exporter-time replay of an already selected
        checkpoint only.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        U = len(masks)
        actions = np.zeros(U, dtype=np.int32)
        diagnostics: list[dict[str, Any] | None] = []
        w = objective_weights or self.config.objective_weights
        q_values = self._predict_objective_q_values(states_encoded)
        scalarized = self._scalarize_q_values(q_values, w)

        for uid in range(U):
            mask = masks[uid].mask
            selected_action = self._select_masked_greedy_action(
                scalarized[uid],
                mask,
            )
            if selected_action is None:
                actions[uid] = 0
                diagnostics.append(None)
                continue

            ordered_actions = self._rank_masked_actions(scalarized[uid], mask)
            if not ordered_actions or ordered_actions[0] != selected_action:
                raise ValueError(
                    "Policy diagnostics could not align the greedy selected action "
                    f"for user {uid}: selected={selected_action}, ordered={ordered_actions[:1]}."
                )

            valid_scalarized = scalarized[uid, mask]
            if not np.all(np.isfinite(valid_scalarized)):
                raise ValueError(
                    "Policy diagnostics require finite scalarized-Q values on the "
                    f"decision mask for user {uid}."
                )

            objective_q_values: list[np.ndarray] = []
            for obj_idx, q_table in enumerate(q_values):
                valid_objective_q = q_table[uid, mask]
                if not np.all(np.isfinite(valid_objective_q)):
                    raise ValueError(
                        "Policy diagnostics require finite objective-Q values on "
                        f"the decision mask for user {uid}, objective {obj_idx}."
                    )
                objective_q_values.append(q_table[uid])

            actions[uid] = selected_action
            runner_up_action = ordered_actions[1] if len(ordered_actions) > 1 else None
            selected_scalarized_q = float(scalarized[uid][selected_action])
            runner_up_scalarized_q = (
                float(scalarized[uid][runner_up_action])
                if runner_up_action is not None
                else None
            )
            scalarized_margin = (
                float(selected_scalarized_q - runner_up_scalarized_q)
                if runner_up_scalarized_q is not None
                else None
            )
            top_candidates = []
            for action in ordered_actions[:top_k]:
                top_candidates.append(
                    {
                        "action": int(action),
                        "validUnderDecisionMask": bool(mask[action]),
                        "objectiveQ": [
                            float(objective_q_values[0][action]),
                            float(objective_q_values[1][action]),
                            float(objective_q_values[2][action]),
                        ],
                        "scalarizedQ": float(scalarized[uid][action]),
                    }
                )

            diagnostics.append(
                {
                    "objectiveWeights": [float(value) for value in w],
                    "availableActionCount": int(np.sum(mask)),
                    "selectedAction": int(selected_action),
                    "selectedScalarizedQ": selected_scalarized_q,
                    "runnerUpAction": (
                        None if runner_up_action is None else int(runner_up_action)
                    ),
                    "runnerUpScalarizedQ": runner_up_scalarized_q,
                    "scalarizedMarginToRunnerUp": scalarized_margin,
                    "topCandidates": top_candidates,
                }
            )

        return actions, diagnostics

    # -- network update -----------------------------------------------------

    def _update_from_arrays(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_masks: np.ndarray,
        dones: np.ndarray,
        discount_factor: float,
        q_nets: nn.ModuleList | None = None,
        target_nets: nn.ModuleList | None = None,
        optimizers_: list[optim.Optimizer] | None = None,
    ) -> tuple[tuple[float, float, float], dict[str, Any]]:
        """Update one objective-network triplet from an explicit batch."""
        active_q_nets = self.q_nets if q_nets is None else q_nets
        active_target_nets = self.target_nets if target_nets is None else target_nets
        active_optimizers = self.optimizers if optimizers_ is None else optimizers_

        st = torch.tensor(states, dtype=torch.float32, device=self.device)
        act = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        nm = torch.tensor(next_masks, dtype=torch.bool, device=self.device)
        dn = torch.tensor(dones, dtype=torch.float32, device=self.device)

        losses: list[float] = []
        q_abs_max_values: list[float] = []
        target_abs_max_values: list[float] = []
        nan_detected = False
        for obj_idx in range(3):
            r = torch.tensor(
                rewards[:, obj_idx], dtype=torch.float32, device=self.device
            )

            # Current Q(s, a)
            q_current = active_q_nets[obj_idx](st).gather(1, act).squeeze(1)

            # Target: r + gamma * max_a' Q_target(s', a') where a' valid
            with torch.no_grad():
                q_next_all = active_target_nets[obj_idx](ns)
                # Mask invalid next-actions to large negative value
                q_next_all = q_next_all.masked_fill(~nm, -1e9)
                q_next_max = q_next_all.max(dim=1).values
                target = r + discount_factor * q_next_max * (1.0 - dn)

            loss = self._loss_fn(q_current, target)
            finite_update = (
                torch.isfinite(q_current).all()
                and torch.isfinite(target).all()
                and torch.isfinite(loss)
            )
            if not finite_update:
                nan_detected = True
                losses.append(float("nan"))
                continue

            active_optimizers[obj_idx].zero_grad()
            loss.backward()
            active_optimizers[obj_idx].step()

            losses.append(loss.item())
            q_abs_max_values.append(float(torch.max(torch.abs(q_current)).item()))
            target_abs_max_values.append(float(torch.max(torch.abs(target)).item()))

        diagnostics = {
            "q_abs_max": (
                max(q_abs_max_values) if q_abs_max_values else float("nan")
            ),
            "target_abs_max": (
                max(target_abs_max_values) if target_abs_max_values else float("nan")
            ),
            "nan_detected": bool(nan_detected),
        }
        return (losses[0], losses[1], losses[2]), diagnostics

    def update(self) -> tuple[float, float, float]:
        """Sample a batch from replay and update all 3 DQNs.

        Returns per-objective MSE losses (0.0 if replay too small).
        """
        cfg = self.config
        if len(self.replay) < cfg.batch_size:
            return (0.0, 0.0, 0.0)

        (
            states, actions, rewards, next_states,
            _masks, next_masks, dones,
        ) = self.replay.sample(cfg.batch_size, self._train_rng)

        st = torch.tensor(states, dtype=torch.float32, device=self.device)
        act = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        nm = torch.tensor(next_masks, dtype=torch.bool, device=self.device)
        dn = torch.tensor(dones, dtype=torch.float32, device=self.device)

        losses: list[float] = []
        for obj_idx in range(3):
            r = torch.tensor(
                rewards[:, obj_idx], dtype=torch.float32, device=self.device
            )

            # Current Q(s, a)
            q_current = self.q_nets[obj_idx](st).gather(1, act).squeeze(1)

            # Target: r + gamma * max_a' Q_target(s', a') where a' valid
            with torch.no_grad():
                q_next_all = self.target_nets[obj_idx](ns)
                # Mask invalid next-actions to large negative value
                q_next_all[~nm] = -1e9
                q_next_max = q_next_all.max(dim=1).values
                target = r + cfg.discount_factor * q_next_max * (1.0 - dn)

            loss = self._loss_fn(q_current, target)

            self.optimizers[obj_idx].zero_grad()
            loss.backward()
            self.optimizers[obj_idx].step()

            losses.append(loss.item())

        return (losses[0], losses[1], losses[2])

    # -- target sync (ASSUME-MODQN-REP-005) --------------------------------

    def sync_targets(self) -> None:
        """Hard copy online networks to target networks."""
        for i in range(3):
            self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())

    def _encode_states(self, states: list[UserState]) -> np.ndarray:
        """Encode a list of user states with the active trainer config."""
        return np.array(
            [encode_state(s, self.num_users, self.config) for s in states],
            dtype=np.float32,
        )

    def encode_states(self, states: list[UserState]) -> np.ndarray:
        """Public wrapper for the active state-encoding surface."""
        return self._encode_states(states)

    def reward_vector_from_step_result(
        self,
        result,
        uid: int,
    ) -> np.ndarray:
        """Return the trainer-selected three-objective reward vector.

        Baseline and MODQN-control configs keep ``r1`` as throughput. Phase 03
        EE-MODQN configs may explicitly gate ``r1`` to the per-user EE
        credit-assignment reward while preserving ``r2`` and ``r3`` unchanged.
        """
        rw = result.rewards[uid]
        r1 = select_r1_reward_value(
            throughput_bps=rw.r1_throughput,
            per_user_ee_credit_bps_per_w=rw.r1_energy_efficiency_credit,
            per_user_beam_ee_credit_bps_per_w=(
                rw.r1_beam_power_efficiency_credit
            ),
            config=self.config,
        )
        return np.array(
            [r1, rw.r2_handover, rw.r3_load_balance],
            dtype=np.float64,
        )

    def _evaluate_one_seed(
        self,
        eval_seed: int,
        *,
        objective_weights: tuple[float, float, float] | None = None,
    ) -> dict[str, float]:
        """Greedy rollout for one evaluation seed."""
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        weights = objective_weights or self.config.objective_weights

        states, masks, _diag = self.env.reset(env_rng, mobility_rng)
        encoded = self._encode_states(states)

        ep_reward = np.zeros(3, dtype=np.float64)
        ep_handovers = 0

        for _step_idx in range(self.env.config.steps_per_episode):
            actions = self.select_actions(
                encoded,
                masks,
                eps=0.0,
                objective_weights=weights,
            )
            result = self.env.step(actions, env_rng)

            for uid, rw in enumerate(result.rewards):
                reward_vec = self.reward_vector_from_step_result(result, uid)
                ep_reward += reward_vec
                if rw.r2_handover < 0:
                    ep_handovers += 1

            if result.done:
                break

            encoded = self._encode_states(result.user_states)
            masks = result.action_masks

        avg_reward = ep_reward / max(self.num_users, 1)
        scalar = scalarize_objectives(avg_reward, weights)

        return {
            "scalar_reward": float(scalar),
            "r1_mean": float(avg_reward[0]),
            "r2_mean": float(avg_reward[1]),
            "r3_mean": float(avg_reward[2]),
            "total_handovers": float(ep_handovers),
        }

    def evaluate_policy(
        self,
        evaluation_seed_set: tuple[int, ...],
        *,
        episode: int,
        evaluation_every_episodes: int,
        objective_weights: tuple[float, float, float] | None = None,
    ) -> EvalSummary:
        """Evaluate the greedy policy over the configured evaluation seeds."""
        if not evaluation_seed_set:
            raise ValueError("evaluation_seed_set must be non-empty for evaluation")

        rows = [
            self._evaluate_one_seed(seed, objective_weights=objective_weights)
            for seed in evaluation_seed_set
        ]

        def mean_std(key: str) -> tuple[float, float]:
            values = np.array([row[key] for row in rows], dtype=np.float64)
            return float(np.mean(values)), float(np.std(values))

        scalar_mean, scalar_std = mean_std("scalar_reward")
        r1_mean, r1_std = mean_std("r1_mean")
        r2_mean, r2_std = mean_std("r2_mean")
        r3_mean, r3_std = mean_std("r3_mean")
        handover_mean, handover_std = mean_std("total_handovers")

        return EvalSummary(
            episode=episode,
            evaluation_every_episodes=evaluation_every_episodes,
            eval_seeds=tuple(int(seed) for seed in evaluation_seed_set),
            mean_scalar_reward=scalar_mean,
            std_scalar_reward=scalar_std,
            mean_r1=r1_mean,
            std_r1=r1_std,
            mean_r2=r2_mean,
            std_r2=r2_std,
            mean_r3=r3_mean,
            std_r3=r3_std,
            mean_total_handovers=handover_mean,
            std_total_handovers=handover_std,
        )

    # -- checkpointing (ASSUME-MODQN-REP-015) ------------------------------

    def checkpoint_rule(self) -> CheckpointRuleV1:
        """Return the active checkpoint-selection rule for metadata/logging."""
        return CheckpointRuleV1(
            assumption_id=self.config.checkpoint_assumption_id,
            primary_report=self.config.checkpoint_primary_report,
            secondary_report=self.config.checkpoint_secondary_report,
            secondary_implemented=self._secondary_checkpoint_enabled,
            secondary_status=self._secondary_checkpoint_status,
        )

    def build_checkpoint_payload(
        self,
        *,
        episode: int,
        checkpoint_kind: str,
        logs: list[EpisodeLog] | None = None,
        include_optimizers: bool = True,
        evaluation_summary: dict[str, Any] | None = None,
    ) -> CheckpointPayloadV1:
        """Build a serializable checkpoint payload."""
        summary = None
        if evaluation_summary is not None:
            summary = copy.deepcopy(evaluation_summary)
            if "eval_seeds" in summary:
                summary["eval_seeds"] = [int(seed) for seed in summary["eval_seeds"]]

        return CheckpointPayloadV1(
            format_version=1,
            checkpoint_kind=checkpoint_kind,
            episode=episode,
            train_seed=self.train_seed,
            env_seed=self.env_seed,
            mobility_seed=self.mobility_seed,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            trainer_config=asdict(self.config),
            checkpoint_rule=self.checkpoint_rule(),
            q_networks=[net.state_dict() for net in self.q_nets],
            target_networks=[net.state_dict() for net in self.target_nets],
            optimizers=(
                [optimizer.state_dict() for optimizer in self.optimizers]
                if include_optimizers
                else None
            ),
            last_episode_log=asdict(logs[-1]) if logs else None,
            evaluation_summary=summary,
        )

    def _write_checkpoint_payload(
        self,
        path: str | Path,
        payload: CheckpointPayloadV1,
    ) -> Path:
        """Write a pre-built checkpoint payload to disk."""
        return write_checkpoint(Path(path), payload)

    def _load_checkpoint_payload(
        self,
        payload: CheckpointPayloadV1,
        *,
        checkpoint_path: str | Path | None = None,
        load_optimizers: bool = True,
    ) -> CheckpointPayloadV1:
        """Load trainer weights/state from a checkpoint payload."""
        for net, state in zip(self.q_nets, payload.q_networks):
            net.load_state_dict(state)
        for net, state in zip(self.target_nets, payload.target_networks):
            net.load_state_dict(state)
            net.eval()

        optimizer_loaded = False
        if load_optimizers and payload.optimizers is not None:
            for optimizer, state in zip(self.optimizers, payload.optimizers):
                optimizer.load_state_dict(state)
            optimizer_loaded = True

        self._loaded_checkpoint_metadata = {
            "path": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_kind": payload.checkpoint_kind,
            "episode": payload.episode,
            "checkpoint_rule": payload.checkpoint_rule.to_dict(),
            "optimizer_loaded": optimizer_loaded,
            "evaluation_summary": copy.deepcopy(payload.evaluation_summary),
        }
        return payload

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        episode: int,
        checkpoint_kind: str,
        logs: list[EpisodeLog] | None = None,
        include_optimizers: bool = True,
        evaluation_summary: dict[str, Any] | None = None,
    ) -> Path:
        """Save the current trainer weights/state to disk."""
        payload = self.build_checkpoint_payload(
            episode=episode,
            checkpoint_kind=checkpoint_kind,
            logs=logs,
            include_optimizers=include_optimizers,
            evaluation_summary=evaluation_summary,
        )
        return self._write_checkpoint_payload(path, payload)

    def has_best_eval_checkpoint(self) -> bool:
        """Whether an eval-selected secondary checkpoint is available."""
        return self._best_eval_payload is not None

    def best_eval_summary(self) -> EvalSummary | None:
        """Return the current best eval summary, if one exists."""
        return self._best_eval_summary

    def save_best_eval_checkpoint(self, path: str | Path) -> Path:
        """Persist the best eval-selected checkpoint discovered during training."""
        if self._best_eval_payload is None:
            raise ValueError("No best-eval checkpoint payload is available")
        return self._write_checkpoint_payload(path, copy.deepcopy(self._best_eval_payload))

    def restore_best_eval_checkpoint(
        self,
        *,
        load_optimizers: bool = True,
    ) -> dict[str, Any]:
        """Restore the in-memory best-eval checkpoint into the live trainer."""
        if self._best_eval_payload is None:
            raise ValueError("No best-eval checkpoint payload is available")
        payload = self._load_checkpoint_payload(
            copy.deepcopy(self._best_eval_payload),
            checkpoint_path="<in-memory-best-eval>",
            load_optimizers=load_optimizers,
        )
        return payload.to_dict()

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizers: bool = True,
    ) -> dict[str, Any]:
        """Load trainer weights/state from a checkpoint file."""
        checkpoint_path = Path(path)
        payload = read_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        loaded = self._load_checkpoint_payload(
            payload,
            checkpoint_path=checkpoint_path,
            load_optimizers=load_optimizers,
        )
        return loaded.to_dict()

    # -- training loop ------------------------------------------------------

    def train(
        self,
        progress_every: int = 100,
        *,
        evaluation_seed_set: tuple[int, ...] | None = None,
        evaluation_every_episodes: int | None = None,
    ) -> list[EpisodeLog]:
        """Full MODQN training loop.

        Returns a list of per-episode metrics.
        """
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

            # Reset environment
            states, masks, _diag = self.env.reset(
                self._env_rng, self._mobility_rng
            )

            # Encode states
            encoded = self._encode_states(states)

            ep_reward = np.zeros(3, dtype=np.float64)
            ep_handovers = 0
            ep_losses = np.zeros(3, dtype=np.float64)
            update_count = 0

            for _step_idx in range(self.env.config.steps_per_episode):
                # Select actions
                actions = self.select_actions(encoded, masks, eps)

                # Step environment
                result = self.env.step(actions, self._env_rng)

                # Encode next states
                next_encoded = self._encode_states(result.user_states)

                # Store transitions per user (ASSUME-MODQN-REP-007: shared policy)
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
                    ep_reward += reward_vec
                    if rw.r2_handover < 0:
                        ep_handovers += 1

                # Update networks
                step_losses = self.update()
                if step_losses[0] > 0:
                    ep_losses += step_losses
                    update_count += 1

                # Advance
                encoded = next_encoded
                masks = result.action_masks

                if result.done:
                    break

            # Target network sync (ASSUME-MODQN-REP-005)
            if (ep + 1) % cfg.target_update_every_episodes == 0:
                self.sync_targets()

            # Record metrics
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
                losses=(float(avg_losses[0]), float(avg_losses[1]), float(avg_losses[2])),
            )
            logs.append(log)

            if self._secondary_checkpoint_enabled:
                should_evaluate = ((ep + 1) % eval_every == 0) or (ep == cfg.episodes - 1)
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
                    f"buf={len(self.replay)}"
                )

        return logs
