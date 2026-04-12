"""Scalar-reward DQN comparators for Table II and future sweep work."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..env.step import ActionMask, StepEnvironment, UserState
from .modqn import (
    DQNNetwork,
    EpisodeLog,
    EvalSummary,
    TrainerConfig,
    apply_reward_calibration,
    encode_state,
    scalarize_objectives,
    state_dim_for,
)


class ScalarReplayBuffer:
    """Fixed-capacity FIFO replay buffer for scalar-reward DQN."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf: list[tuple[np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]] = []
        self._cursor = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward_scalar: float,
        next_state: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        row = (state, action, reward_scalar, next_state, mask, next_mask, done)
        if len(self._buf) < self._capacity:
            self._buf.append(row)
        else:
            self._buf[self._cursor] = row
        self._cursor = (self._cursor + 1) % self._capacity

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[i] for i in indices]
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        masks = np.array([b[4] for b in batch], dtype=bool)
        next_masks = np.array([b[5] for b in batch], dtype=bool)
        dones = np.array([b[6] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, masks, next_masks, dones

    def __len__(self) -> int:
        return len(self._buf)


@dataclass(frozen=True)
class ScalarDQNPolicyConfig:
    """Policy metadata for scalar DQN comparators."""

    name: str
    scalar_reward_weights: tuple[float, float, float]


class ScalarDQNTrainer:
    """Single-network scalar-reward DQN for comparator runs."""

    def __init__(
        self,
        env: StepEnvironment,
        config: TrainerConfig,
        policy: ScalarDQNPolicyConfig,
        train_seed: int = 42,
        env_seed: int = 1337,
        mobility_seed: int = 7,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.config = config
        self.policy = policy
        self.device = torch.device(device)
        self.train_seed = train_seed
        self.env_seed = env_seed
        self.mobility_seed = mobility_seed

        self.num_beams = env.num_beams_total
        self.num_users = env.config.num_users
        self.state_dim = state_dim_for(self.num_beams)
        self.action_dim = self.num_beams

        torch.manual_seed(train_seed)
        self._train_rng = np.random.default_rng(train_seed)
        self._env_rng = np.random.default_rng(env_seed)
        self._mobility_rng = np.random.default_rng(mobility_seed)

        self.q_net = DQNNetwork(
            self.state_dim,
            self.action_dim,
            config.hidden_layers,
            config.activation,
        ).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.replay = ScalarReplayBuffer(config.replay_capacity)
        self._loss_fn = nn.MSELoss()
        self._best_eval_summary: EvalSummary | None = None
        self._best_q_state: dict | None = None
        self._best_target_state: dict | None = None

    def has_best_eval_checkpoint(self) -> bool:
        """Whether an eval-selected checkpoint is available."""
        return self._best_q_state is not None and self._best_target_state is not None

    def best_eval_summary(self) -> EvalSummary | None:
        """Return the best eval summary captured during training."""
        return self._best_eval_summary

    def epsilon(self, episode: int) -> float:
        cfg = self.config
        if episode >= cfg.epsilon_decay_episodes:
            return cfg.epsilon_end
        frac = episode / cfg.epsilon_decay_episodes
        return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac

    def _encode_states(self, states: list[UserState]) -> np.ndarray:
        return np.array(
            [encode_state(s, self.num_users, self.config) for s in states],
            dtype=np.float32,
        )

    def select_actions(
        self,
        states_encoded: np.ndarray,
        masks: list[ActionMask],
        eps: float,
    ) -> np.ndarray:
        actions = np.zeros(len(masks), dtype=np.int32)
        with torch.no_grad():
            st = torch.tensor(states_encoded, dtype=torch.float32, device=self.device)
            q_values = self.q_net(st).cpu().numpy()

        for uid, mask in enumerate(masks):
            valid = np.where(mask.mask)[0]
            if len(valid) == 0:
                actions[uid] = 0
                continue
            if self._train_rng.random() < eps:
                actions[uid] = int(self._train_rng.choice(valid))
            else:
                q_row = q_values[uid].copy()
                q_row[~mask.mask] = -np.inf
                actions[uid] = int(np.argmax(q_row))
        return actions

    def update(self) -> float:
        cfg = self.config
        if len(self.replay) < cfg.batch_size:
            return 0.0

        (
            states,
            actions,
            rewards,
            next_states,
            _masks,
            next_masks,
            dones,
        ) = self.replay.sample(cfg.batch_size, self._train_rng)

        st = torch.tensor(states, dtype=torch.float32, device=self.device)
        act = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rew = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        nm = torch.tensor(next_masks, dtype=torch.bool, device=self.device)
        dn = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_current = self.q_net(st).gather(1, act).squeeze(1)
        with torch.no_grad():
            q_next_all = self.target_net(ns)
            q_next_all[~nm] = -1e9
            q_next_max = q_next_all.max(dim=1).values
            target = rew + cfg.discount_factor * q_next_max * (1.0 - dn)

        loss = self._loss_fn(q_current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_targets(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _evaluate_one_seed(
        self,
        eval_seed: int,
        *,
        scalarization_weights: tuple[float, float, float] | None = None,
    ) -> dict[str, float]:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        weights = scalarization_weights or self.policy.scalar_reward_weights

        states, masks, _diag = self.env.reset(env_rng, mobility_rng)
        encoded = self._encode_states(states)
        ep_reward = np.zeros(3, dtype=np.float64)
        ep_handovers = 0

        for _step_idx in range(self.env.config.steps_per_episode):
            actions = self.select_actions(encoded, masks, eps=0.0)
            result = self.env.step(actions, env_rng)
            for rw in result.rewards:
                reward_vec = np.array(
                    [rw.r1_throughput, rw.r2_handover, rw.r3_load_balance],
                    dtype=np.float64,
                )
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
        scalarization_weights: tuple[float, float, float] | None = None,
    ) -> EvalSummary:
        if not evaluation_seed_set:
            raise ValueError("evaluation_seed_set must be non-empty for evaluation")
        rows = [
            self._evaluate_one_seed(seed, scalarization_weights=scalarization_weights)
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

    def restore_best_eval_checkpoint(self) -> None:
        if self._best_q_state is None or self._best_target_state is None:
            raise ValueError("No best-eval checkpoint is available")
        self.q_net.load_state_dict(copy.deepcopy(self._best_q_state))
        self.target_net.load_state_dict(copy.deepcopy(self._best_target_state))
        self.target_net.eval()

    def train(
        self,
        progress_every: int = 100,
        *,
        evaluation_seed_set: tuple[int, ...] | None = None,
        evaluation_every_episodes: int | None = None,
    ) -> list[EpisodeLog]:
        cfg = self.config
        eval_seeds = tuple(int(seed) for seed in (evaluation_seed_set or ()))
        eval_every = max(int(evaluation_every_episodes or cfg.target_update_every_episodes), 1)
        self._best_eval_summary = None
        self._best_q_state = None
        self._best_target_state = None

        logs: list[EpisodeLog] = []
        for ep in range(cfg.episodes):
            eps = self.epsilon(ep)
            states, masks, _diag = self.env.reset(self._env_rng, self._mobility_rng)
            encoded = self._encode_states(states)

            ep_reward = np.zeros(3, dtype=np.float64)
            ep_handovers = 0
            loss_sum = 0.0
            update_count = 0

            for _step_idx in range(self.env.config.steps_per_episode):
                actions = self.select_actions(encoded, masks, eps)
                result = self.env.step(actions, self._env_rng)
                next_encoded = self._encode_states(result.user_states)

                for uid in range(self.num_users):
                    rw = result.rewards[uid]
                    reward_vec = np.array(
                        [rw.r1_throughput, rw.r2_handover, rw.r3_load_balance],
                        dtype=np.float64,
                    )
                    reward_scalar = scalarize_objectives(
                        apply_reward_calibration(reward_vec, cfg),
                        self.policy.scalar_reward_weights,
                    )
                    self.replay.push(
                        encoded[uid],
                        int(actions[uid]),
                        reward_scalar,
                        next_encoded[uid],
                        masks[uid].mask.copy(),
                        result.action_masks[uid].mask.copy(),
                        result.done,
                    )
                    ep_reward += reward_vec
                    if rw.r2_handover < 0:
                        ep_handovers += 1

                step_loss = self.update()
                if step_loss > 0.0:
                    loss_sum += step_loss
                    update_count += 1

                encoded = next_encoded
                masks = result.action_masks
                if result.done:
                    break

            if (ep + 1) % cfg.target_update_every_episodes == 0:
                self.sync_targets()

            avg_reward = ep_reward / max(self.num_users, 1)
            scalar = scalarize_objectives(avg_reward, self.policy.scalar_reward_weights)
            avg_loss = loss_sum / max(update_count, 1)
            logs.append(
                EpisodeLog(
                    episode=ep,
                    epsilon=eps,
                    r1_mean=float(avg_reward[0]),
                    r2_mean=float(avg_reward[1]),
                    r3_mean=float(avg_reward[2]),
                    scalar_reward=scalar,
                    total_handovers=ep_handovers,
                    replay_size=len(self.replay),
                    losses=(avg_loss, 0.0, 0.0),
                )
            )

            if eval_seeds and (((ep + 1) % eval_every == 0) or (ep == cfg.episodes - 1)):
                eval_summary = self.evaluate_policy(
                    eval_seeds,
                    episode=ep,
                    evaluation_every_episodes=eval_every,
                )
                is_best = (
                    self._best_eval_summary is None
                    or eval_summary.mean_scalar_reward > self._best_eval_summary.mean_scalar_reward
                )
                if is_best:
                    self._best_eval_summary = eval_summary
                    self._best_q_state = copy.deepcopy(self.q_net.state_dict())
                    self._best_target_state = copy.deepcopy(self.target_net.state_dict())
                    if progress_every > 0:
                        print(
                            f"[{self.policy.name} eval {ep+1:5d}/{cfg.episodes}] "
                            f"best-mean-scalar={eval_summary.mean_scalar_reward:.4e} "
                            f"std={eval_summary.std_scalar_reward:.4e} "
                            f"seeds={len(eval_seeds)}"
                        )

            if progress_every > 0 and (ep + 1) % progress_every == 0:
                print(
                    f"[{self.policy.name} ep {ep+1:5d}/{cfg.episodes}] "
                    f"eps={eps:.3f} "
                    f"scalar={scalar:.4e} "
                    f"r1={avg_reward[0]:.4e} "
                    f"r2={avg_reward[1]:.4f} "
                    f"r3={avg_reward[2]:.4e} "
                    f"ho={ep_handovers} "
                    f"buf={len(self.replay)}"
                )

        return logs
