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
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import Generator

from ..env.step import (
    ActionMask,
    StepEnvironment,
    UserState,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainerConfig:
    """All trainer hyperparameters — none may be hidden in code.

    Paper-backed (SDD §3.5):
        hidden_layers, activation, learning_rate, discount_factor,
        batch_size, episodes, objective_weights.

    Reproduction-assumption:
        epsilon_* (ASSUME-MODQN-REP-004),
        target_update_every_episodes (ASSUME-MODQN-REP-005),
        replay_capacity (ASSUME-MODQN-REP-006),
        policy_sharing_mode (ASSUME-MODQN-REP-007),
        state encoding fields (ASSUME-MODQN-REP-013).
    """

    # -- paper-backed (SDD §3.5) ------------------------------------------
    hidden_layers: tuple[int, ...] = (100, 50, 50)
    activation: str = "tanh"
    learning_rate: float = 0.01
    discount_factor: float = 0.9
    batch_size: int = 128
    episodes: int = 9000
    objective_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)

    # -- ASSUME-MODQN-REP-004: epsilon schedule ----------------------------
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 7000

    # -- ASSUME-MODQN-REP-005: target-network update -----------------------
    target_update_every_episodes: int = 50

    # -- ASSUME-MODQN-REP-006: replay buffer -------------------------------
    replay_capacity: int = 50_000

    # -- ASSUME-MODQN-REP-007: policy sharing ------------------------------
    policy_sharing_mode: str = "shared"

    # -- ASSUME-MODQN-REP-013: state encoding / normalization --------------
    snr_encoding: str = "log1p"
    offset_scale_km: float = 100.0
    load_normalization: str = "divide_by_num_users"

    # -- ASSUME-MODQN-REP-015: checkpoint selection rule -------------------
    checkpoint_assumption_id: str = "ASSUME-MODQN-REP-015"
    checkpoint_primary_report: str = "final-episode-policy"
    checkpoint_secondary_report: str = "best-weighted-reward-on-eval"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class DQNNetwork(nn.Module):
    """Single DQN for one reward objective.

    Architecture: state_dim -> hidden[0] -> act -> ... -> hidden[-1] -> act -> action_dim
    Paper specifies [100, 50, 50] with tanh activation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        act_cls = nn.Tanh if activation == "tanh" else nn.ReLU
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity FIFO experience replay (ASSUME-MODQN-REP-006).

    Each transition stores: (state, action, reward_3, next_state,
    mask, next_mask, done).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf: deque[tuple[Any, ...]] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward_3: np.ndarray,
        next_state: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward_3, next_state, mask, next_mask, done))

    def sample(
        self, batch_size: int, rng: Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[i] for i in indices]

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        masks = np.array([b[4] for b in batch])
        next_masks = np.array([b[5] for b in batch])
        dones = np.array([b[6] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, masks, next_masks, dones

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# State encoding (ASSUME-MODQN-REP-013)
# ---------------------------------------------------------------------------


def encode_state(
    user_state: UserState,
    num_users: int,
    config: TrainerConfig,
) -> np.ndarray:
    """Encode a UserState into a flat numpy vector.

    ASSUME-MODQN-REP-013 state encoding contract:
        [access_vector, encoded_snr, encoded_offsets, encoded_loads]

    Encoding rules (all explicitly configured, no hidden transforms):
        - access_vector: raw one-hot (already 0/1)
        - channel_quality: log1p(snr_linear) — bounded, monotonic
        - beam_offsets: flatten(offsets_km) / offset_scale_km
        - beam_loads: loads / num_users
    """
    access = user_state.access_vector.astype(np.float32)

    # Channel quality encoding
    snr = user_state.channel_quality.astype(np.float64)
    if config.snr_encoding == "log1p":
        snr = np.log1p(np.maximum(snr, 0.0)).astype(np.float32)
    else:
        snr = snr.astype(np.float32)

    # Beam offsets: flatten (east, north) and scale
    offsets = user_state.beam_offsets.astype(np.float32).flatten()
    if config.offset_scale_km > 0:
        offsets = offsets / config.offset_scale_km

    # Beam loads
    loads = user_state.beam_loads.astype(np.float32)
    if config.load_normalization == "divide_by_num_users" and num_users > 0:
        loads = loads / num_users

    return np.concatenate([access, snr, offsets, loads])


def state_dim_for(num_beams_total: int) -> int:
    """Compute the flat state dimension for a given topology.

    Layout: access(LK) + snr(LK) + offsets(LK*2) + loads(LK) = 5*LK
    """
    return 5 * num_beams_total


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class EpisodeLog:
    """Per-episode training metrics."""

    episode: int
    epsilon: float
    r1_mean: float
    r2_mean: float
    r3_mean: float
    scalar_reward: float
    total_handovers: int
    replay_size: int
    losses: tuple[float, float, float] = (0.0, 0.0, 0.0)


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

    # -- epsilon schedule (ASSUME-MODQN-REP-004) ----------------------------

    def epsilon(self, episode: int) -> float:
        """Linear epsilon decay from start to end over decay_episodes."""
        cfg = self.config
        if episode >= cfg.epsilon_decay_episodes:
            return cfg.epsilon_end
        frac = episode / cfg.epsilon_decay_episodes
        return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac

    # -- action selection ---------------------------------------------------

    def select_actions(
        self,
        states_encoded: np.ndarray,
        masks: list[ActionMask],
        eps: float,
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
        w = self.config.objective_weights

        with torch.no_grad():
            st = torch.tensor(
                states_encoded, dtype=torch.float32, device=self.device
            )
            q_values = [self.q_nets[i](st).cpu().numpy() for i in range(3)]
            scalarized = (
                w[0] * q_values[0] + w[1] * q_values[1] + w[2] * q_values[2]
            )

        for uid in range(U):
            valid = np.where(masks[uid].mask)[0]
            if len(valid) == 0:
                actions[uid] = 0
                continue

            if self._train_rng.random() < eps:
                actions[uid] = self._train_rng.choice(valid)
            else:
                q_row = scalarized[uid].copy()
                q_row[~masks[uid].mask] = -np.inf
                actions[uid] = int(np.argmax(q_row))

        return actions

    # -- network update -----------------------------------------------------

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

    # -- checkpointing (ASSUME-MODQN-REP-015) ------------------------------

    def checkpoint_rule(self) -> dict[str, Any]:
        """Return the active checkpoint-selection rule for metadata/logging."""
        return {
            "assumption_id": self.config.checkpoint_assumption_id,
            "primary_report": self.config.checkpoint_primary_report,
            "secondary_report": self.config.checkpoint_secondary_report,
            "secondary_implemented": False,
            "secondary_status": (
                "not-yet-implemented: no eval loop / best-eval checkpoint in this hardening pass"
            ),
        }

    def build_checkpoint_payload(
        self,
        *,
        episode: int,
        checkpoint_kind: str,
        logs: list[EpisodeLog] | None = None,
        include_optimizers: bool = True,
    ) -> dict[str, Any]:
        """Build a serializable checkpoint payload."""
        payload: dict[str, Any] = {
            "format_version": 1,
            "checkpoint_kind": checkpoint_kind,
            "episode": episode,
            "train_seed": self.train_seed,
            "env_seed": self.env_seed,
            "mobility_seed": self.mobility_seed,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "trainer_config": asdict(self.config),
            "checkpoint_rule": self.checkpoint_rule(),
            "q_networks": [net.state_dict() for net in self.q_nets],
            "target_networks": [net.state_dict() for net in self.target_nets],
        }
        if include_optimizers:
            payload["optimizers"] = [
                optimizer.state_dict() for optimizer in self.optimizers
            ]
        if logs:
            payload["last_episode_log"] = asdict(logs[-1])
        return payload

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        episode: int,
        checkpoint_kind: str,
        logs: list[EpisodeLog] | None = None,
        include_optimizers: bool = True,
    ) -> Path:
        """Save the current trainer weights/state to disk."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_checkpoint_payload(
            episode=episode,
            checkpoint_kind=checkpoint_kind,
            logs=logs,
            include_optimizers=include_optimizers,
        )
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizers: bool = True,
    ) -> dict[str, Any]:
        """Load trainer weights/state from a checkpoint file."""
        checkpoint_path = Path(path)
        payload = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        for net, state in zip(self.q_nets, payload["q_networks"]):
            net.load_state_dict(state)
        for net, state in zip(self.target_nets, payload["target_networks"]):
            net.load_state_dict(state)
            net.eval()

        optimizer_loaded = False
        if load_optimizers and "optimizers" in payload:
            for optimizer, state in zip(self.optimizers, payload["optimizers"]):
                optimizer.load_state_dict(state)
            optimizer_loaded = True

        self._loaded_checkpoint_metadata = {
            "path": str(checkpoint_path),
            "checkpoint_kind": payload.get("checkpoint_kind"),
            "episode": payload.get("episode"),
            "checkpoint_rule": payload.get("checkpoint_rule", {}),
            "optimizer_loaded": optimizer_loaded,
        }
        return payload

    # -- training loop ------------------------------------------------------

    def train(self, progress_every: int = 100) -> list[EpisodeLog]:
        """Full MODQN training loop.

        Returns a list of per-episode metrics.
        """
        cfg = self.config
        logs: list[EpisodeLog] = []

        for ep in range(cfg.episodes):
            eps = self.epsilon(ep)

            # Reset environment
            states, masks, _diag = self.env.reset(
                self._env_rng, self._mobility_rng
            )

            # Encode states
            encoded = np.array(
                [encode_state(s, self.num_users, cfg) for s in states],
                dtype=np.float32,
            )

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
                next_encoded = np.array(
                    [encode_state(s, self.num_users, cfg) for s in result.user_states],
                    dtype=np.float32,
                )

                # Store transitions per user (ASSUME-MODQN-REP-007: shared policy)
                for uid in range(self.num_users):
                    rw = result.rewards[uid]
                    r3 = np.array(
                        [rw.r1_throughput, rw.r2_handover, rw.r3_load_balance],
                        dtype=np.float32,
                    )
                    self.replay.push(
                        encoded[uid],
                        int(actions[uid]),
                        r3,
                        next_encoded[uid],
                        masks[uid].mask.copy(),
                        result.action_masks[uid].mask.copy(),
                        result.done,
                    )
                    ep_reward += r3
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
            scalar = (
                cfg.objective_weights[0] * avg_reward[0]
                + cfg.objective_weights[1] * avg_reward[1]
                + cfg.objective_weights[2] * avg_reward[2]
            )
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
