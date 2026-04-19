"""Trainer-adjacent runtime types for Phase 04D."""

from __future__ import annotations

from dataclasses import dataclass


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

    # -- explicit experiment surface ---------------------------------------
    training_experiment_kind: str = "baseline"
    training_experiment_id: str = ""
    reward_calibration_enabled: bool = False
    reward_calibration_mode: str = "raw-unscaled"
    reward_calibration_source: str = "raw-unscaled"
    reward_calibration_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.objective_weights) != 3:
            raise ValueError(
                "objective_weights must have length 3, "
                f"got {self.objective_weights!r}"
            )
        if len(self.reward_calibration_scales) != 3:
            raise ValueError(
                "reward_calibration_scales must have length 3, "
                f"got {self.reward_calibration_scales!r}"
            )
        if self.reward_calibration_mode not in {
            "raw-unscaled",
            "divide-by-fixed-scales",
        }:
            raise ValueError(
                "reward_calibration_mode must be one of "
                "{'raw-unscaled', 'divide-by-fixed-scales'}, "
                f"got {self.reward_calibration_mode!r}"
            )
        if self.reward_calibration_enabled:
            if self.training_experiment_kind != "reward-calibration":
                raise ValueError(
                    "reward_calibration_enabled requires "
                    "training_experiment_kind='reward-calibration'."
                )
            if self.reward_calibration_mode != "divide-by-fixed-scales":
                raise ValueError(
                    "reward_calibration_enabled currently requires "
                    "reward_calibration_mode='divide-by-fixed-scales'."
                )
            if any(scale <= 0 for scale in self.reward_calibration_scales):
                raise ValueError(
                    "reward_calibration_scales must all be > 0 when calibration is enabled, "
                    f"got {self.reward_calibration_scales!r}"
                )


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


@dataclass(frozen=True)
class EvalSummary:
    """Aggregated greedy-policy evaluation over the configured eval seeds."""

    episode: int
    evaluation_every_episodes: int
    eval_seeds: tuple[int, ...]
    mean_scalar_reward: float
    std_scalar_reward: float
    mean_r1: float
    std_r1: float
    mean_r2: float
    std_r2: float
    mean_r3: float
    std_r3: float
    mean_total_handovers: float
    std_total_handovers: float
