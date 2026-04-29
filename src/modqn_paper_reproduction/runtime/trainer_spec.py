"""Trainer-adjacent runtime types for Phase 04D."""

from __future__ import annotations

from dataclasses import dataclass


R1_REWARD_MODE_THROUGHPUT = "throughput"
R1_REWARD_MODE_PER_USER_EE_CREDIT = "per-user-ee-credit"
R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT = "per-user-beam-ee-credit"
PHASE_04_B_SINGLE_CATFISH_KIND = "phase-04-b-single-catfish-feasibility"


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
    method_family: str = "MODQN-baseline"
    phase: str = "baseline"
    comparison_role: str = "not-applicable"
    r1_reward_mode: str = R1_REWARD_MODE_THROUGHPUT
    r1_reward_label: str = "throughput"
    r1_reward_provenance: str = "paper-backed MODQN throughput objective"
    reward_calibration_enabled: bool = False
    reward_calibration_mode: str = "raw-unscaled"
    reward_calibration_source: str = "raw-unscaled"
    reward_calibration_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
    reward_normalization_mode: str = "raw-unscaled"
    load_balance_calibration_mode: str = "baseline-paper-weight"

    # -- Phase 04-B Catfish-MODQN opt-in surface ---------------------------
    catfish_enabled: bool = False
    catfish_ablation: str = "none"
    catfish_discount_factor: float = 0.9
    catfish_replay_capacity: int = 50_000
    catfish_quality_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    catfish_quality_threshold_mode: str = "quantile"
    catfish_quality_quantile: float = 0.8
    catfish_quality_fixed_threshold: float | None = None
    catfish_quality_threshold_window: int = 1_000
    catfish_warmup_transitions: int = 256
    catfish_warmup_trigger: str = "main-replay-size"
    catfish_partition_mode: str = "duplicate-high-value"
    catfish_intervention_enabled: bool = False
    catfish_intervention_period_updates: int = 1
    catfish_intervention_catfish_ratio: float = 0.3
    catfish_min_catfish_replay_size: int = 32
    catfish_competitive_shaping_enabled: bool = False

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
        if len(self.catfish_quality_weights) != 3:
            raise ValueError(
                "catfish_quality_weights must have length 3, "
                f"got {self.catfish_quality_weights!r}"
            )
        if self.training_experiment_kind not in {
            "baseline",
            "reward-calibration",
            "phase-03-objective-substitution",
            "phase-03b-objective-geometry",
            "phase-03c-c-power-mdp-pilot",
            PHASE_04_B_SINGLE_CATFISH_KIND,
        }:
            raise ValueError(
                "training_experiment_kind must be one of "
                "{'baseline', 'reward-calibration', "
                "'phase-03-objective-substitution', "
                "'phase-03b-objective-geometry', "
                "'phase-03c-c-power-mdp-pilot', "
                f"{PHASE_04_B_SINGLE_CATFISH_KIND!r}" + "}, "
                f"got {self.training_experiment_kind!r}"
            )
        if self.r1_reward_mode not in {
            R1_REWARD_MODE_THROUGHPUT,
            R1_REWARD_MODE_PER_USER_EE_CREDIT,
            R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT,
        }:
            raise ValueError(
                "r1_reward_mode must be one of "
                f"{{{R1_REWARD_MODE_THROUGHPUT!r}, "
                f"{R1_REWARD_MODE_PER_USER_EE_CREDIT!r}, "
                f"{R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT!r}}}, "
                f"got {self.r1_reward_mode!r}"
            )
        if (
            self.r1_reward_mode
            in {
                R1_REWARD_MODE_PER_USER_EE_CREDIT,
                R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT,
            }
            and self.training_experiment_kind
            not in {
                "phase-03-objective-substitution",
                "phase-03b-objective-geometry",
                "phase-03c-c-power-mdp-pilot",
            }
        ):
            raise ValueError(
                "EE r1 reward modes are only allowed for "
                "training_experiment_kind='phase-03-objective-substitution', "
                "'phase-03b-objective-geometry', or "
                "'phase-03c-c-power-mdp-pilot'."
            )
        if (
            self.training_experiment_kind
            not in {
                "phase-03-objective-substitution",
                "phase-03b-objective-geometry",
                "phase-03c-c-power-mdp-pilot",
            }
            and self.method_family == "EE-MODQN"
        ):
            raise ValueError(
                "method_family='EE-MODQN' requires "
                "a Phase 03/03B/03C-C EE training_experiment_kind."
            )
        if self.method_family == "Catfish-MODQN":
            if self.training_experiment_kind != PHASE_04_B_SINGLE_CATFISH_KIND:
                raise ValueError(
                    "method_family='Catfish-MODQN' requires "
                    f"training_experiment_kind={PHASE_04_B_SINGLE_CATFISH_KIND!r}."
                )
            if not self.catfish_enabled:
                raise ValueError(
                    "method_family='Catfish-MODQN' requires catfish_enabled=True."
                )
        if self.training_experiment_kind == PHASE_04_B_SINGLE_CATFISH_KIND:
            if self.method_family not in {"Catfish-MODQN", "MODQN-control"}:
                raise ValueError(
                    f"{PHASE_04_B_SINGLE_CATFISH_KIND!r} only supports "
                    "method_family='Catfish-MODQN' or 'MODQN-control'."
                )
            if self.r1_reward_mode != R1_REWARD_MODE_THROUGHPUT:
                raise ValueError(
                    "Phase 04-B requires r1_reward_mode='throughput'."
                )
            if self.reward_calibration_enabled:
                raise ValueError("Phase 04-B keeps reward calibration disabled.")
        if (
            self.method_family == "MODQN-control"
            and self.training_experiment_kind == PHASE_04_B_SINGLE_CATFISH_KIND
            and self.catfish_enabled
        ):
            raise ValueError("Phase 04-B MODQN-control must keep Catfish disabled.")
        if self.catfish_enabled:
            if self.catfish_replay_capacity <= 0:
                raise ValueError("catfish_replay_capacity must be > 0.")
            if self.catfish_discount_factor <= 0.0:
                raise ValueError("catfish_discount_factor must be > 0.")
            if self.catfish_quality_threshold_mode not in {"quantile", "fixed"}:
                raise ValueError(
                    "catfish_quality_threshold_mode must be 'quantile' or 'fixed'."
                )
            if self.catfish_quality_threshold_mode == "quantile":
                if not 0.0 < self.catfish_quality_quantile < 1.0:
                    raise ValueError(
                        "catfish_quality_quantile must be between 0 and 1."
                    )
            if (
                self.catfish_quality_threshold_mode == "fixed"
                and self.catfish_quality_fixed_threshold is None
            ):
                raise ValueError(
                    "catfish_quality_fixed_threshold is required for fixed mode."
                )
            if self.catfish_quality_threshold_window <= 0:
                raise ValueError("catfish_quality_threshold_window must be > 0.")
            if self.catfish_warmup_transitions < 0:
                raise ValueError("catfish_warmup_transitions must be >= 0.")
            if self.catfish_warmup_trigger != "main-replay-size":
                raise ValueError(
                    "catfish_warmup_trigger currently supports only "
                    "'main-replay-size'."
                )
            if self.catfish_partition_mode != "duplicate-high-value":
                raise ValueError(
                    "Phase 04-B currently supports only "
                    "catfish_partition_mode='duplicate-high-value'."
                )
            if self.catfish_intervention_period_updates <= 0:
                raise ValueError(
                    "catfish_intervention_period_updates must be > 0."
                )
            if not 0.0 <= self.catfish_intervention_catfish_ratio < 1.0:
                raise ValueError(
                    "catfish_intervention_catfish_ratio must be in [0, 1)."
                )
            if self.catfish_min_catfish_replay_size < 0:
                raise ValueError("catfish_min_catfish_replay_size must be >= 0.")
            if self.catfish_competitive_shaping_enabled:
                raise ValueError(
                    "Phase 04-B primary/initial ablations keep competitive "
                    "shaping disabled."
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
            if self.training_experiment_kind not in {
                "reward-calibration",
                "phase-03b-objective-geometry",
                "phase-03c-c-power-mdp-pilot",
            }:
                raise ValueError(
                    "reward_calibration_enabled requires "
                    "training_experiment_kind='reward-calibration' or "
                    "'phase-03b-objective-geometry' or "
                    "'phase-03c-c-power-mdp-pilot'."
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
