"""Trainer-adjacent runtime types for Phase 04D."""

from __future__ import annotations

from dataclasses import dataclass


R1_REWARD_MODE_THROUGHPUT = "throughput"
R1_REWARD_MODE_PER_USER_EE_CREDIT = "per-user-ee-credit"
R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT = "per-user-beam-ee-credit"
R1_REWARD_MODE_HOBS_ACTIVE_TX_EE = "hobs-active-tx-ee"
PHASE_04_B_SINGLE_CATFISH_KIND = "phase-04-b-single-catfish-feasibility"
HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND = "hobs-active-tx-ee-modqn-feasibility"
PHASE_05_B_MULTI_CATFISH_KIND = "phase-05-b-multi-catfish-bounded-pilot"
PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND = (
    "phase-07-b-single-catfish-intervention-utility"
)
PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND = (
    "phase-07-d-r2-guarded-single-catfish-robustness"
)


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

    # -- Phase 05-B Multi-Catfish opt-in surface --------------------------
    catfish_phase05b_variant: str = "not-applicable"
    catfish_objective_admission_rule: str = "disabled"
    catfish_objective_tie_policy: str = "not-applicable"
    catfish_source_ratios: tuple[float, float, float] = (0.0, 0.0, 0.0)
    catfish_total_intervention_ratio: float = 0.0
    catfish_specialist_mode: str = "disabled"
    catfish_r1_threshold: float = 131.66555786132812
    catfish_r1_r3_guardrail: float = -2.155817623138428
    catfish_r2_best_value: float = 0.0
    catfish_r3_threshold: float = -1.4955613708496094
    catfish_r3_r1_guardrail: float = 96.77507400512695
    catfish_random_buffer_admission_probability: float = 0.15
    catfish_phase05b_seed_triplets: tuple[tuple[int, int, int], ...] = ()

    # -- Phase 07-B single-Catfish recovery opt-in surface ----------------
    catfish_phase07b_variant: str = "not-applicable"
    catfish_intervention_source_mode: str = "catfish-replay"
    catfish_challenger_enabled: bool = True
    catfish_lineage_tracking_enabled: bool = False
    catfish_phase07b_seed_triplets: tuple[tuple[int, int, int], ...] = ()

    # -- Phase 07-D r2-guarded single-Catfish robustness surface ----------
    catfish_phase07d_variant: str = "not-applicable"
    catfish_r2_guard_enabled: bool = False
    catfish_r2_admission_guard_enabled: bool = False
    catfish_r2_intervention_guard_enabled: bool = False
    catfish_r2_strict_no_handover_sample_guard: bool = False
    catfish_r2_guard_max_batch_attempts: int = 16
    catfish_handover_spike_guard_enabled: bool = False
    catfish_handover_spike_window: int = 5
    catfish_handover_spike_margin: float = 0.10
    catfish_handover_spike_min_windows: int = 3
    catfish_phase07d_seed_triplets: tuple[tuple[int, int, int], ...] = ()

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
            PHASE_05_B_MULTI_CATFISH_KIND,
            PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND,
            PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND,
            HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND,
        }:
            raise ValueError(
                "training_experiment_kind must be one of "
                "{'baseline', 'reward-calibration', "
                "'phase-03-objective-substitution', "
                "'phase-03b-objective-geometry', "
                "'phase-03c-c-power-mdp-pilot', "
                f"{PHASE_04_B_SINGLE_CATFISH_KIND!r}, "
                f"{PHASE_05_B_MULTI_CATFISH_KIND!r}, "
                f"{PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND!r}, "
                f"{PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND!r}, "
                f"{HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND!r}" + "}, "
                f"got {self.training_experiment_kind!r}"
            )
        if self.r1_reward_mode not in {
            R1_REWARD_MODE_THROUGHPUT,
            R1_REWARD_MODE_PER_USER_EE_CREDIT,
            R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT,
            R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
        }:
            raise ValueError(
                "r1_reward_mode must be one of "
                f"{{{R1_REWARD_MODE_THROUGHPUT!r}, "
                f"{R1_REWARD_MODE_PER_USER_EE_CREDIT!r}, "
                f"{R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT!r}, "
                f"{R1_REWARD_MODE_HOBS_ACTIVE_TX_EE!r}}}, "
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
            self.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
            and self.training_experiment_kind
            != HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND
        ):
            raise ValueError(
                f"r1_reward_mode={R1_REWARD_MODE_HOBS_ACTIVE_TX_EE!r} requires "
                f"training_experiment_kind={HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND!r}."
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
            if self.training_experiment_kind not in {
                PHASE_04_B_SINGLE_CATFISH_KIND,
                PHASE_05_B_MULTI_CATFISH_KIND,
                PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND,
                PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND,
            }:
                raise ValueError(
                    "method_family='Catfish-MODQN' requires "
                    f"training_experiment_kind={PHASE_04_B_SINGLE_CATFISH_KIND!r} "
                    f"or {PHASE_05_B_MULTI_CATFISH_KIND!r} "
                    f"or {PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND!r} "
                    f"or {PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND!r}."
                )
            if not self.catfish_enabled:
                raise ValueError(
                    "method_family='Catfish-MODQN' requires catfish_enabled=True."
                )
        if self.method_family == "Multi-Catfish-MODQN":
            if self.training_experiment_kind != PHASE_05_B_MULTI_CATFISH_KIND:
                raise ValueError(
                    "method_family='Multi-Catfish-MODQN' requires "
                    f"training_experiment_kind={PHASE_05_B_MULTI_CATFISH_KIND!r}."
                )
            if not self.catfish_enabled:
                raise ValueError(
                    "method_family='Multi-Catfish-MODQN' requires "
                    "catfish_enabled=True."
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
        if self.training_experiment_kind == PHASE_05_B_MULTI_CATFISH_KIND:
            if self.method_family not in {
                "MODQN-control",
                "Catfish-MODQN",
                "Multi-Catfish-MODQN",
            }:
                raise ValueError(
                    "Phase 05-B supports only method_family='MODQN-control', "
                    "'Catfish-MODQN', or 'Multi-Catfish-MODQN'."
                )
            if self.r1_reward_mode != R1_REWARD_MODE_THROUGHPUT:
                raise ValueError("Phase 05-B requires r1_reward_mode='throughput'.")
            if self.reward_calibration_enabled:
                raise ValueError("Phase 05-B keeps reward calibration disabled.")
            if self.catfish_competitive_shaping_enabled:
                raise ValueError("Phase 05-B primary/control configs keep shaping off.")
            if self.method_family == "MODQN-control":
                if self.catfish_enabled:
                    raise ValueError("Phase 05-B MODQN-control keeps Catfish disabled.")
                if self.catfish_total_intervention_ratio != 0.0:
                    raise ValueError(
                        "Phase 05-B MODQN-control requires total Catfish ratio 0.0."
                    )
            else:
                if not self.catfish_enabled:
                    raise ValueError("Phase 05-B Catfish variants require enabled=true.")
                if abs(self.catfish_total_intervention_ratio - 0.30) > 1e-9:
                    raise ValueError(
                        "Phase 05-B Catfish variants require total Catfish ratio 0.30."
                    )
                if self.catfish_phase05b_variant == "single-catfish-equal-budget":
                    if self.method_family != "Catfish-MODQN":
                        raise ValueError(
                            "single-catfish-equal-budget requires Catfish-MODQN."
                        )
                    if self.catfish_objective_admission_rule != "scalar-phase04":
                        raise ValueError(
                            "single-catfish-equal-budget uses scalar Phase 04 replay."
                        )
                else:
                    if self.method_family != "Multi-Catfish-MODQN":
                        raise ValueError(
                            "Phase 05-B objective-buffer variants require "
                            "Multi-Catfish-MODQN."
                        )
                    if abs(sum(self.catfish_source_ratios) - 0.30) > 1e-9:
                        raise ValueError(
                            "Phase 05-B objective-buffer ratios must sum to 0.30."
                        )
                    if any(abs(value - 0.10) > 1e-9 for value in self.catfish_source_ratios):
                        raise ValueError(
                            "Phase 05-B first bounded pilot requires 0.10 per "
                            "r1/r2/r3 specialist source."
                        )
                    if self.catfish_objective_admission_rule not in {
                        "guarded-residual-objective-admission",
                        "random-uniform-buffer-control",
                    }:
                        raise ValueError(
                            "Phase 05-B objective-buffer variants require "
                            "guarded-residual-objective-admission or "
                            "random-uniform-buffer-control."
                        )
                    if (
                        self.catfish_objective_admission_rule
                        == "guarded-residual-objective-admission"
                        and self.catfish_objective_tie_policy
                        != "complete-threshold-ties"
                    ):
                        raise ValueError(
                            "Phase 05-B guarded residual admission requires "
                            "complete-threshold-ties."
                        )
                    if self.catfish_specialist_mode not in {
                        "three-specialists",
                        "single-learner",
                    }:
                        raise ValueError(
                            "Phase 05-B objective-buffer variants require "
                            "catfish_specialist_mode='three-specialists' or "
                            "'single-learner'."
                        )
                if len(self.catfish_phase05b_seed_triplets) < 3:
                    raise ValueError(
                        "Phase 05-B requires at least three matched seed triplets."
                    )
            if len(self.catfish_phase05b_seed_triplets) < 3:
                raise ValueError(
                    "Phase 05-B requires at least three matched seed triplets."
                )
        if self.training_experiment_kind == PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND:
            if self.method_family not in {"MODQN-control", "Catfish-MODQN"}:
                raise ValueError(
                    "Phase 07-B supports only method_family='MODQN-control' "
                    "or 'Catfish-MODQN'."
                )
            if self.r1_reward_mode != R1_REWARD_MODE_THROUGHPUT:
                raise ValueError("Phase 07-B requires r1_reward_mode='throughput'.")
            if self.reward_calibration_enabled:
                raise ValueError("Phase 07-B keeps reward calibration disabled.")
            if self.catfish_competitive_shaping_enabled:
                raise ValueError("Phase 07-B primary configs keep shaping off.")
            if len(self.catfish_phase07b_seed_triplets) < 3:
                raise ValueError(
                    "Phase 07-B requires at least three matched seed triplets."
                )
            allowed_phase07b_variants = {
                "modqn-control",
                "single-catfish-primary-shaping-off",
                "no-intervention",
                "random-equal-budget-injection",
                "replay-only-single-learner",
                "no-asymmetric-gamma",
            }
            if self.catfish_phase07b_variant not in allowed_phase07b_variants:
                raise ValueError(
                    "Unsupported Phase 07-B variant "
                    f"{self.catfish_phase07b_variant!r}."
                )
            if self.catfish_intervention_source_mode not in {
                "disabled",
                "catfish-replay",
                "random-main-replay",
            }:
                raise ValueError(
                    "catfish_intervention_source_mode must be one of "
                    "'disabled', 'catfish-replay', or 'random-main-replay'."
                )
            if self.method_family == "MODQN-control":
                if self.catfish_phase07b_variant != "modqn-control":
                    raise ValueError(
                        "Phase 07-B MODQN-control requires variant='modqn-control'."
                    )
                if self.catfish_enabled:
                    raise ValueError("Phase 07-B MODQN-control keeps Catfish disabled.")
                if self.catfish_intervention_enabled:
                    raise ValueError(
                        "Phase 07-B MODQN-control keeps intervention disabled."
                    )
                if self.catfish_intervention_catfish_ratio != 0.0:
                    raise ValueError(
                        "Phase 07-B MODQN-control requires Catfish ratio 0.0."
                    )
                if self.catfish_intervention_source_mode != "disabled":
                    raise ValueError(
                        "Phase 07-B MODQN-control requires source_mode='disabled'."
                    )
            else:
                if not self.catfish_enabled:
                    raise ValueError(
                        "Phase 07-B Catfish comparator configs require enabled=true."
                    )
                if self.catfish_phase07b_variant == "no-intervention":
                    if self.catfish_intervention_enabled:
                        raise ValueError(
                            "Phase 07-B no-intervention keeps intervention disabled."
                        )
                    if self.catfish_intervention_catfish_ratio != 0.0:
                        raise ValueError(
                            "Phase 07-B no-intervention requires Catfish ratio 0.0."
                        )
                    if self.catfish_intervention_source_mode != "disabled":
                        raise ValueError(
                            "Phase 07-B no-intervention requires "
                            "source_mode='disabled'."
                        )
                else:
                    if not self.catfish_intervention_enabled:
                        raise ValueError(
                            "Phase 07-B Catfish utility comparators require "
                            "intervention enabled unless variant is no-intervention."
                        )
                    if abs(self.catfish_intervention_catfish_ratio - 0.30) > 1e-9:
                        raise ValueError(
                            "Phase 07-B intervention comparators require ratio 0.30."
                        )
                if self.catfish_phase07b_variant == "random-equal-budget-injection":
                    if self.catfish_intervention_source_mode != "random-main-replay":
                        raise ValueError(
                            "Phase 07-B random/equal-budget comparator requires "
                            "source_mode='random-main-replay'."
                        )
                    if self.catfish_challenger_enabled:
                        raise ValueError(
                            "Phase 07-B random/equal-budget comparator does not "
                            "instantiate a challenger learner."
                        )
                if self.catfish_phase07b_variant in {
                    "single-catfish-primary-shaping-off",
                    "replay-only-single-learner",
                    "no-asymmetric-gamma",
                }:
                    if self.catfish_intervention_source_mode != "catfish-replay":
                        raise ValueError(
                            "Phase 07-B Catfish replay variants require "
                            "source_mode='catfish-replay'."
                        )
                if self.catfish_phase07b_variant == "replay-only-single-learner":
                    if self.catfish_challenger_enabled:
                        raise ValueError(
                            "Phase 07-B replay-only single learner must not "
                            "instantiate a challenger learner."
                        )
                if self.catfish_phase07b_variant in {
                    "single-catfish-primary-shaping-off",
                    "no-intervention",
                    "no-asymmetric-gamma",
                } and not self.catfish_challenger_enabled:
                    raise ValueError(
                        "Phase 07-B primary/no-intervention/no-asymmetric-gamma "
                        "comparators require the single Catfish challenger."
                    )
                if self.catfish_phase07b_variant == "no-asymmetric-gamma":
                    if abs(self.catfish_discount_factor - self.discount_factor) > 1e-9:
                        raise ValueError(
                            "Phase 07-B no-asymmetric-gamma requires equal main "
                            "and Catfish discount factors."
                        )
        if self.training_experiment_kind == PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND:
            if self.method_family not in {"MODQN-control", "Catfish-MODQN"}:
                raise ValueError(
                    "Phase 07-D supports only method_family='MODQN-control' "
                    "or 'Catfish-MODQN'."
                )
            if self.r1_reward_mode != R1_REWARD_MODE_THROUGHPUT:
                raise ValueError("Phase 07-D requires r1_reward_mode='throughput'.")
            if self.reward_calibration_enabled:
                raise ValueError("Phase 07-D keeps reward calibration disabled.")
            if self.catfish_competitive_shaping_enabled:
                raise ValueError("Phase 07-D primary/configs keep shaping off.")
            if len(self.catfish_phase07d_seed_triplets) < 3:
                raise ValueError(
                    "Phase 07-D requires at least three matched seed triplets."
                )

            allowed_phase07d_variants = {
                "modqn-control",
                "r2-guarded-primary-shaping-off",
                "no-intervention",
                "random-equal-budget-injection",
                "replay-only-single-learner",
                "no-asymmetric-gamma",
                "admission-only-guard",
                "intervention-only-guard",
                "full-admission-intervention-guard",
                "strict-no-handover-sample-guard",
            }
            if self.catfish_phase07d_variant not in allowed_phase07d_variants:
                raise ValueError(
                    "Unsupported Phase 07-D variant "
                    f"{self.catfish_phase07d_variant!r}."
                )
            if self.catfish_intervention_source_mode not in {
                "disabled",
                "catfish-replay",
                "random-main-replay",
            }:
                raise ValueError(
                    "catfish_intervention_source_mode must be one of "
                    "'disabled', 'catfish-replay', or 'random-main-replay'."
                )

            if self.method_family == "MODQN-control":
                if self.catfish_phase07d_variant != "modqn-control":
                    raise ValueError(
                        "Phase 07-D MODQN-control requires variant='modqn-control'."
                    )
                if self.catfish_enabled:
                    raise ValueError("Phase 07-D MODQN-control keeps Catfish disabled.")
                if self.catfish_intervention_enabled:
                    raise ValueError(
                        "Phase 07-D MODQN-control keeps intervention disabled."
                    )
                if self.catfish_intervention_catfish_ratio != 0.0:
                    raise ValueError(
                        "Phase 07-D MODQN-control requires Catfish ratio 0.0."
                    )
                if self.catfish_intervention_source_mode != "disabled":
                    raise ValueError(
                        "Phase 07-D MODQN-control requires source_mode='disabled'."
                    )
                if self.catfish_r2_guard_enabled:
                    raise ValueError("Phase 07-D MODQN-control must not enable guards.")
            else:
                if not self.catfish_enabled:
                    raise ValueError(
                        "Phase 07-D Catfish comparator configs require enabled=true."
                    )
                if self.catfish_phase07d_variant == "no-intervention":
                    if self.catfish_intervention_enabled:
                        raise ValueError(
                            "Phase 07-D no-intervention keeps intervention disabled."
                        )
                    if self.catfish_intervention_catfish_ratio != 0.0:
                        raise ValueError(
                            "Phase 07-D no-intervention requires Catfish ratio 0.0."
                        )
                    if self.catfish_intervention_source_mode != "disabled":
                        raise ValueError(
                            "Phase 07-D no-intervention requires "
                            "source_mode='disabled'."
                        )
                    if self.catfish_r2_guard_enabled:
                        raise ValueError(
                            "Phase 07-D no-intervention must not enable guards."
                        )
                else:
                    if not self.catfish_intervention_enabled:
                        raise ValueError(
                            "Phase 07-D Catfish comparators require intervention "
                            "enabled unless variant is no-intervention."
                        )
                    if abs(self.catfish_intervention_catfish_ratio - 0.30) > 1e-9:
                        raise ValueError(
                            "Phase 07-D intervention comparators require ratio 0.30."
                        )

                if self.catfish_phase07d_variant == "random-equal-budget-injection":
                    if self.catfish_intervention_source_mode != "random-main-replay":
                        raise ValueError(
                            "Phase 07-D random/equal-budget comparator requires "
                            "source_mode='random-main-replay'."
                        )
                    if self.catfish_challenger_enabled:
                        raise ValueError(
                            "Phase 07-D random/equal-budget comparator does not "
                            "instantiate a challenger learner."
                        )
                    if self.catfish_r2_guard_enabled:
                        raise ValueError(
                            "Phase 07-D random/equal-budget comparator must not "
                            "enable Catfish r2 guards."
                        )

                catfish_replay_variants = {
                    "r2-guarded-primary-shaping-off",
                    "replay-only-single-learner",
                    "no-asymmetric-gamma",
                    "admission-only-guard",
                    "intervention-only-guard",
                    "full-admission-intervention-guard",
                    "strict-no-handover-sample-guard",
                }
                if self.catfish_phase07d_variant in catfish_replay_variants:
                    if self.catfish_intervention_source_mode != "catfish-replay":
                        raise ValueError(
                            "Phase 07-D Catfish replay variants require "
                            "source_mode='catfish-replay'."
                        )
                    if not self.catfish_r2_guard_enabled:
                        raise ValueError(
                            "Phase 07-D guarded Catfish replay variants require "
                            "catfish_r2_guard_enabled=True."
                        )

                admission_expected = self.catfish_phase07d_variant in {
                    "r2-guarded-primary-shaping-off",
                    "replay-only-single-learner",
                    "no-asymmetric-gamma",
                    "admission-only-guard",
                    "full-admission-intervention-guard",
                }
                intervention_expected = self.catfish_phase07d_variant in {
                    "r2-guarded-primary-shaping-off",
                    "replay-only-single-learner",
                    "no-asymmetric-gamma",
                    "intervention-only-guard",
                    "full-admission-intervention-guard",
                    "strict-no-handover-sample-guard",
                }
                strict_expected = (
                    self.catfish_phase07d_variant
                    == "strict-no-handover-sample-guard"
                )
                if (
                    self.catfish_phase07d_variant in catfish_replay_variants
                    and self.catfish_r2_admission_guard_enabled != admission_expected
                ):
                    raise ValueError(
                        "Phase 07-D variant has mismatched admission guard setting."
                    )
                if (
                    self.catfish_phase07d_variant in catfish_replay_variants
                    and self.catfish_r2_intervention_guard_enabled
                    != intervention_expected
                ):
                    raise ValueError(
                        "Phase 07-D variant has mismatched intervention guard setting."
                    )
                if (
                    self.catfish_phase07d_variant in catfish_replay_variants
                    and self.catfish_r2_strict_no_handover_sample_guard
                    != strict_expected
                ):
                    raise ValueError(
                        "Phase 07-D strict no-handover sample guard is only "
                        "allowed for the strict diagnostic variant."
                    )

                if self.catfish_phase07d_variant == "replay-only-single-learner":
                    if self.catfish_challenger_enabled:
                        raise ValueError(
                            "Phase 07-D replay-only single learner must not "
                            "instantiate a challenger learner."
                        )
                if self.catfish_phase07d_variant in {
                    "r2-guarded-primary-shaping-off",
                    "no-intervention",
                    "no-asymmetric-gamma",
                    "admission-only-guard",
                    "intervention-only-guard",
                    "full-admission-intervention-guard",
                    "strict-no-handover-sample-guard",
                } and not self.catfish_challenger_enabled:
                    raise ValueError(
                        "Phase 07-D primary/guard comparators require the single "
                        "Catfish challenger."
                    )
                if self.catfish_phase07d_variant == "no-asymmetric-gamma":
                    if abs(self.catfish_discount_factor - self.discount_factor) > 1e-9:
                        raise ValueError(
                            "Phase 07-D no-asymmetric-gamma requires equal main "
                            "and Catfish discount factors."
                        )

            if self.catfish_r2_guard_enabled:
                if self.catfish_r2_guard_max_batch_attempts <= 0:
                    raise ValueError(
                        "catfish_r2_guard_max_batch_attempts must be > 0."
                    )
                if self.catfish_handover_spike_window <= 0:
                    raise ValueError("catfish_handover_spike_window must be > 0.")
                if self.catfish_handover_spike_margin < 0.0:
                    raise ValueError("catfish_handover_spike_margin must be >= 0.")
                if self.catfish_handover_spike_min_windows < 0:
                    raise ValueError(
                        "catfish_handover_spike_min_windows must be >= 0."
                    )
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
            allowed_partition_modes = {"duplicate-high-value"}
            if self.training_experiment_kind == PHASE_05_B_MULTI_CATFISH_KIND:
                allowed_partition_modes |= {
                    "guarded-residual-objective-admission",
                    "random-uniform-buffer-control",
                }
            if self.catfish_partition_mode not in allowed_partition_modes:
                raise ValueError(
                    "Unsupported catfish_partition_mode "
                    f"{self.catfish_partition_mode!r} for "
                    f"{self.training_experiment_kind!r}."
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
                    "Catfish primary/initial ablations keep competitive "
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
