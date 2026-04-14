from __future__ import annotations

from dataclasses import dataclass

from .contracts import PaperBaselineParameters


@dataclass(frozen=True)
class ReproductionAssumptions:
    orbit_layout: str = "ASSUME-MODQN-REP-001"
    beam_geometry: str = "ASSUME-MODQN-REP-002"
    handover_cost_values: str = "ASSUME-MODQN-REP-003"
    epsilon_schedule: str = "ASSUME-MODQN-REP-004"
    target_update_cadence: str = "ASSUME-MODQN-REP-005"
    replay_capacity: str = "ASSUME-MODQN-REP-006"
    policy_sharing_mode: str = "ASSUME-MODQN-REP-007"
    noise_semantics: str = "ASSUME-MODQN-REP-008"
    atmospheric_formula_sign: str = "ASSUME-MODQN-REP-009"
    evaluation_aggregation: str = "ASSUME-MODQN-REP-010"
    topology_handling_strategy: str = "ASSUME-MODQN-REP-011"
    action_masking_semantics: str = "ASSUME-MODQN-REP-012"
    state_encoding_and_normalization: str = "ASSUME-MODQN-REP-013"
    trace_source_and_stk_sampling: str = "ASSUME-MODQN-REP-014"
    checkpoint_selection_rule: str = "ASSUME-MODQN-REP-015"
    figure_discrete_point_set: str = "ASSUME-MODQN-REP-016"
    comparator_training_protocol: str = "ASSUME-MODQN-REP-017"
    seed_and_rng_policy: str = "ASSUME-MODQN-REP-018"
    r3_gap_beam_scope: str = "ASSUME-MODQN-REP-019"
    user_heading_stride: str = "ASSUME-MODQN-REP-020"
    user_scatter_radius: str = "ASSUME-MODQN-REP-021"
    user_area_geometry: str = "ASSUME-MODQN-REP-022"
    user_mobility_model: str = "ASSUME-MODQN-REP-023"


DEFAULT_PAPER_PARAMETERS = PaperBaselineParameters()
DEFAULT_ASSUMPTIONS = ReproductionAssumptions()
