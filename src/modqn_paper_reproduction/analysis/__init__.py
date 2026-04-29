"""Analysis and plotting helpers for landed sweep/export surfaces."""

from .atmospheric_sign_counterfactual import (
    export_atmospheric_sign_counterfactual_eval,
)
from .beam_counterfactual import export_counterfactual_eligibility_eval
from .beam_semantics import export_beam_semantics_audit
from .ee_denominator import export_ee_denominator_audit
from .figures import export_figure_sweep_results
from .phase03_ee_modqn import export_phase03_paired_validation
from .phase03c_b_power_mdp_audit import export_phase03c_b_power_mdp_audit
from .phase03c_c_power_mdp_pilot import (
    export_phase03c_c_power_mdp_paired_validation,
)
from .ra_ee_02_oracle_power_allocation import (
    export_ra_ee_02_oracle_power_allocation_audit,
)
from .ra_ee_04_bounded_power_allocator import (
    export_ra_ee_04_bounded_power_allocator_pilot,
)
from .ra_ee_06_association_counterfactual_oracle import (
    export_ra_ee_06_association_counterfactual_oracle,
)
from .reward_geometry import (
    build_reward_geometry_scale_table,
    build_reward_geometry_table_ii_frames,
    collect_reward_diagnostics,
    export_reward_geometry_analysis,
)
from .table_ii import (
    build_table_ii_analysis_frames,
    export_table_ii_results,
    write_table_ii_analysis_markdown,
)
from .training_log import (
    export_training_log_artifacts,
    summarize_training_log,
    window_means,
)

__all__ = [
    "build_reward_geometry_scale_table",
    "build_reward_geometry_table_ii_frames",
    "build_table_ii_analysis_frames",
    "collect_reward_diagnostics",
    "export_atmospheric_sign_counterfactual_eval",
    "export_counterfactual_eligibility_eval",
    "export_beam_semantics_audit",
    "export_ee_denominator_audit",
    "export_figure_sweep_results",
    "export_phase03_paired_validation",
    "export_phase03c_b_power_mdp_audit",
    "export_phase03c_c_power_mdp_paired_validation",
    "export_ra_ee_02_oracle_power_allocation_audit",
    "export_ra_ee_04_bounded_power_allocator_pilot",
    "export_ra_ee_06_association_counterfactual_oracle",
    "export_reward_geometry_analysis",
    "export_table_ii_results",
    "export_training_log_artifacts",
    "summarize_training_log",
    "window_means",
    "write_table_ii_analysis_markdown",
]
