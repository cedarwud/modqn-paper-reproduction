"""Public compatibility facade for RA-EE-09 fixed-association RB/bandwidth replay.

Implementation is split across focused private modules:

* ``_ra_ee_09_common`` for constants, dataclasses, and config parsing,
* ``_ra_ee_09_resource`` for allocator and accounting logic,
* ``_ra_ee_09_replay`` for control/candidate replay exports, and
* ``_ra_ee_09_compare`` for Slice 09E matched held-out comparison.

This module keeps the original import surface stable for scripts and tests.
"""

from __future__ import annotations

from ._ra_ee_09_common import (
    DEFAULT_CANDIDATE_OUTPUT_DIR,
    DEFAULT_COMPARISON_OUTPUT_DIR,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    RA_EE_09_ASSUMPTION_KEY,
    RA_EE_09_CANDIDATE,
    RA_EE_09_CANDIDATE_ALLOCATOR,
    RA_EE_09_CANDIDATE_MAX_EQUAL_SHARE_MULTIPLIER,
    RA_EE_09_CANDIDATE_MIN_EQUAL_SHARE_MULTIPLIER,
    RA_EE_09_CONTROL,
    RA_EE_09_EQUAL_SHARE_ALLOCATOR,
    RA_EE_09_GATE_ID,
    RA_EE_09_METHOD_LABEL,
    RA_EE_09_POWER_ALLOCATOR_ID,
    RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC,
    RA_EE_09_RESOURCE_UNIT,
    _candidate_settings_from_control,
    _settings_from_config,
    ra_ee_09_resource_accounting_enabled,
)
from ._ra_ee_09_compare import (
    export_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison,
)
from ._ra_ee_09_replay import (
    export_ra_ee_09_fixed_association_rb_bandwidth_candidate,
    export_ra_ee_09_fixed_association_rb_bandwidth_control,
)
from ._ra_ee_09_resource import (
    _audit_resource_accounting,
    _bounded_qos_slack_resource_share_allocator,
    _compute_user_throughputs_from_resource,
    _equal_share_resource_fractions,
)


__all__ = [
    "DEFAULT_CANDIDATE_OUTPUT_DIR",
    "DEFAULT_COMPARISON_OUTPUT_DIR",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "RA_EE_09_ASSUMPTION_KEY",
    "RA_EE_09_CANDIDATE",
    "RA_EE_09_CANDIDATE_ALLOCATOR",
    "RA_EE_09_CANDIDATE_MAX_EQUAL_SHARE_MULTIPLIER",
    "RA_EE_09_CANDIDATE_MIN_EQUAL_SHARE_MULTIPLIER",
    "RA_EE_09_CONTROL",
    "RA_EE_09_EQUAL_SHARE_ALLOCATOR",
    "RA_EE_09_GATE_ID",
    "RA_EE_09_METHOD_LABEL",
    "RA_EE_09_POWER_ALLOCATOR_ID",
    "RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC",
    "RA_EE_09_RESOURCE_UNIT",
    "_audit_resource_accounting",
    "_bounded_qos_slack_resource_share_allocator",
    "_candidate_settings_from_control",
    "_compute_user_throughputs_from_resource",
    "_equal_share_resource_fractions",
    "_settings_from_config",
    "export_ra_ee_09_fixed_association_rb_bandwidth_candidate",
    "export_ra_ee_09_fixed_association_rb_bandwidth_control",
    "export_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison",
    "ra_ee_09_resource_accounting_enabled",
]
