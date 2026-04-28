"""Config loader for MODQN paper reproduction YAML.

Training entrypoints must load an executable resolved-run config and
must reject the paper-envelope config. Critical runtime assumptions for
ASSUME-MODQN-REP-015/019/020/021 are validated here so that the trainer
cannot silently fall back to module constants.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from .env.orbit import OrbitConfig
from .env.beam import BeamConfig
from .env.channel import AtmosphericSignMode, ChannelConfig
from .env.step import PowerSurfaceConfig, StepConfig, StepEnvironment
from .runtime.trainer_spec import TrainerConfig


class ConfigValidationError(ValueError):
    """Raised when a training config violates the authority contract."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    inherits_from = cfg.get("inherits_from")
    if inherits_from:
        parent_path = Path(inherits_from)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        parent_cfg = load_yaml(parent_path)
        cfg = _merge_dicts(parent_cfg, cfg)

    return cfg


def load_training_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate an executable resolved-run training config."""
    cfg = load_yaml(path)
    require_training_config(cfg, config_path=path)
    return cfg


def require_training_config(
    cfg: dict[str, Any],
    config_path: str | Path | None = None,
) -> None:
    """Reject non-executable training configs.

    Paper-envelope configs are authority/reference surfaces only and may
    not be used as training input. The executable training surface must
    be a resolved-run config.
    """
    role = str(cfg.get("config_role", "")).strip()
    location = str(config_path) if config_path is not None else "<config>"

    if role == "paper-envelope":
        raise ConfigValidationError(
            f"Training input {location!r} has config_role='paper-envelope'. "
            "Use a resolved-run config instead; the paper-envelope config is "
            "authority-only and cannot start training."
        )
    if not role.startswith("resolved-run"):
        raise ConfigValidationError(
            f"Training input {location!r} must be a resolved-run config, "
            f"got config_role={role!r}."
        )
    if not isinstance(cfg.get("resolved_assumptions"), dict):
        raise ConfigValidationError(
            f"Training input {location!r} is missing 'resolved_assumptions'."
        )


def _resolved_assumption_block(
    cfg: dict[str, Any],
    name: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    resolved = cfg.get("resolved_assumptions", {})
    block = resolved.get(name, {})
    if isinstance(block, dict):
        return block
    if required:
        raise ConfigValidationError(
            f"Resolved config is missing assumption block '{name}'."
        )
    return {}


def _resolved_assumption_value(
    cfg: dict[str, Any],
    name: str,
    *,
    required: bool = False,
) -> Any:
    block = _resolved_assumption_block(cfg, name, required=required)
    if "value" in block:
        return block["value"]
    if required:
        raise ConfigValidationError(
            f"Resolved config assumption '{name}' is missing its 'value' field."
        )
    return {}


def _required_mapping_field(
    mapping: dict[str, Any],
    key: str,
    *,
    context: str,
) -> Any:
    if key not in mapping:
        raise ConfigValidationError(f"{context} is missing required field '{key}'.")
    return mapping[key]


def _merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge config dictionaries with child values overriding parent."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _training_experiment_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the optional training-experiment block."""
    block = cfg.get("training_experiment", {})
    if block in ({}, None):
        return {}
    if not isinstance(block, dict):
        raise ConfigValidationError("'training_experiment' must be a mapping when present.")
    return block


def build_step_config(cfg: dict[str, Any]) -> StepConfig:
    """Build StepConfig from the resolved-run YAML."""
    base = cfg.get("baseline", cfg)
    mask_val = _resolved_assumption_value(cfg, "action_masking_semantics")
    mask_val = mask_val if isinstance(mask_val, dict) else {}

    phi1 = 0.5
    phi2 = 1.0
    ho_val = _resolved_assumption_value(cfg, "handover_cost_values")
    if isinstance(ho_val, dict):
        phi1 = ho_val.get("phi1", phi1)
        phi2 = ho_val.get("phi2", phi2)

    gp_val = _resolved_assumption_value(cfg, "ground_point")
    gp_val = gp_val if isinstance(gp_val, dict) else {}
    lat = gp_val.get("lat_deg", 0.0)
    lon = gp_val.get("lon_deg", 0.0)

    r3_gap_val = _resolved_assumption_value(
        cfg, "r3_gap_beam_scope", required=True
    )
    if not isinstance(r3_gap_val, dict):
        raise ConfigValidationError(
            "Resolved assumption 'r3_gap_beam_scope' must be a mapping."
        )

    heading_val = _resolved_assumption_value(
        cfg, "user_heading_stride", required=True
    )
    if not isinstance(heading_val, dict):
        raise ConfigValidationError(
            "Resolved assumption 'user_heading_stride' must be a mapping."
        )

    scatter_val = _resolved_assumption_value(
        cfg, "user_scatter_radius", required=True
    )
    if not isinstance(scatter_val, dict):
        raise ConfigValidationError(
            "Resolved assumption 'user_scatter_radius' must be a mapping."
        )
    area_val = _resolved_assumption_value(cfg, "user_area_geometry")
    area_val = area_val if isinstance(area_val, dict) else {}
    mobility_val = _resolved_assumption_value(cfg, "user_mobility_model")
    mobility_val = mobility_val if isinstance(mobility_val, dict) else {}

    scatter_distribution = str(
        area_val.get(
            "distribution",
            scatter_val.get("distribution", "uniform-circular"),
        )
    )
    area_width_km = float(area_val.get("width_km", 0.0) or 0.0)
    area_height_km = float(area_val.get("height_km", 0.0) or 0.0)
    mobility_model = str(
        mobility_val.get("model", "deterministic-heading")
    )
    random_wandering_max_turn_rad = float(
        mobility_val.get("max_turn_rad", math.pi / 4.0)
    )
    eligibility_mode_raw = str(
        mask_val.get(
            "eligibility_mode",
            mask_val.get("eligibility_rule", "satellite-visible-above-0deg-horizon"),
        )
    )
    if eligibility_mode_raw == "satellite-visible-above-0deg-horizon":
        action_mask_eligibility_mode = "satellite-visible-all-beams"
    elif eligibility_mode_raw in {
        "satellite-visible-all-beams",
        "nearest-beam-per-visible-satellite",
    }:
        action_mask_eligibility_mode = eligibility_mode_raw
    else:
        raise ConfigValidationError(
            "Resolved assumption 'action_masking_semantics' has unsupported "
            f"eligibility mode/rule {eligibility_mode_raw!r}."
        )

    return StepConfig(
        num_users=base.get("users", 100),
        slot_duration_s=base.get("slot_duration_s", 1.0),
        episode_duration_s=base.get("episode_duration_s", 10.0),
        user_speed_kmh=base.get("user_speed_kmh", 30.0),
        phi1=phi1,
        phi2=phi2,
        user_lat_deg=lat,
        user_lon_deg=lon,
        r3_gap_scope=_required_mapping_field(
            r3_gap_val,
            "scope",
            context="Resolved assumption 'r3_gap_beam_scope'",
        ),
        r3_empty_beam_throughput=float(
            r3_gap_val.get("empty_beam_throughput", 0.0)
        ),
        action_mask_eligibility_mode=action_mask_eligibility_mode,
        user_heading_stride_rad=float(
            _required_mapping_field(
                heading_val,
                "stride_rad",
                context="Resolved assumption 'user_heading_stride'",
            )
        ),
        user_scatter_radius_km=float(
            _required_mapping_field(
                scatter_val,
                "radius_km",
                context="Resolved assumption 'user_scatter_radius'",
            )
        ),
        user_scatter_distribution=scatter_distribution,
        user_area_width_km=area_width_km,
        user_area_height_km=area_height_km,
        mobility_model=mobility_model,
        random_wandering_max_turn_rad=random_wandering_max_turn_rad,
    )


def build_orbit_config(cfg: dict[str, Any]) -> OrbitConfig:
    """Build OrbitConfig from the resolved-run YAML."""
    base = cfg.get("baseline", cfg)
    ol_val = _resolved_assumption_value(cfg, "orbit_layout")
    ol_val = ol_val if isinstance(ol_val, dict) else {}

    return OrbitConfig(
        altitude_km=ol_val.get("altitude_km", base.get("altitude_km", 780.0)),
        num_satellites=ol_val.get("satellites_per_plane", base.get("satellites", 4)),
        num_planes=ol_val.get("orbital_planes", 1),
        satellite_speed_km_s=base.get("satellite_speed_km_s", 7.4),
        inclination_deg=ol_val.get("inclination_deg", 90.0),
        raan_deg=ol_val.get("raan_deg", 0.0),
        initial_true_anomaly_offset_deg=ol_val.get("initial_true_anomaly_offset_deg", 0.0),
        min_elevation_deg=0.0,
    )


def build_beam_config(cfg: dict[str, Any]) -> BeamConfig:
    """Build BeamConfig from the resolved-run YAML."""
    base = cfg.get("baseline", cfg)
    bg_val = _resolved_assumption_value(cfg, "beam_geometry")
    bg_val = bg_val if isinstance(bg_val, dict) else {}

    return BeamConfig(
        beams_per_satellite=base.get("beams_per_satellite", 7),
        theta_3db_deg=bg_val.get("theta_3db_deg", 2.0),
    )


def build_channel_config(cfg: dict[str, Any]) -> ChannelConfig:
    """Build ChannelConfig from the resolved-run YAML."""
    base = cfg.get("baseline", cfg)
    atm_val = _resolved_assumption_value(cfg, "atmospheric_formula_sign")
    atm_val = atm_val if isinstance(atm_val, dict) else {}
    sign_str = atm_val.get("primary_run_formula", "paper-published-sign")
    if "corrected" in sign_str:
        sign_mode = AtmosphericSignMode.CORRECTED_LOSSY
    else:
        sign_mode = AtmosphericSignMode.PAPER_PUBLISHED

    return ChannelConfig(
        carrier_frequency_hz=base.get("carrier_frequency_ghz", 20.0) * 1e9,
        bandwidth_hz=base.get("bandwidth_mhz", 500.0) * 1e6,
        tx_power_w=base.get("tx_power_w", 2.0),
        noise_psd_dbm_hz=base.get("noise_psd_dbm_hz", -174.0),
        rician_k_db=base.get("rician_k_db", 20.0),
        attenuation_db_per_km=base.get(
            "atmospheric_attenuation_coefficient_db_per_km", 0.05
        ),
        atmospheric_sign_mode=sign_mode,
    )


def build_power_surface_config(cfg: dict[str, Any]) -> PowerSurfaceConfig:
    """Build the opt-in HOBS power surface config from resolved YAML."""
    power_val = _resolved_assumption_value(cfg, "hobs_power_surface")
    power_val = power_val if isinstance(power_val, dict) else {}
    levels = power_val.get("power_codebook_levels_w", (0.5, 1.0, 2.0))

    return PowerSurfaceConfig(
        hobs_power_surface_mode=power_val.get("mode", "static-config"),
        inactive_beam_policy=power_val.get(
            "inactive_beam_policy",
            "excluded-from-active-beams",
        ),
        active_base_power_w=float(power_val.get("active_base_power_w", 0.25)),
        load_scale_power_w=float(power_val.get("load_scale_power_w", 0.35)),
        load_exponent=float(power_val.get("load_exponent", 0.5)),
        max_power_w=(
            None
            if power_val.get("max_power_w", 2.0) is None
            else float(power_val.get("max_power_w", 2.0))
        ),
        power_codebook_profile=str(
            power_val.get("power_codebook_profile", "fixed-mid")
        ),
        power_codebook_levels_w=tuple(float(level) for level in levels),
        total_power_budget_w=(
            None
            if power_val.get("total_power_budget_w", 8.0) is None
            else float(power_val.get("total_power_budget_w", 8.0))
        ),
    )


def build_trainer_config(cfg: dict[str, Any]) -> TrainerConfig:
    """Build TrainerConfig from the resolved-run YAML."""
    base = cfg.get("baseline", cfg)
    experiment_block = _training_experiment_block(cfg)

    # Epsilon schedule (ASSUME-MODQN-REP-004)
    eps_val = _resolved_assumption_value(cfg, "epsilon_schedule")
    eps_val = eps_val if isinstance(eps_val, dict) else {}

    # Target update (ASSUME-MODQN-REP-005)
    tgt_val = _resolved_assumption_value(cfg, "target_update_cadence")
    tgt_val = tgt_val if isinstance(tgt_val, dict) else {}

    # Replay (ASSUME-MODQN-REP-006)
    rep_val = _resolved_assumption_value(cfg, "replay_capacity")
    rep_val = rep_val if isinstance(rep_val, dict) else {}

    # Policy sharing (ASSUME-MODQN-REP-007)
    pol_val = _resolved_assumption_value(cfg, "policy_sharing_mode")
    pol_val = pol_val if isinstance(pol_val, dict) else {}

    # State encoding (ASSUME-MODQN-REP-013)
    enc_val = _resolved_assumption_value(cfg, "state_encoding_and_normalization")
    enc_val = enc_val if isinstance(enc_val, dict) else {}

    ckpt_block = _resolved_assumption_block(
        cfg, "checkpoint_selection_rule", required=True
    )
    ckpt_val = _resolved_assumption_value(
        cfg, "checkpoint_selection_rule", required=True
    )
    if not isinstance(ckpt_val, dict):
        raise ConfigValidationError(
            "Resolved assumption 'checkpoint_selection_rule' must be a mapping."
        )

    hidden = base.get("hidden_layers", [100, 50, 50])

    training_experiment_kind = "baseline"
    training_experiment_id = ""
    method_family = "MODQN-baseline"
    phase = str(cfg.get("track", {}).get("phase", "baseline"))
    comparison_role = "not-applicable"
    r1_reward_mode = "throughput"
    r1_reward_label = "throughput"
    r1_reward_provenance = "paper-backed MODQN throughput objective"
    reward_calibration_enabled = False
    reward_calibration_mode = "raw-unscaled"
    reward_calibration_source = "raw-unscaled"
    reward_calibration_scales = (1.0, 1.0, 1.0)
    reward_normalization_mode = "raw-unscaled"
    load_balance_calibration_mode = "baseline-paper-weight"

    if experiment_block:
        training_experiment_kind = str(
            _required_mapping_field(
                experiment_block,
                "kind",
                context="training_experiment",
            )
        )
        training_experiment_id = str(experiment_block.get("experiment_id", "")).strip()
        method_family = str(
            experiment_block.get(
                "method_family",
                cfg.get("track", {}).get("method_family", "MODQN-baseline"),
            )
        )
        phase = str(experiment_block.get("phase", phase))

        if training_experiment_kind == "reward-calibration":
            reward_calibration_block = _required_mapping_field(
                experiment_block,
                "reward_calibration",
                context="training_experiment",
            )
            if not isinstance(reward_calibration_block, dict):
                raise ConfigValidationError(
                    "training_experiment.reward_calibration must be a mapping."
                )

            reward_calibration_enabled = bool(
                _required_mapping_field(
                    reward_calibration_block,
                    "enabled",
                    context="training_experiment.reward_calibration",
                )
            )
            reward_calibration_mode = str(
                _required_mapping_field(
                    reward_calibration_block,
                    "mode",
                    context="training_experiment.reward_calibration",
                )
            )
            reward_calibration_source = str(
                reward_calibration_block.get("source", "unspecified")
            )

            if not reward_calibration_enabled:
                raise ConfigValidationError(
                    "training_experiment.reward_calibration.enabled must be true when the "
                    "reward-calibration experiment surface is selected."
                )

            if reward_calibration_mode != "divide-by-fixed-scales":
                raise ConfigValidationError(
                    "Only reward_calibration.mode='divide-by-fixed-scales' is currently "
                    f"supported, got {reward_calibration_mode!r}."
                )

            scales_val = _required_mapping_field(
                reward_calibration_block,
                "scales",
                context="training_experiment.reward_calibration",
            )
            if not isinstance(scales_val, dict):
                raise ConfigValidationError(
                    "training_experiment.reward_calibration.scales must be a mapping."
                )
            reward_calibration_scales = (
                float(
                    _required_mapping_field(
                        scales_val,
                        "r1",
                        context="training_experiment.reward_calibration.scales",
                    )
                ),
                float(
                    _required_mapping_field(
                        scales_val,
                        "r2",
                        context="training_experiment.reward_calibration.scales",
                    )
                ),
                float(
                    _required_mapping_field(
                        scales_val,
                        "r3",
                        context="training_experiment.reward_calibration.scales",
                    )
                ),
            )
        elif training_experiment_kind == "phase-03-objective-substitution":
            objective_block = _required_mapping_field(
                experiment_block,
                "objective_substitution",
                context="training_experiment",
            )
            if not isinstance(objective_block, dict):
                raise ConfigValidationError(
                    "training_experiment.objective_substitution must be a mapping."
                )
            if not bool(objective_block.get("enabled", False)):
                raise ConfigValidationError(
                    "training_experiment.objective_substitution.enabled must be true "
                    "for Phase 03 objective-substitution configs."
                )
            r1_reward_mode = str(
                _required_mapping_field(
                    objective_block,
                    "r1_reward_mode",
                    context="training_experiment.objective_substitution",
                )
            )
            r1_reward_label = str(objective_block.get("r1_label", r1_reward_mode))
            r1_reward_provenance = str(
                objective_block.get("r1_provenance", "unspecified")
            )
            comparison_role = str(
                objective_block.get("comparison_role", comparison_role)
            )
        elif training_experiment_kind == "phase-03b-objective-geometry":
            geometry_block = _required_mapping_field(
                experiment_block,
                "objective_geometry",
                context="training_experiment",
            )
            if not isinstance(geometry_block, dict):
                raise ConfigValidationError(
                    "training_experiment.objective_geometry must be a mapping."
                )
            if not bool(geometry_block.get("enabled", False)):
                raise ConfigValidationError(
                    "training_experiment.objective_geometry.enabled must be true "
                    "for Phase 03B objective-geometry configs."
                )
            r1_reward_mode = str(
                _required_mapping_field(
                    geometry_block,
                    "r1_reward_mode",
                    context="training_experiment.objective_geometry",
                )
            )
            r1_reward_label = str(geometry_block.get("r1_label", r1_reward_mode))
            r1_reward_provenance = str(
                geometry_block.get("r1_provenance", "unspecified")
            )
            comparison_role = str(
                geometry_block.get("comparison_role", comparison_role)
            )

            reward_norm_block = _required_mapping_field(
                geometry_block,
                "reward_normalization",
                context="training_experiment.objective_geometry",
            )
            if not isinstance(reward_norm_block, dict):
                raise ConfigValidationError(
                    "training_experiment.objective_geometry.reward_normalization "
                    "must be a mapping."
                )
            reward_calibration_enabled = bool(
                _required_mapping_field(
                    reward_norm_block,
                    "enabled",
                    context="training_experiment.objective_geometry.reward_normalization",
                )
            )
            reward_calibration_mode = str(
                _required_mapping_field(
                    reward_norm_block,
                    "mode",
                    context="training_experiment.objective_geometry.reward_normalization",
                )
            )
            reward_normalization_mode = reward_calibration_mode
            reward_calibration_source = str(
                reward_norm_block.get("source", "phase-03b-config")
            )
            if reward_calibration_enabled:
                if reward_calibration_mode != "divide-by-fixed-scales":
                    raise ConfigValidationError(
                        "Phase 03B reward_normalization currently supports only "
                        "mode='divide-by-fixed-scales'."
                    )
                scales_val = _required_mapping_field(
                    reward_norm_block,
                    "scales",
                    context=(
                        "training_experiment.objective_geometry."
                        "reward_normalization"
                    ),
                )
                if not isinstance(scales_val, dict):
                    raise ConfigValidationError(
                        "training_experiment.objective_geometry."
                        "reward_normalization.scales must be a mapping."
                    )
                reward_calibration_scales = (
                    float(
                        _required_mapping_field(
                            scales_val,
                            "r1",
                            context=(
                                "training_experiment.objective_geometry."
                                "reward_normalization.scales"
                            ),
                        )
                    ),
                    float(
                        _required_mapping_field(
                            scales_val,
                            "r2",
                            context=(
                                "training_experiment.objective_geometry."
                                "reward_normalization.scales"
                            ),
                        )
                    ),
                    float(
                        _required_mapping_field(
                            scales_val,
                            "r3",
                            context=(
                                "training_experiment.objective_geometry."
                                "reward_normalization.scales"
                            ),
                        )
                    ),
                )

            load_balance_block = _required_mapping_field(
                geometry_block,
                "load_balance_calibration",
                context="training_experiment.objective_geometry",
            )
            if not isinstance(load_balance_block, dict):
                raise ConfigValidationError(
                    "training_experiment.objective_geometry."
                    "load_balance_calibration must be a mapping."
                )
            load_balance_calibration_mode = str(
                _required_mapping_field(
                    load_balance_block,
                    "mode",
                    context=(
                        "training_experiment.objective_geometry."
                        "load_balance_calibration"
                    ),
                )
            )
        else:
            raise ConfigValidationError(
                "Only training_experiment.kind values 'reward-calibration', "
                "'phase-03-objective-substitution', and "
                "'phase-03b-objective-geometry' are currently supported, "
                f"got {training_experiment_kind!r}."
            )

    return TrainerConfig(
        hidden_layers=tuple(hidden),
        activation=base.get("activation", "tanh"),
        learning_rate=base.get("learning_rate", 0.01),
        discount_factor=base.get("discount_factor", 0.9),
        batch_size=base.get("batch_size", 128),
        episodes=base.get("episodes", 9000),
        objective_weights=tuple(base.get("objective_weights", [0.5, 0.3, 0.2])),
        epsilon_start=eps_val.get("epsilon_start", 1.0),
        epsilon_end=eps_val.get("epsilon_end", 0.01),
        epsilon_decay_episodes=eps_val.get("epsilon_decay_episodes", 7000),
        target_update_every_episodes=tgt_val.get("target_update_every_episodes", 50),
        replay_capacity=rep_val.get("capacity", 50_000),
        policy_sharing_mode=pol_val.get("mode", "shared"),
        snr_encoding=enc_val.get("snr_encoding", "log1p"),
        offset_scale_km=enc_val.get("offset_scale_km", 100.0),
        load_normalization=enc_val.get("load_normalization", "divide_by_num_users"),
        checkpoint_assumption_id=str(
            ckpt_block.get("assumption_id", "ASSUME-MODQN-REP-015")
        ),
        checkpoint_primary_report=str(
            _required_mapping_field(
                ckpt_val,
                "primary_report",
                context="Resolved assumption 'checkpoint_selection_rule'",
            )
        ),
        checkpoint_secondary_report=str(
            _required_mapping_field(
                ckpt_val,
                "secondary_report",
                context="Resolved assumption 'checkpoint_selection_rule'",
            )
        ),
        training_experiment_kind=training_experiment_kind,
        training_experiment_id=training_experiment_id,
        method_family=method_family,
        phase=phase,
        comparison_role=comparison_role,
        r1_reward_mode=r1_reward_mode,
        r1_reward_label=r1_reward_label,
        r1_reward_provenance=r1_reward_provenance,
        reward_calibration_enabled=reward_calibration_enabled,
        reward_calibration_mode=reward_calibration_mode,
        reward_calibration_source=reward_calibration_source,
        reward_calibration_scales=reward_calibration_scales,
        reward_normalization_mode=reward_normalization_mode,
        load_balance_calibration_mode=load_balance_calibration_mode,
    )


def get_seeds(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract seed values from the resolved-run YAML."""
    resolved = cfg.get("resolved_assumptions", {})
    seed_block = resolved.get("seed_and_rng_policy", {})
    seed_val = seed_block.get("value", {}) if isinstance(seed_block, dict) else {}
    eval_seed_set = seed_val.get("evaluation_seed_set", [])
    return {
        "train_seed": seed_val.get("train_seed", 42),
        "environment_seed": seed_val.get("environment_seed", 1337),
        "mobility_seed": seed_val.get("mobility_seed", 7),
        "evaluation_seed_set": [int(seed) for seed in eval_seed_set],
    }


def build_environment(cfg: dict[str, Any]) -> StepEnvironment:
    """Build a complete StepEnvironment from the resolved-run YAML."""
    return StepEnvironment(
        step_config=build_step_config(cfg),
        orbit_config=build_orbit_config(cfg),
        beam_config=build_beam_config(cfg),
        channel_config=build_channel_config(cfg),
        power_surface_config=build_power_surface_config(cfg),
    )
