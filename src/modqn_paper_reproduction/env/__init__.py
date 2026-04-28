"""Environment and state-transition logic for the paper baseline."""

from .orbit import (
    EARTH_RADIUS_KM,
    OrbitConfig,
    OrbitProxy,
    SatelliteSnapshot,
    VisibilityResult,
    ground_eci,
)

from .beam import (
    BeamCenterGround,
    BeamConfig,
    BeamPattern,
)

from .channel import (
    SPEED_OF_LIGHT_M_S,
    AtmosphericSignMode,
    ChannelConfig,
    ChannelResult,
    PathLossResult,
    atmospheric_factor,
    compute_channel,
    compute_path_loss,
    fspl_linear,
    rician_fading_sample,
)

from .step import (
    ActionMask,
    DiagnosticsReport,
    PowerSurfaceConfig,
    RewardComponents,
    StepConfig,
    StepEnvironment,
    StepResult,
    UserState,
)

__all__ = [
    "EARTH_RADIUS_KM",
    "SPEED_OF_LIGHT_M_S",
    "ActionMask",
    "AtmosphericSignMode",
    "BeamCenterGround",
    "BeamConfig",
    "BeamPattern",
    "ChannelConfig",
    "ChannelResult",
    "DiagnosticsReport",
    "OrbitConfig",
    "OrbitProxy",
    "PathLossResult",
    "PowerSurfaceConfig",
    "RewardComponents",
    "SatelliteSnapshot",
    "StepConfig",
    "StepEnvironment",
    "StepResult",
    "UserState",
    "VisibilityResult",
    "atmospheric_factor",
    "compute_channel",
    "compute_path_loss",
    "fspl_linear",
    "ground_eci",
    "rician_fading_sample",
]
