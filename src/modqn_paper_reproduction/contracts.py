from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RewardVector:
    throughput: float
    handover: float
    load_balance: float

    def scalarize(self, weights: tuple[float, float, float]) -> float:
        return (
            self.throughput * weights[0]
            + self.handover * weights[1]
            + self.load_balance * weights[2]
        )


@dataclass(frozen=True)
class PaperBaselineParameters:
    satellites: int = 4
    beams_per_satellite: int = 7
    users: int = 100
    altitude_km: float = 780.0
    user_speed_kmh: float = 30.0
    satellite_speed_km_s: float = 7.4
    carrier_frequency_ghz: float = 20.0
    bandwidth_mhz: float = 500.0
    tx_power_w: float = 2.0
    noise_psd_dbm_hz: float = -174.0
    rician_k_db: float = 20.0
    attenuation_db_per_km: float = 0.05
    slot_duration_s: float = 1.0
    episode_duration_s: float = 10.0
    episodes: int = 9000
    hidden_layers: tuple[int, int, int] = (100, 50, 50)
    learning_rate: float = 0.01
    discount_factor: float = 0.9
    batch_size: int = 128
    objective_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_index: int
    steps: int
    reward: RewardVector
    scalar_reward: float
    total_handovers: int


@dataclass(frozen=True)
class SweepPoint:
    sweep_name: str
    x_value: float
    method: str
    reward: RewardVector
    scalar_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactManifest:
    run_id: str
    paper_id: str
    seed: int
    assumption_ids: tuple[str, ...]
    notes: tuple[str, ...] = ()
