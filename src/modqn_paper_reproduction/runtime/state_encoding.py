"""State encoding helpers for the runtime seam."""

from __future__ import annotations

import numpy as np

from ..env.step import UserState
from .trainer_spec import TrainerConfig


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

    snr = user_state.channel_quality.astype(np.float64)
    if config.snr_encoding == "log1p":
        snr = np.log1p(np.maximum(snr, 0.0)).astype(np.float32)
    else:
        snr = snr.astype(np.float32)

    offsets = user_state.beam_offsets.astype(np.float32).flatten()
    if config.offset_scale_km > 0:
        offsets = offsets / config.offset_scale_km

    loads = user_state.beam_loads.astype(np.float32)
    if config.load_normalization == "divide_by_num_users" and num_users > 0:
        loads = loads / num_users

    return np.concatenate([access, snr, offsets, loads])


def state_dim_for(num_beams_total: int) -> int:
    """Compute the flat state dimension for a given topology."""
    return 5 * num_beams_total
