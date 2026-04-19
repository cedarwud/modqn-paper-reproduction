"""Replay-buffer implementation for the runtime seam."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.random import Generator


class ReplayBuffer:
    """Fixed-capacity FIFO experience replay (ASSUME-MODQN-REP-006)."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf: deque[tuple[Any, ...]] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward_3: np.ndarray,
        next_state: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward_3, next_state, mask, next_mask, done))

    def sample(
        self, batch_size: int, rng: Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[i] for i in indices]

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        masks = np.array([b[4] for b in batch])
        next_masks = np.array([b[5] for b in batch])
        dones = np.array([b[6] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, masks, next_masks, dones

    def __len__(self) -> int:
        return len(self._buf)
