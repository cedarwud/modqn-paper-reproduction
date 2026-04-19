"""Q-network implementation for the runtime seam."""

from __future__ import annotations

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Single DQN for one reward objective."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        act_cls = nn.Tanh if activation == "tanh" else nn.ReLU
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
