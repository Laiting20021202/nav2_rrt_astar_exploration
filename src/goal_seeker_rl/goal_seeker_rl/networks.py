"""Neural network definitions used by TD3."""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_linear(module: nn.Module) -> None:
    """Apply Xavier initialization to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    """Deterministic actor network for continuous control."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.apply(_init_linear)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return normalized action in [-1, 1] for each action dimension."""
        return self.model(state)


class Critic(nn.Module):
    """Twin-critic network used by TD3 for clipped double Q-learning."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = state_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_linear)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q-value estimates from both critics."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only the first critic value for actor optimization."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class ReferenceActor(nn.Module):
    """Actor architecture compatible with turtlebot3_drlnav DDPG/TD3 checkpoints."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.fa1 = nn.Linear(state_dim, hidden_dim)
        self.fa2 = nn.Linear(hidden_dim, hidden_dim)
        self.fa3 = nn.Linear(hidden_dim, action_dim)
        self.apply(_init_linear)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return normalized action in [-1, 1] for each action dimension."""
        x1 = torch.relu(self.fa1(state))
        x2 = torch.relu(self.fa2(x1))
        return torch.tanh(self.fa3(x2))


class ReferenceCritic(nn.Module):
    """Twin-critic architecture compatible with turtlebot3_drlnav TD3 training."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()

        # Q1 branch
        self.l1 = nn.Linear(state_dim, hidden_dim // 2)
        self.l2 = nn.Linear(action_dim, hidden_dim // 2)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

        # Q2 branch
        self.l5 = nn.Linear(state_dim, hidden_dim // 2)
        self.l6 = nn.Linear(action_dim, hidden_dim // 2)
        self.l7 = nn.Linear(hidden_dim, hidden_dim)
        self.l8 = nn.Linear(hidden_dim, 1)
        self.apply(_init_linear)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q-value estimates from both critics."""
        xs1 = torch.relu(self.l1(state))
        xa1 = torch.relu(self.l2(action))
        x1 = torch.cat((xs1, xa1), dim=-1)
        x1 = torch.relu(self.l3(x1))
        q1 = self.l4(x1)

        xs2 = torch.relu(self.l5(state))
        xa2 = torch.relu(self.l6(action))
        x2 = torch.cat((xs2, xa2), dim=-1)
        x2 = torch.relu(self.l7(x2))
        q2 = self.l8(x2)
        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only the first critic value for actor optimization."""
        xs = torch.relu(self.l1(state))
        xa = torch.relu(self.l2(action))
        x = torch.cat((xs, xa), dim=-1)
        x = torch.relu(self.l3(x))
        return self.l4(x)
