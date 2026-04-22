"""TD3 agent implementation for goal_seeker_rl."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from .networks import Actor, Critic


@dataclass
class TD3Config:
    """Hyperparameters for TD3 training."""

    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 128
    replay_size: int = 200_000
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_std: float = 0.1
    warmup_steps: int = 2_000
    hidden_dim: int = 256


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy learning."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        self.state[self._ptr] = state
        self.action[self._ptr] = action
        self.reward[self._ptr] = reward
        self.next_state[self._ptr] = next_state
        self.done[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one mini-batch and transfer tensors to the target device."""
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self.state[idx]).to(device),
            torch.from_numpy(self.action[idx]).to(device),
            torch.from_numpy(self.reward[idx]).to(device),
            torch.from_numpy(self.next_state[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device),
        )


class TD3Agent:
    """Twin Delayed DDPG implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
        config: Optional[TD3Config] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device if device is not None else torch.device("cpu")
        self.config = config if config is not None else TD3Config()

        self.actor = Actor(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.config.hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.config.critic_lr)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, self.config.replay_size)
        self.total_it = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Return normalized action in [-1, 1]."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if explore:
            noise = np.random.normal(0.0, self.config.exploration_std, size=self.action_dim)
            action = action + noise

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def sample_random_action(self) -> np.ndarray:
        """Sample a random action in the normalized action space."""
        return np.random.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push transition into replay memory."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[dict]:
        """Run one TD3 update if enough samples are available."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.config.batch_size,
            self.device,
        )

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.config.policy_noise
            ).clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = reward + (1.0 - done) * self.config.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = f.mse_loss(current_q1, target) + f.mse_loss(current_q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0)
        self.critic_opt.step()

        actor_loss_value: Optional[float] = None
        if self.total_it % self.config.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
            self.actor_opt.step()
            self._soft_update(self.actor_target, self.actor, self.config.tau)
            self._soft_update(self.critic_target, self.critic, self.config.tau)
            actor_loss_value = float(actor_loss.detach().cpu().item())

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": actor_loss_value,
        }

    def save(self, path: str) -> None:
        """Save model and optimizer states."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "total_it": self.total_it,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "config": asdict(self.config),
            },
            output_path,
        )

    def load(self, path: str, strict: bool = True) -> None:
        """Load model and optimizer states."""
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.actor_target.load_state_dict(payload["actor_target"], strict=strict)
        self.critic.load_state_dict(payload["critic"], strict=strict)
        self.critic_target.load_state_dict(payload["critic_target"], strict=strict)
        if "actor_opt" in payload:
            self.actor_opt.load_state_dict(payload["actor_opt"])
        if "critic_opt" in payload:
            self.critic_opt.load_state_dict(payload["critic_opt"])
        self.total_it = int(payload.get("total_it", 0))

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        """Polyak averaging update."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

