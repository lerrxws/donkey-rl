import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor):
        x = self.shared(state)

        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)

        return logits, value


class MonteCarloActorCriticAgent:
    def __init__(
            self,
            state_dim: int,
            action_dim: int = 2,
            hidden_size: int = 64,
            gamma: float = 0.99,
            lr: float = 1e-4,
            value_coef: float = 0.5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.value_coef = value_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.reset_episode()

    def reset_episode(self):
        self.log_probs = []
        self.values = []
        self.rewards = []

    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32).flatten()
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        if len(state) != self.state_dim:
            raise ValueError(
                f"Invalid state size: expected {self.state_dim}, got {len(state)}"
            )

        return torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def select_action(self, state: np.ndarray) -> tuple[int, float]:
        state_tensor = self.state_to_tensor(state)

        logits, value = self.model(state_tensor)

        dist = Categorical(logits=logits)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        self.log_probs.append(log_prob.squeeze(0))
        self.values.append(value.squeeze(0))

        return int(action.item()), float(value.item())

    def record_reward(self, reward: float):
        self.rewards.append(float(reward))

    def compute_returns(self) -> torch.Tensor:
        """
        Monte Carlo return:

            G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        """
        returns = []
        G = 0.0

        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.append(G)

        returns.reverse()

        return torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

    def update(self):
        """
        Monte Carlo Actor-Critic update.
        Called once after the episode ends.
        """
        if len(self.rewards) == 0:
            return None

        if not (len(self.log_probs) == len(self.values) == len(self.rewards)):
            raise RuntimeError(
                f"Buffer length mismatch: "
                f"log_probs={len(self.log_probs)}, "
                f"values={len(self.values)}, "
                f"rewards={len(self.rewards)}"
            )

        returns = self.compute_returns()

        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        # Advantage:
        # A_t = G_t - V(s_t)
        advantages = returns - values.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss:
        # increase probability of good actions,
        # decrease probability of bad actions
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss:
        # make V(s_t) closer to G_t
        critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + self.value_coef * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "loss": loss.item(),
            "total_reward": sum(self.rewards),
        }

        self.reset_episode()

        return metrics

    def save(self, path: str):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        if checkpoint["state_dim"] != self.state_dim:
            raise ValueError("Checkpoint state_dim does not match current state_dim")

        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint action_dim does not match current action_dim")

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])