import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.actor_critic_model import ActorCriticModel


class ActorCriticAgent:
    def __init__(
            self,
            state_size: int = 9,
            action_size: int = 2,
            hidden_layers: list[int] | None = None,
            gamma: float = 0.97,
            lr: float = 0.0001,
            value_coef: float = 0.5,
            entropy_coef: float = 0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCriticModel(
            state_size=state_size,
            action_size=action_size,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.episode_counter = 0
        self.step_counter = 0
        self.reset_episode()

    def reset_episode(self):
        self.pending_value = None
        self.pending_logits = None
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.rewards = []

    def _state_tensor(self, state) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32).flatten()
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        if len(state) != self.state_size:
            raise ValueError(
                f"Invalid state size: expected {self.state_size}, got {len(state)}"
            )

        return torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def select_action(self, state) -> int:
        state_tensor = self._state_tensor(state)
        logits, value = self.model(state_tensor)
        distribution = Categorical(logits=logits)
        action = distribution.sample()

        self.pending_logits = logits.squeeze(0)
        self.pending_value = value.squeeze(0)

        return int(action.item())

    def remember(self, state, action, reward, next_state, done):
        if self.pending_logits is None or self.pending_value is None:
            raise RuntimeError("remember() called before select_action()")

        action_tensor = torch.tensor(int(action), dtype=torch.long, device=self.device)
        distribution = Categorical(logits=self.pending_logits)

        self.log_probs.append(distribution.log_prob(action_tensor))
        self.values.append(self.pending_value)
        self.entropies.append(distribution.entropy())
        self.rewards.append(float(reward))

        self.pending_logits = None
        self.pending_value = None

    def _compute_returns(self) -> torch.Tensor:
        returns = []
        value = 0.0

        for reward in reversed(self.rewards):
            value = reward + self.gamma * value
            returns.append(value)

        returns.reverse()

        return torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

    def train_step(self):
        if not self.rewards:
            return None

        self.step_counter += 1

        # Actor-Critic updates only at the end of the current on-policy episode.
        # While the episode is running we only collect log_probs, values and rewards.
        return None

    def finish_episode(self):
        if not self.rewards:
            return None

        if not (len(self.log_probs) == len(self.values) == len(self.entropies) == len(self.rewards)):
            raise RuntimeError(
                f"Buffer length mismatch: "
                f"log_probs={len(self.log_probs)}, "
                f"values={len(self.values)}, "
                f"entropies={len(self.entropies)}, "
                f"rewards={len(self.rewards)}"
            )

        returns = self._compute_returns()
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)

        scaled_returns = torch.clamp(returns / 100.0, -1.0, 1.0)
        advantages = scaled_returns - values.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.smooth_l1_loss(values, scaled_returns)
        entropy_loss = entropies.mean()

        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_loss.item(),
            "loss": loss.item(),
            "total_reward": sum(self.rewards),
        }

        self.episode_counter += 1
        self.reset_episode()

        if self.episode_counter % 10 == 0:
            print(
                f"ac_episode={self.episode_counter} "
                f"loss={metrics['loss']:.4f} "
                f"actor={metrics['actor_loss']:.4f} "
                f"critic={metrics['critic_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}"
            )

        return metrics

    def save(self, path: str):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        if checkpoint["state_size"] != self.state_size:
            raise ValueError("Checkpoint state_size does not match current state_size")

        if checkpoint["action_size"] != self.action_size:
            raise ValueError("Checkpoint action_size does not match current action_size")

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
