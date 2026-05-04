import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.actor_critic_model import ActorCriticModel


class BootstrapActorCriticAgent:
    def __init__(
        self,
        state_size: int = 9,
        action_size: int = 2,
        hidden_layers: list[int] | None = None,
        gamma: float = 0.97,
        lr: float = 0.0001,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        reward_scale: float = 100.0,
        max_grad_norm: float = 1.0,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm

        # Compatibility with your current logging code.
        # Actor-Critic does not use epsilon-greedy exploration.
        self.epsilon = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCriticModel(
            state_size=state_size,
            action_size=action_size,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        self.episode_counter = 0
        self.step_counter = 0

        self.pending_action: int | None = None
        self.pending_log_prob: torch.Tensor | None = None
        self.pending_value: torch.Tensor | None = None
        self.pending_entropy: torch.Tensor | None = None

        self.last_transition: dict | None = None

    def _state_tensor(self, state) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32).flatten()
        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if len(state) != self.state_size:
            raise ValueError(
                f"Invalid state size: expected {self.state_size}, got {len(state)}"
            )

        return torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def select_action(
        self,
        state,
        action_mask: list[bool] | None = None,
    ) -> int:
        """
        action_mask example:
            [True, True]   -> actions 0 and 1 are allowed
            [True, False]  -> only action 0 is allowed
        """

        state_tensor = self._state_tensor(state)

        logits, value = self.model(state_tensor)

        logits = logits.squeeze(0)
        value = value.squeeze(0)

        if action_mask is not None:
            if len(action_mask) != self.action_size:
                raise ValueError(
                    f"Invalid action_mask size: expected {self.action_size}, got {len(action_mask)}"
                )

            mask = torch.tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )

            if not mask.any():
                raise ValueError("action_mask must allow at least one action")

            logits = logits.masked_fill(~mask, -1e9)

        distribution = Categorical(logits=logits)
        action = distribution.sample()

        self.pending_action = int(action.item())
        self.pending_log_prob = distribution.log_prob(action)
        self.pending_value = value
        self.pending_entropy = distribution.entropy()

        return self.pending_action

    def remember(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> None:
        if (
            self.pending_action is None
            or self.pending_log_prob is None
            or self.pending_value is None
            or self.pending_entropy is None
        ):
            raise RuntimeError("remember() called before select_action()")

        if int(action) != self.pending_action:
            raise RuntimeError(
                f"Action mismatch: policy selected {self.pending_action}, "
                f"but environment executed {action}. "
                f"Do not override the selected action. Use action_mask instead."
            )

        self.last_transition = {
            "reward": float(reward),
            "next_state": next_state,
            "done": bool(done),
            "log_prob": self.pending_log_prob,
            "value": self.pending_value,
            "entropy": self.pending_entropy,
        }

        self.pending_action = None
        self.pending_log_prob = None
        self.pending_value = None
        self.pending_entropy = None

    def train_step(self):
        if self.last_transition is None:
            return None

        self.step_counter += 1

        reward = self.last_transition["reward"]
        next_state = self.last_transition["next_state"]
        done = self.last_transition["done"]

        log_prob = self.last_transition["log_prob"]
        value = self.last_transition["value"]
        entropy = self.last_transition["entropy"]

        scaled_reward = reward / self.reward_scale
        scaled_reward = max(-1.0, min(1.0, scaled_reward))

        reward_tensor = torch.tensor(
            scaled_reward,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            if done:
                next_value = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                next_state_tensor = self._state_tensor(next_state)
                _, next_value = self.model(next_state_tensor)
                next_value = next_value.squeeze(0)

            td_target = reward_tensor + self.gamma * next_value

        advantage = td_target - value

        actor_loss = -(log_prob * advantage.detach())
        critic_loss = F.smooth_l1_loss(value, td_target)
        entropy_bonus = entropy

        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy_bonus
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.max_grad_norm,
        )

        self.optimizer.step()

        metrics = {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "value": float(value.item()),
            "td_target": float(td_target.item()),
            "advantage": float(advantage.item()),
            "reward": float(reward),
            "scaled_reward": float(scaled_reward),
        }

        self.last_transition = None

        return metrics

    def finish_episode(self):
        self.episode_counter += 1

        if self.pending_action is not None:
            self.pending_action = None
            self.pending_log_prob = None
            self.pending_value = None
            self.pending_entropy = None

        self.last_transition = None

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "reward_scale": self.reward_scale,
                "max_grad_norm": self.max_grad_norm,
                "episode_counter": self.episode_counter,
                "step_counter": self.step_counter,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if checkpoint["state_size"] != self.state_size:
            raise ValueError(
                f"Checkpoint state_size mismatch: "
                f"expected {self.state_size}, got {checkpoint['state_size']}"
            )

        if checkpoint["action_size"] != self.action_size:
            raise ValueError(
                f"Checkpoint action_size mismatch: "
                f"expected {self.action_size}, got {checkpoint['action_size']}"
            )

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.gamma = checkpoint.get("gamma", self.gamma)
        self.value_coef = checkpoint.get("value_coef", self.value_coef)
        self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)
        self.reward_scale = checkpoint.get("reward_scale", self.reward_scale)
        self.max_grad_norm = checkpoint.get("max_grad_norm", self.max_grad_norm)

        self.episode_counter = checkpoint.get("episode_counter", 0)
        self.step_counter = checkpoint.get("step_counter", 0)