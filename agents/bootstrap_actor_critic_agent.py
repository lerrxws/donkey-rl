import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.actor_critic_model import ActorCriticModel


class BootstrapActorCriticAgent:
    """
    Bootstrap Actor-Critic агент.

    Основна ідея:
        target = reward + gamma * V(next_state)

    Тобто агент навчається не в кінці епізоду,
    а після кожного кроку гри.

    Це краще для твого випадку, бо епізоди короткі,
    rewards шумні, а чекати finish_episode() занадто слабко.
    """

    def __init__(
        self,
        state_size: int = 9,
        action_size: int = 2,
        hidden_layers: list[int] | None = None,
        gamma: float = 0.97,
        lr: float = 0.0001,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
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
        self.pending_policy_action_used: bool = True

        self.last_transition: dict | None = None
        self.last_probs = None

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

    def select_action(self, state) -> int:
        state_tensor = self._state_tensor(state)

        logits, value = self.model(state_tensor)

        logits = logits.squeeze(0)
        value = value.squeeze(0)

        distribution = Categorical(logits=logits)
        action = distribution.sample()

        probs = torch.softmax(logits, dim=-1)
        self.last_probs = probs.detach().cpu().numpy()

        self.pending_action = int(action.item())
        self.pending_log_prob = distribution.log_prob(action)
        self.pending_value = value
        self.pending_entropy = distribution.entropy()
        self.pending_policy_action_used = True

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

        policy_action_used = int(action) == self.pending_action
        self.pending_policy_action_used = policy_action_used

        self.last_transition = {
            "reward": float(reward),
            "next_state": next_state,
            "done": bool(done),
            "log_prob": self.pending_log_prob,
            "value": self.pending_value,
            "entropy": self.pending_entropy,
            "policy_action_used": policy_action_used,
        }

        self.pending_action = None
        self.pending_log_prob = None
        self.pending_value = None
        self.pending_entropy = None
        self.pending_policy_action_used = True

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
        policy_action_used = self.last_transition.get("policy_action_used", True)

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

        if policy_action_used:
            actor_loss = -(log_prob * advantage.detach())
        else:
            actor_loss = torch.zeros_like(value)

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
            "actor_update": bool(policy_action_used),
        }

        self.last_transition = None

        return metrics

    def finish_episode(self):
        self.episode_counter += 1

        self.pending_action = None
        self.pending_log_prob = None
        self.pending_value = None
        self.pending_entropy = None
        self.pending_policy_action_used = True
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
            raise ValueError("Checkpoint state_size does not match current state_size")

        if checkpoint["action_size"] != self.action_size:
            raise ValueError("Checkpoint action_size does not match current action_size")

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.gamma = checkpoint.get("gamma", self.gamma)
        self.value_coef = checkpoint.get("value_coef", self.value_coef)
        self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)
        self.reward_scale = checkpoint.get("reward_scale", self.reward_scale)
        self.max_grad_norm = checkpoint.get("max_grad_norm", self.max_grad_norm)
        self.episode_counter = checkpoint.get("episode_counter", 0)
        self.step_counter = checkpoint.get("step_counter", 0)