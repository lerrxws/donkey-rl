from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from agents.actor_critic.networks.actor import ActorNetwork
from agents.actor_critic.networks.critic import CriticNetwork


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    action_probs: np.ndarray


class BaseActorCriticAgent(ABC):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: list[int] | None = None,
        gamma: float = 0.97,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
        entropy_coef: float = 0.01,
        reward_scale: float = 100.0,
        max_grad_norm: float = 1.0,
    ):
        self.state_size = state_size  # Number of values in one state.
        self.action_size = action_size  # Number of discrete actions.

        self.gamma = gamma  # Discount factor for future rewards.
        self.actor_lr = actor_lr  # Learning rate for Actor.
        self.critic_lr = critic_lr  # Learning rate for Critic.
        self.entropy_coef = entropy_coef  # Exploration bonus weight.
        self.reward_scale = reward_scale  # Scales raw rewards before target calculation.
        self.max_grad_norm = max_grad_norm  # Gradient clipping threshold.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Training device.

        self.actor = ActorNetwork(  # Policy network π(a | s).
            state_size=state_size,
            action_size=action_size,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.critic = CriticNetwork(  # Value network V(s).
            state_size=state_size,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(  # Optimizer for Actor parameters.
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(  # Optimizer for Critic parameters.
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.episode_counter = 0  # Number of completed episodes.
        self.step_counter = 0  # Number of processed environment steps.

        self.last_probs: np.ndarray | None = None  # Last action probabilities.
        self.last_metrics: dict | None = None  # Last training metrics.

        self.__pending_log_prob: torch.Tensor | None = None  # Stored logπ(a | s).
        self.__pending_value: torch.Tensor | None = None  # Stored V(s).
        self.__pending_entropy: torch.Tensor | None = None  # Stored policy entropy.
        self.__pending_action_probs: np.ndarray | None = None  # Stored π(a | s).
        self.__pending_action: int | None = None  # Stored selected action.

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def select_action(self, state) -> int:
        """
        Samples an action from Actor policy:

            A ~ π(. | S, θ)

        Stores:
            log π(A | S, θ)
            V(S, w)
            H(π(. | S, θ))
            π(. | S, θ)
        """
        state_tensor = self._state_tensor(state)

        logits = self.actor(state_tensor).squeeze(0)
        value = self.critic(state_tensor).reshape(())

        distribution = Categorical(logits=logits)
        action = distribution.sample()
        action_int = int(action.item())

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        self.last_probs = probs
        self.__pending_action_probs = probs.copy()

        self.__pending_log_prob = distribution.log_prob(action)
        self.__pending_value = value
        self.__pending_entropy = distribution.entropy()
        self.__pending_action = action_int

        return action_int

    @abstractmethod
    def remember(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> dict | None:
        pass

    # -------------------------------------------------------------------------
    # Protected methods for subclasses
    # -------------------------------------------------------------------------

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

    def _make_transition(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> Transition:
        if (
            self.__pending_log_prob is None
            or self.__pending_value is None
            or self.__pending_entropy is None
            or self.__pending_action_probs is None
            or self.__pending_action is None
        ):
            raise RuntimeError("select_action() must be called before remember().")

        action = int(action)
        if action != self.__pending_action:
            raise ValueError(
                "remember() action does not match the action returned by "
                f"select_action(): expected {self.__pending_action}, got {action}"
            )

        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=action,
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
            log_prob=self.__pending_log_prob,
            value=self.__pending_value,
            entropy=self.__pending_entropy,
            action_probs=self.__pending_action_probs.copy(),
        )

        self.__clear_pending_action()

        return transition

    def _scale_reward(self, reward: float) -> float:
        r = reward / self.reward_scale
        return max(-1.0, min(1.0, r))

    def _train_actor_critic_separately(
        self,
        transitions: list[Transition],
        targets: torch.Tensor,
    ) -> dict:
        """
        Batch Actor-Critic update with precomputed targets.

        Used by Monte Carlo / n-step:
            targets are computed first,
            then Critic and Actor are updated separately.
        """
        log_probs, values, entropies = self.__extract_training_tensors(transitions)

        # δ = target - V(S)
        td_errors = targets.detach() - values

        # Actor must not backprop through Critic.
        advantages = td_errors.detach()

        critic_loss = self._update_critic(
            values=values,
            targets=targets,
        )

        actor_loss, entropy_bonus = self._update_actor(
            log_probs=log_probs,
            advantages=advantages,
            entropies=entropies,
        )

        metrics = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "mean_value": float(values.detach().mean().item()),
            "mean_target": float(targets.detach().mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "batch_size": len(transitions),
        }

        self.last_metrics = metrics
        return metrics

    def _update_critic(
        self,
        values: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Updates Critic.

        Pseudocode idea:
            φ ← φ + αφ δ ∇V(S)

        where:
            δ = target - V(S)

        Implemented as minimizing:
            L_critic = 1/2 * δ²
        """
        td_errors = targets.detach() - values
        critic_loss = 0.5 * td_errors.pow(2).mean()

        self.__optimize_critic(critic_loss)

        return critic_loss.detach()

    def _update_actor(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropies: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates Actor.

        Pseudocode idea:
            θ ← θ + αθ δ ∇logπ(A | S)

        Since PyTorch minimizes loss:
            L_actor = -logπ(A | S) * δ
        """
        actor_loss = -(log_probs * advantages.detach()).mean()

        entropy_bonus = entropies.mean()

        if self.entropy_coef > 0.0:
            actor_loss = actor_loss - self.entropy_coef * entropy_bonus

        self.__optimize_actor(actor_loss)

        return actor_loss.detach(), entropy_bonus.detach()

    # -------------------------------------------------------------------------
    # Public checkpoint API
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor_state": self.actor.state_dict(),
                "critic_state": self.critic.state_dict(),
                "actor_optimizer_state": self.actor_optimizer.state_dict(),
                "critic_optimizer_state": self.critic_optimizer.state_dict(),
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "entropy_coef": self.entropy_coef,
                "reward_scale": self.reward_scale,
                "max_grad_norm": self.max_grad_norm,
                "episode_counter": self.episode_counter,
                "step_counter": self.step_counter,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        if checkpoint["state_size"] != self.state_size:
            raise ValueError("Checkpoint state_size mismatch")

        if checkpoint["action_size"] != self.action_size:
            raise ValueError("Checkpoint action_size mismatch")

        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])

        self.gamma = checkpoint.get("gamma", self.gamma)
        self.actor_lr = checkpoint.get("actor_lr", self.actor_lr)
        self.critic_lr = checkpoint.get("critic_lr", self.critic_lr)
        self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)
        self.reward_scale = checkpoint.get("reward_scale", self.reward_scale)
        self.max_grad_norm = checkpoint.get("max_grad_norm", self.max_grad_norm)
        self.episode_counter = checkpoint.get("episode_counter", 0)
        self.step_counter = checkpoint.get("step_counter", 0)

    # -------------------------------------------------------------------------
    # Private methods used only inside BaseActorCriticAgent
    # -------------------------------------------------------------------------

    def __extract_training_tensors(
        self,
        transitions: list[Transition],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = torch.stack([t.log_prob for t in transitions])
        values = torch.stack([t.value for t in transitions])
        entropies = torch.stack([t.entropy for t in transitions])

        return log_probs, values, entropies

    def __optimize_actor(self, actor_loss: torch.Tensor) -> None:
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

    def __optimize_critic(self, critic_loss: torch.Tensor) -> None:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

    def __clear_pending_action(self) -> None:
        self.__pending_log_prob = None
        self.__pending_value = None
        self.__pending_entropy = None
        self.__pending_action_probs = None
        self.__pending_action = None
