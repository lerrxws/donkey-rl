import numpy as np
import torch

from agents.actor_critic.agents.actor_critic import (
    BaseActorCriticAgent,
    Transition,
)


class EpisodicActorCriticAgent(BaseActorCriticAgent):
    def __init__(
        self,
        state_size: int = 5,
        action_size: int = 2,
        hidden_layers: list[int] | None = None,
        gamma: float = 0.97,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
        entropy_coef: float = 0.01,
        reward_scale: float = 100.0,
        max_grad_norm: float = 1.0,
        normalize_returns: bool = True,
    ):
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            hidden_layers=hidden_layers,
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            entropy_coef=entropy_coef,
            reward_scale=reward_scale,
            max_grad_norm=max_grad_norm,
        )

        self.normalize_returns = normalize_returns  # Normalizes episode returns.
        self.__episode_buffer: list[Transition] = []  # Stores one full episode.

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def remember(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> None:
        """
        Stores one transition until the episode ends.
        """
        transition = self._make_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self.__episode_buffer.append(transition)

    def finish_episode(self) -> dict | None:
        """
        Trains after the full episode using Monte Carlo returns.

            G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...

        No bootstrap is used.
        """
        if not self.__episode_buffer:
            return None

        self.episode_counter += 1
        self.step_counter += len(self.__episode_buffer)

        targets = self.__compute_returns(self.__episode_buffer)

        metrics = self._train_actor_critic_separately(
            transitions=self.__episode_buffer,
            targets=targets,
        )

        metrics["episode_steps"] = len(self.__episode_buffer)
        metrics["total_raw_reward"] = sum(
            transition.reward for transition in self.__episode_buffer
        )

        self.__add_probability_metrics(metrics)

        self.last_metrics = metrics
        self.__clear_episode_buffer()

        return metrics

    # -------------------------------------------------------------------------
    # Private methods used only inside MonteCarloActorCriticAgent
    # -------------------------------------------------------------------------

    def __compute_returns(self, transitions: list[Transition]) -> torch.Tensor:
        """
        Computes discounted Monte Carlo returns.

            G_t = r_t + γG_{t+1}
        """
        returns: list[float] = []
        G = 0.0

        for transition in reversed(transitions):
            reward = self._scale_reward(transition.reward)
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        if self.normalize_returns and len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        return returns_tensor

    def __add_probability_metrics(self, metrics: dict) -> None:
        """
        Adds episode-level policy probability metrics.

            probs = π(a | s)
        """
        if not self.__episode_buffer:
            return

        probs = np.stack(
            [transition.action_probs for transition in self.__episode_buffer]
        )

        for action_idx in range(self.action_size):
            metrics[f"mean_prob_action_{action_idx}"] = float(
                probs[:, action_idx].mean()
            )

        selected_probs = np.array(
            [
                transition.action_probs[transition.action]
                for transition in self.__episode_buffer
            ],
            dtype=np.float32,
        )

        metrics["mean_selected_action_prob"] = float(selected_probs.mean())
        metrics["min_selected_action_prob"] = float(selected_probs.min())
        metrics["max_selected_action_prob"] = float(selected_probs.max())

        if self.action_size == 2:
            metrics["mean_prob_no_jump"] = float(probs[:, 0].mean())
            metrics["mean_prob_jump"] = float(probs[:, 1].mean())

    def __clear_episode_buffer(self) -> None:
        self.__episode_buffer.clear()