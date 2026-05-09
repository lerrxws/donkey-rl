import torch

from src.agents.actor_critic.agents.actor_critic import BaseActorCriticAgent


class OneStepActorCriticAgent(BaseActorCriticAgent):
    def __init__(
        self,
        state_size: int = 5,
        action_size: int = 2,
        hidden_layers: list[int] | None = None,
        gamma: float = 0.97,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
        entropy_coef: float = 0.0,
        reward_scale: float = 100.0,
        max_grad_norm: float = 1.0,
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
    ) -> dict:
        """
        One-step Actor-Critic update.

        Pseudocode form:
            A ~ pi_theta(. | S)
            observe R, S'
            delta = R + gamma * V(S') - V(S)
            update Critic with delta
            update Actor with advantage * log pi(A | S)
        """
        transition = self._make_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        reward_tensor = torch.tensor(
            self._scale_reward(transition.reward),
            dtype=torch.float32,
            device=self.device,
        )

        value = self.__as_scalar(transition.value, "value")
        log_prob = self.__as_scalar(transition.log_prob, "log_prob")
        entropy = self.__as_scalar(transition.entropy, "entropy")

        if transition.done:
            next_value = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=self.device,
            )
            target = reward_tensor
        else:
            next_value = self.__value_of_state(transition.next_state)
            target = reward_tensor + self.gamma * next_value

        next_value = self.__as_scalar(next_value, "next_value")
        target = self.__as_scalar(target, "target")

        # TD error is logged; its detached value is the actor advantage.
        td_error = target.detach() - value.detach()
        advantage = td_error.detach()

        self.__require_finite("value", value.detach())
        self.__require_finite("target", target.detach())
        self.__require_finite("advantage", advantage)
        self.__require_finite("log_prob", log_prob.detach())
        self.__require_finite("entropy", entropy.detach())

        values = value.reshape(1)
        targets = target.detach().reshape(1)
        log_probs = log_prob.reshape(1)
        entropies = entropy.reshape(1)
        advantages = advantage.reshape(1)

        critic_loss = self._update_critic(
            values=values,
            targets=targets,
        )

        actor_loss, entropy_bonus = self._update_actor(
            log_probs=log_probs,
            advantages=advantages,
            entropies=entropies,
        )

        self.__require_finite("actor_loss", actor_loss)
        self.__require_finite("critic_loss", critic_loss)
        self.__require_finite("entropy_mean", entropy_bonus)

        self.step_counter += 1

        action_probs = transition.action_probs
        selected_action_prob = None

        if 0 <= transition.action < len(action_probs):
            selected_action_prob = float(action_probs[transition.action])

        self.last_metrics = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy_mean": float(entropy_bonus.item()),
            "td_error": float(td_error.item()),
            "advantage": float(advantage.item()),
            "value": float(value.detach().item()),
            "next_value": float(next_value.detach().item()),
            "target": float(target.detach().item()),
            "reward": float(transition.reward),
            "scaled_reward": float(reward_tensor.item()),
            "done": transition.done,
            "action": int(transition.action),
            "batch_size": 1,
        }

        if len(action_probs) >= 1:
            self.last_metrics["prob_no_jump"] = float(action_probs[0])

        if len(action_probs) >= 2:
            self.last_metrics["prob_jump"] = float(action_probs[1])

        if selected_action_prob is not None:
            self.last_metrics["selected_action_prob"] = selected_action_prob

        return self.last_metrics.copy()

    def finish_episode(self) -> dict | None:
        """
        One-step updates on every transition,
        so there is nothing to train at the end of the episode.
        """
        self.episode_counter += 1
        if self.last_metrics is None:
            return None

        return self.last_metrics.copy()

    # -------------------------------------------------------------------------
    # Private methods used only inside OneStepActorCriticAgent
    # -------------------------------------------------------------------------

    def __value_of_state(self, state) -> torch.Tensor:
        """
        Computes V(S') without gradients for bootstrap target.
        """
        with torch.no_grad():
            state_tensor = self._state_tensor(state)
            value = self.critic(state_tensor)
            return self.__as_scalar(value, "next_value")

    def __as_scalar(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """
        Converts a single-value tensor to a scalar tensor.
        """
        if tensor.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one value, got shape {tuple(tensor.shape)}"
            )

        return tensor.reshape(())

    def __require_finite(self, name: str, tensor: torch.Tensor) -> None:
        """
        Fails fast when RL training produces NaN or inf values.
        """
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN or inf")
