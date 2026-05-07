import torch

from agents.actor_critic.agents.actor_critic import BaseActorCriticAgent


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
    ) -> None:
        """
        One-step Actor-Critic update.

        Pseudocode form:
            A ~ πθ(. | S)
            observe R, S'
            δ = R + γV(S') - V(S)
            update Critic with δ
            update Actor with δ logπ(A | S)
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

        if transition.done:
            next_value = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            next_value = self.__value_of_state(transition.next_state)

        # target = R + γV(S')
        target = reward_tensor + self.gamma * next_value

        # δ = target - V(S)
        td_error = target.detach() - transition.value.detach()

        values = transition.value.unsqueeze(0)
        targets = target.detach().unsqueeze(0)
        log_probs = transition.log_prob.unsqueeze(0)
        entropies = transition.entropy.unsqueeze(0)
        advantages = td_error.unsqueeze(0)

        critic_loss = self._update_critic(
            values=values,
            targets=targets,
        )

        actor_loss, entropy_bonus = self._update_actor(
            log_probs=log_probs,
            advantages=advantages,
            entropies=entropies,
        )

        self.step_counter += 1

        self.last_metrics = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "td_error": float(td_error.item()),
            "value": float(transition.value.detach().item()),
            "target": float(target.detach().item()),
            "reward": float(transition.reward),
            "scaled_reward": float(reward_tensor.item()),
            "done": transition.done,
            "batch_size": 1,
        }

    def finish_episode(self) -> dict | None:
        """
        One-step updates on every transition,
        so there is nothing to train at the end of the episode.
        """
        self.episode_counter += 1
        return self.last_metrics

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
            return value.squeeze(0)