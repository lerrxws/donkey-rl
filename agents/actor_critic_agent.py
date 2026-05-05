import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.actor_critic_model import ActorCriticModel


class MonteCarloActorCriticAgent:
    """
    Monte Carlo Actor-Critic агент.

    Ключова відмінність від Bootstrap:
        Bootstrap: target = r + gamma * V(s')   — оновлення по одному кроку
        Monte Carlo: target = G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    Чому MC краще для цієї задачі:
        - Епізоди короткі (5-15 кроків)
        - Краш завжди в кінці — потрібно щоб сигнал дійшов назад до ранніх кроків
        - Bootstrap не "бачить" що ранні кроки вели до краша
        - MC рахує повний return, тому крок перед крашем отримує ~-97,
          два кроки перед крашем ~-94, і т.д.

    Навчання відбувається після кожного епізоду (не по кроках).
    """

    def __init__(
        self,
        state_size: int = 5,
        action_size: int = 2,
        hidden_layers: list[int] | None = None,
        gamma: float = 0.97,
        lr: float = 0.0003,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        reward_scale: float = 100.0,
        max_grad_norm: float = 1.0,
        normalize_returns: bool = True,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        self.normalize_returns = normalize_returns

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCriticModel(
            state_size=state_size,
            action_size=action_size,
            hidden_layers=hidden_layers or [64, 64],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        self.episode_counter = 0
        self.step_counter = 0

        # Буфер поточного епізоду
        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._entropies: list[torch.Tensor] = []

        # Для логів
        self.last_probs: np.ndarray | None = None
        self.last_metrics: dict | None = None

    def _state_tensor(self, state) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32).flatten()
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        if len(state) != self.state_size:
            raise ValueError(
                f"Invalid state size: expected {self.state_size}, got {len(state)}"
            )

        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state) -> int:
        state_tensor = self._state_tensor(state)
        logits, value = self.model(state_tensor)

        logits = logits.squeeze(0)
        value = value.squeeze(0)

        distribution = Categorical(logits=logits)
        action = distribution.sample()

        self.last_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        self._log_probs.append(distribution.log_prob(action))
        self._values.append(value)
        self._entropies.append(distribution.entropy())

        return int(action.item())

    def remember(
        self,
        state,
        action: int,
        reward: float,
        next_state,  # не використовується в MC, але зберігаємо для сумісності
        done: bool,
    ) -> None:
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)

    def train_step(self):
        """
        Заглушка для сумісності з train.py.
        MC навчається після епізоду — викликай finish_episode().
        """
        return None

    def finish_episode(self) -> dict | None:
        """
        Повний gradient update після завершення епізоду.

        Кроки:
        1. Рахуємо discounted returns G_t для кожного кроку
        2. Нормалізуємо (опційно) — зменшує дисперсію
        3. Advantage = G_t - V(s_t)
        4. Actor loss + Critic loss + Entropy bonus
        5. Один optimizer step
        """
        if not self._rewards:
            self._clear_buffers()
            return None

        self.episode_counter += 1
        self.step_counter += len(self._rewards)

        # 1. Discounted returns
        returns = self._compute_returns(self._rewards)

        returns_tensor = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )

        # 2. Нормалізація returns
        if self.normalize_returns and len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # 3. Stack tensors
        log_probs = torch.stack(self._log_probs)
        values = torch.stack(self._values).squeeze(-1)
        entropies = torch.stack(self._entropies)

        # 4. Advantage
        advantages = returns_tensor - values.detach()

        # 5. Losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.smooth_l1_loss(values, returns_tensor)
        entropy_bonus = entropies.mean()

        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy_bonus
        )

        # 6. Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        metrics = {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "mean_value": float(values.mean().item()),
            "mean_return": float(returns_tensor.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "episode_steps": len(self._rewards),
            "total_raw_reward": sum(self._rewards),
        }

        self.last_metrics = metrics
        self._clear_buffers()

        return metrics

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

        Рахуємо з кінця епізоду назад — ефективно і просто.
        """
        returns = []
        G = 0.0

        for r in reversed(rewards):
            r_scaled = r / self.reward_scale
            r_scaled = max(-1.0, min(1.0, r_scaled))
            G = r_scaled + self.gamma * G
            returns.insert(0, G)

        return returns

    def _clear_buffers(self):
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._entropies.clear()

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
                "normalize_returns": self.normalize_returns,
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

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.gamma = checkpoint.get("gamma", self.gamma)
        self.value_coef = checkpoint.get("value_coef", self.value_coef)
        self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)
        self.reward_scale = checkpoint.get("reward_scale", self.reward_scale)
        self.max_grad_norm = checkpoint.get("max_grad_norm", self.max_grad_norm)
        self.normalize_returns = checkpoint.get("normalize_returns", self.normalize_returns)
        self.episode_counter = checkpoint.get("episode_counter", 0)
        self.step_counter = checkpoint.get("step_counter", 0)