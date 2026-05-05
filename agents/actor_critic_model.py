from collections.abc import Sequence

import torch
from torch import nn

class ActorCriticModel(nn.Module):
    def __init__(
        self,
        state_size: int = 9,
        action_size: int = 2,
        hidden_layers: list[int] | None = None,
    ):
        super().__init__()

        if state_size <= 0:
            raise ValueError("state_size must be positive")

        if action_size <= 0:
            raise ValueError("action_size must be positive")

        if hidden_layers is None:
            hidden_layers = [64, 64]

        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")

        layers = []
        input_size = state_size

        for hidden_size in hidden_layers:
            if hidden_size <= 0:
                raise ValueError("hidden layer sizes must be positive")

            layer = nn.Linear(input_size, hidden_size)
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)

            layers.append(layer)
            layers.append(nn.ReLU())

            input_size = hidden_size

        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(input_size, action_size)
        self.critic = nn.Linear(input_size, 1)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"Expected input shape [batch_size, state_size], got {tuple(x.shape)}")

        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        return logits, value