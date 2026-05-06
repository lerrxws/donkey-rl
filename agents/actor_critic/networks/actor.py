import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: list[int] | None = None,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64]

        layers = []
        input_size = state_size

        for hidden_size in hidden_layers:
            layer = nn.Linear(input_size, hidden_size)
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)

            layers.append(layer)
            layers.append(nn.ReLU())

            input_size = hidden_size

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_size, action_size)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.policy_head(features)
        return logits