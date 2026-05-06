import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
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
        self.value_head = nn.Linear(input_size, 1)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        value = self.value_head(features).squeeze(-1)
        return value