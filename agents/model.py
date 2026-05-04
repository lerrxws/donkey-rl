import torch
from torch import nn


class DQNModel(nn.Module):
    def __init__(self, state_size: int = 9, action_size: int = 2, hidden_layers: list = [64, 64]):
        super(DQNModel, self).__init__()

        layers = []
        input_size = state_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model_1 = DQNModel(hidden_layers=[64])
    model_2 = DQNModel(hidden_layers=[64, 64])
    model_3 = DQNModel(hidden_layers=[64, 64, 64])

    print(model_1)
    print(model_2)
    print(model_3)

    test_input = torch.rand(1, 7)
    print(model_2(test_input)) 