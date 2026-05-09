import random
from enum import Enum

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from src.agents.q_learning.model import DQNModel
from src.agents.q_learning.replay_buffer import ReplayBuffer


class Action(Enum):
    NOTHING = 0
    CHANGE = 1


class DQNAgent:
    def __init__(
            self,
            state_size:int,
            action_size:int,
            hidden_layers: list[int] | None = None,
            flag_double=False
        ):
        self.flag_double=flag_double
        if flag_double:
            print("Init Double DQN agent...")
        else:   
            print("Init DQN agent...")
        self.training_net = DQNModel(state_size=state_size,action_size=action_size,hidden_layers=hidden_layers)
        self.target_net = DQNModel(state_size=state_size,action_size=action_size,hidden_layers=hidden_layers)
        self.target_net.load_state_dict(self.training_net.state_dict())
        self.target_net.eval()

        self.replay_buffer = ReplayBuffer()

        self.optimizer = optim.Adam(self.training_net.parameters(), lr=0.0001)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        self.gamma = 0.99
        self.step_counter = 0
        self.target_update_every = 500

    def select_action(self, state):
        if random.random() < self.epsilon:
            return Action.CHANGE.value if random.random() < 0.15 else Action.NOTHING.value

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.training_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        return None

    def train_step(self):
        batch = self.replay_buffer.sample()

        if batch is None:
            return

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        rewards = rewards / 100.0

        q_values = self.training_net(states)
        q_values = q_values.gather(1, actions)

        with torch.no_grad():
            if self.flag_double:
                next_actions = self.training_net(next_states).argmax(dim=1, keepdim=True)
                q_next = self.target_net(next_states).gather(1, next_actions)
                targets = rewards + self.gamma * q_next * (1.0 - dones)
            else:
                q_next = self.target_net(next_states)
                q_next_max = q_next.max(dim=1, keepdim=True).values
                targets = rewards + self.gamma * q_next_max * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.training_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_counter += 1

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        if self.step_counter % self.target_update_every == 0:
            self.target_net.load_state_dict(self.training_net.state_dict())

        if self.step_counter % 1000 == 0:
            print(
                f"train_step={self.step_counter} "
                f"loss={loss.item():.4f} "
                f"epsilon={self.epsilon:.3f}"
            )