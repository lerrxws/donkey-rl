import random
from enum import Enum

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from agents.model import DQNModel
from agents.replay_buffer import ReplayBuffer


class Action(Enum):
    NOTHING = 0
    CHANGE = 1


class DQNAgent:
    def __init__(self):
        self.training_net = DQNModel()
        self.target_net = DQNModel()
        self.target_net.load_state_dict(self.training_net.state_dict())
        self.target_net.eval()

        self.replay_buffer = ReplayBuffer()

        self.optimizer = optim.Adam(self.training_net.parameters(), lr=0.0001)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.gamma = 0.99
        self.step_counter = 0
        self.target_update_every = 500

    def select_action(self, state):
        if random.random() < self.epsilon:
            # Space is rare. Random 50/50 is bad for this game.
            return Action.CHANGE.value if random.random() < 0.15 else Action.NOTHING.value

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.training_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

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

        # scaling:
        # +0.1  -> +0.001
        # +50   -> +0.5
        # -100  -> -1.0
        rewards = torch.clamp(rewards / 100.0, -1.0, 1.0)

        q_values = self.training_net(states)
        q_values = q_values.gather(1, actions)

        with torch.no_grad():
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

        if self.step_counter % 100 == 0:
            print(
                f"train_step={self.step_counter} "
                f"loss={loss.item():.4f} "
                f"epsilon={self.epsilon:.3f}"
            )