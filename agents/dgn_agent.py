import copy
import random
from enum import Enum

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.model import DQNModel
from agents.replay_buffer import ReplayBuffer

class Action(Enum):
    NOTHING=0
    CHANGE=1

class DQNAgent:
    def __init__(self):
        self.training_net=DQNModel()
        self.target_net=copy.deepcopy(self.training_net)
        self.replay_buffer=ReplayBuffer()

        self.optimizer = optim.Adam(self.training_net.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

        self.epsilon=1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma=0.99
        self.step_counter=0
    
    def select_action(self,state):
        possibility=random.random()
        action=None

        if possibility<self.epsilon:
            if random.random() < 0.15:
                action = Action(1).value
            else:
                action = Action(0).value
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            result=self.training_net(state_tensor)
            action = result.argmax().item()
        

        return action

    def remember(self,state,action,reward,next_state,done):
        self.replay_buffer.push(state,action,reward,next_state,done)
        print(f"Buffer size: {len(self.replay_buffer)}")

    
    def train_step(self):
        buffer=self.replay_buffer.sample()
        if buffer is None:
            return
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        states, actions, rewards, next_states, dones = zip(*buffer)

        states=torch.FloatTensor(np.array(states))      
        actions=torch.LongTensor(actions).unsqueeze(1)
        rewards=torch.FloatTensor(rewards).unsqueeze(1)
        next_states=torch.FloatTensor(np.array(next_states))
        dones=torch.FloatTensor(dones).unsqueeze(1)

        rewards = torch.clamp(rewards / 100.0, -1.0, 1.0)

        with torch.no_grad():
            q_next = self.target_net(next_states)  
            q_next_max = q_next.max(dim=1,keepdim=True).values
            targets = rewards + self.gamma * q_next_max * (1 - dones) 
        
        q_values=self.training_net(states)
        q_values=q_values.gather(1,actions)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.training_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        print(f"loss={loss.item():.4f}") 

        self.step_counter+=1
        if self.step_counter%1000==0:
            self.target_net.load_state_dict(self.training_net.state_dict())

        
