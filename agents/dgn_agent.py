import copy
import random
from enum import Enum

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from agents.model import DQNModel
from agents.replay_buffer import ReplayBuffer

class Action(Enum):
    NOTHING=0
    CHANGE=1

class DQNAgent:
    def __init__(self):
        self.training_net=DQNModel(hidden_layers=[64, 64])
        self.target_net=copy.deepcopy(self.training_net)
        self.replay_buffer=ReplayBuffer()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.epsilon=0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma=0.99
        self.step_counter=0
    
    def selec_action(self,state):
        possibility=random.random()
        action=None

        if possibility<self.epsilon:
            action = Action(random.randint(0,1))
        else:
            result=self.training_net.forward(state)
            action = result.argmax().item()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action

    def remember(self,state,action,reward,next_state,done):
        self.replay_buffer.push(state,action,reward,next_state,done)

    
    def train_step(self):
        buffer=self.replay_buffer.sample()
        if buffer is None:
            return
        
        states, actions, rewards, next_states, dones = zip(*buffer)

        states=torch.FloatTensor(np.array(states))      
        actions=torch.LongTensor(actions)                
        rewards=torch.FloatTensor(rewards)               
        next_states=torch.FloatTensor(np.array(next_states))
        dones=torch.FloatTensor(dones)

        
