# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:55:36 2025

@author: quent
"""

import random
import torch
import torch.optim as optim
import numpy as np
from collections import deque

from AI.Network import DeepRLNetwork

class DQNAgent:
    def __init__(self, dimensions, batch_size, lr, sync_rate, buffer_size, epsilon_decay, linear_decay=True, 
                 epsilon=1.0, epsilon_min=0.05, gamma=0.99, betas=(0.9, 0.999), eps=1e-8,
                 random=False):
        self.batch_size = batch_size
        self.action_dim = dimensions[-1]
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.linear_decay = linear_decay
        self.epsilon_min = epsilon_min
        
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        
        self.onlineNetwork = DeepRLNetwork(dimensions)
        self.targetNetwork = DeepRLNetwork(dimensions)
        
        self.sync_rate = sync_rate
        self.sync_value = 0
        
        self.optimizer = optim.Adam(self.onlineNetwork.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        self.loss_function = torch.nn.MSELoss()
        
        self.random = random

    def act(self, state, train=True):
        if self.random or (train and np.random.rand() <= self.epsilon):
            return np.random.choice(self.action_dim)
        
        q_values = self.onlineNetwork(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    
    def replay(self):
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        
        ## convert to torch tensors ##
        observations = torch.tensor(np.array(observations), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        ##############################
        
        predicted_q = self.onlineNetwork(observations).gather(1, actions)
        with torch.no_grad():
            next_q_values, _ = self.targetNetwork(next_observations).max(dim=1, keepdim=True)
            target_q = rewards + self.gamma * (1 - dones) * next_q_values

        self.loss = self.loss_function(predicted_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        self.sync_value += 1
        if(self.sync_value >= self.sync_rate):
            self.sync_value = 0
            self.syncTargetNetwork()
    
    def decayEpsilon(self):
        if(self.linear_decay):
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        
    def syncTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
        self.targetNetwork.eval() # no gradient for target network, it's not "learning"
        
    def save(self, path):
        self.onlineNetwork.save(path)
        
    def load(self, path, device="cpu"):
        self.onlineNetwork.load(path, device)
        self.syncTargetNetwork()
        self.random = False


def getRandomDQNAgents(n, dimensions, batch_size=128, lr=3e-4, sync_rate=1000, buffer_size=50_000, 
                       epsilon_decay=0.99995, linear_decay=True, epsilon=1.0, epsilon_min=0.05, gamma=0.99, 
                       betas=(0.9, 0.999), eps=1e-8):
    agents = []
    for i in range(n):
        agents.append(DQNAgent(dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                               epsilon_decay=epsilon_decay, linear_decay=linear_decay, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma, 
                               betas=betas, eps=eps, random=True))
    return agents



















