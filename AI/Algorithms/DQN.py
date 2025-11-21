# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:55:36 2025

@author: quent
"""

import torch
from torchrl.data import (
    ReplayBuffer,
    LazyTensorStorage,
    PrioritizedSampler,
)
from tensordict import TensorDict
import numpy as np

from AI.Network import DeepRLNetwork

class DQNAgent:
    def __init__(self, dimensions, scoring_function, reward_coeff_dict, batch_size, lr, sync_rate, buffer_size, epsilon_decay, linear_decay=True, 
                 epsilon=1.0, epsilon_min=0.05, gamma=0.99, betas=(0.9, 0.999), eps=1e-8,
                 random=False, cuda=False):
        self.batch_size = batch_size
        self.action_dim = dimensions[-1]
        self.lr = lr
        self.gamma = gamma
        self.epsilon = float(epsilon)
        self.epsilon_decay = epsilon_decay
        self.linear_decay = linear_decay
        self.epsilon_min = epsilon_min
        self.random = random
        
        self.scoring_function = scoring_function
        self.reward_coeff_dict = reward_coeff_dict

        self.device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

        # Buffer
        sampler = PrioritizedSampler(
            max_capacity=buffer_size,
            alpha=0.6,
            beta=0.4,
            eps=eps,
            dtype=torch.float32,
            reduction="mean",
            max_priority_within_buffer=1.0,
        )
        self.memory = ReplayBuffer(
            storage=LazyTensorStorage(buffer_size),
            sampler=sampler,
            pin_memory=False,
            prefetch=0,
        )

        # Networks
        self.onlineNetwork = DeepRLNetwork(dimensions).to(self.device)
        self.targetNetwork = DeepRLNetwork(dimensions).to(self.device)
        self.targetNetwork.eval()

        self.sync_rate = sync_rate
        self.sync_value = 0

        # Optimizer & Loss
        self.optimizer = torch.optim.Adam(self.onlineNetwork.parameters(), lr=lr, betas=betas, eps=eps)
        self.loss_function = torch.nn.MSELoss()

    @torch.no_grad()
    def act(self, state, train=True):
        if self.random or (train and np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.onlineNetwork(state_tensor)
        return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done, td_priority=1.0):
        transition = TensorDict(
            {
                "state": torch.tensor(state, dtype=torch.float32, device=self.device),
                "action": torch.tensor(action, dtype=torch.long, device=self.device),
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
                "next_state": torch.tensor(next_state, dtype=torch.float32, device=self.device),
                "done": torch.tensor(done, dtype=torch.float32, device=self.device),
            },
            batch_size=[]
        )
        self.memory.add(transition)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        states = batch["state"]
        actions = batch["action"].long().unsqueeze(1)
        rewards = batch["reward"].unsqueeze(1)
        next_states = batch["next_state"]
        dones = batch["done"].unsqueeze(1)
        weights = batch.get("weights", torch.ones(self.batch_size, 1, device=self.device))
        indices = batch.get("idx", None)

        # Q-learning target
        predicted_q = self.onlineNetwork(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.targetNetwork(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * (1 - dones) * next_q_values

        td_errors = predicted_q - target_q
        loss = (weights * td_errors.pow(2)).mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineNetwork.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        if indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().abs())

        # Target network sync
        self.sync_value += 1
        if self.sync_value >= self.sync_rate:
            self.sync_value = 0
            self.syncTargetNetwork()

    def decayEpsilon(self):
        if self.linear_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def syncTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
        self.targetNetwork.eval()

    def save(self, path):
        torch.save(self.onlineNetwork.state_dict(), path)

    def load(self, path):
        self.onlineNetwork.load_state_dict(torch.load(path, map_location=self.device))
        self.syncTargetNetwork()
        self.random = False


def getRandomDQNAgents(n, dimensions, scoring_function, reward_coeff_dict, batch_size=128, lr=3e-4, sync_rate=1000, buffer_size=50_000, 
                       epsilon_decay=0.99995, linear_decay=True, epsilon=1.0, epsilon_min=0.05, gamma=0.99, 
                       betas=(0.9, 0.999), eps=1e-8, cuda=False):
    agents = []
    for i in range(n):
        agents.append(DQNAgent(dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                               epsilon_decay=epsilon_decay, linear_decay=linear_decay, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma, 
                               scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, betas=betas, eps=eps, random=True, cuda=cuda))
    return agents










































