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
    """
    DQN (Deep Q-Network Agent) for deep reinforcement learning

    Use two neural network:
        - The "Online" network, for learning and decision-making 
        - The "Target" network, returning stable learning targets to train the Online network. It has the same architecture 
        that the Online network and his weights are periodically remplaced by the weight of the Online network.


    Args:
        dimensions (tuple[int, int, int, int]): Dimensions of the neural networks
        batch_size (int): The size of the batch use for training
        lr (float): The learning rate of the training
        sync_rate (int): The number of training to do before updating the Target Network with the values of the Online Network
        buffer_size (int): Define the maximum size of the memory use for the training
        epsilon_decay (float): Define the rate at which epsilon decay
        linear_decay (bool): If True, substract the epsilon decay to the value of epsilon. If False, multiply the epsilon decay to the value of epsilon
        epsilon (float): The probability for the agent to pick a action randomly. Decay during training 
        epsilon_min (float): The minimum value of epsilon
        gamma (float): The discount factor: Determine the importance of future rewards compared to the actual reward
        betas: (tuple[float, float]): Hyperparameters for the Adam optimizer
        eps (float): Hyperparameter for the Adam optimizer
        random (bool): if True, the agent will only choose actions randomly
        cuda (bool): If True, the training function will use cuda to accelerate the learning process
    """
    def __init__(self, dimensions: tuple, batch_size: int, lr: float, sync_rate: int, buffer_size: int, epsilon_decay: float, linear_decay: bool=True, 
                 epsilon: float=1.0, epsilon_min: float=0.05, gamma: float=0.99, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8,
                 random: bool=False, cuda: bool=False):
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
        
        self.device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")
        self.cuda = self.device.type == "cuda"

        self.onlineNetwork = DeepRLNetwork(dimensions).to(self.device)
        self.targetNetwork = DeepRLNetwork(dimensions).to(self.device)
        self.targetNetwork.eval()
        
        self.sync_rate = sync_rate
        self.sync_value = 0
        
        self.optimizer = optim.Adam(
            self.onlineNetwork.parameters(), 
            lr=lr, betas=betas, eps=eps)
        self.loss_function = torch.nn.MSELoss()
        
        self.random = random
        
    @torch.no_grad()
    def act(self, state: np.ndarray, train: bool=True) -> int:
        """Choose an action depending on the actual state. Have a probability epsilon to choose an action randomly.
        If self.random is True, always choose randomly

        Args:
            state (np.ndarray): The state of the game 
            train (bool, optional): Is True if the method is used during training. Defaults to True.

        Returns:
            int: The id of the action chosen by the agent
        """
        if self.random or (train and np.random.rand() <= self.epsilon):
            return np.random.choice(self.action_dim)
        
        q_values = self.onlineNetwork(torch.tensor(state, dtype=torch.float32, device=self.device))
        return torch.argmax(q_values).item()

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store a transition in the memory buffer of the model

        Args:
            state (np.ndarray): The current state of the game
            action (int): The action chosen by the agent
            reward (float): The reward for the action
            next_state (np.ndarray): The next state of the game
            done (bool): If True, the episode is over 
        """
        self.memory.append((state, action, reward, next_state, done))

    
    def replay(self):
        """Execute a training phase for randomly chosen transitions from the memory buffer of the model.

        Update the weights of the Online network by backpropagating the MSE loss between the predicted Q-values and the target Q-values.

        Update periodically the weights of the Target network, by replacing them with the weights of the Online network
        """
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        
        ## convert to torch tensors ##
        observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        ##############################
        
        predicted_q = self.onlineNetwork(observations).gather(1, actions)
        with torch.no_grad():
            next_q_values, _ = self.targetNetwork(next_observations).max(dim=1, keepdim=True)
            target_q = rewards + self.gamma * (1 - dones) * next_q_values

        self.loss = self.loss_function(predicted_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineNetwork.parameters(), max_norm=1.0) # Gradient Clipping
        self.optimizer.step()
        
        self.sync_value += 1
        if(self.sync_value >= self.sync_rate):
            self.sync_value = 0
            self.syncTargetNetwork()
    
    def decayEpsilon(self):
        """Decrease the value of epsilon, linearly or exponentially. The value of epsilon cannot be inferior to epsilon_min
        """
        if(self.linear_decay):
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        
    def syncTargetNetwork(self):
        """Replace the weights of the Target network with the weights of the Online network
        """
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
        self.targetNetwork.eval() # no gradient for target network, it's not "learning"
        
    def save(self, path: str):
        """Save the weights of the Online network in a file

        Args:
            path (str): The path of the save file
        """
        self.onlineNetwork.save(path)
        
    def load(self, path: str):
        """Load the weights of the Online network from a file

        Args:
            path (str): The path of the file
        """
        self.onlineNetwork.load(path, device=self.device)
        self.syncTargetNetwork()
        self.random = False


def getRandomDQNAgents(n: int, dimensions: tuple[int], batch_size: int=128, lr: float=3e-4, sync_rate: int=1000, buffer_size: int=50_000, 
                       epsilon_decay: float=0.99995, linear_decay: bool=True, epsilon: float=1.0, epsilon_min: float=0.05, gamma: float=0.99, 
                       betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8, cuda: bool=False) -> list[DQNAgent]:
    """Create a list of n DQNAgent initialized randomly

    Args:
        n (int): The number of DQN Agents to create
        dimensions (tuple[int]): Dimensions of the neural networks
        batch_size (int, optional): The size of the batch use for training. Defaults to 128.
        lr (float, optional): The learning rate of the training. Defaults to 3e-4.
        sync_rate (int, optional): The number of training to do before updating the Target Network with the values of the Online Network. Defaults to 1000.
        buffer_size (int, optional): Define the maximum size of the memory use for the training. Defaults to 50_000.
        epsilon_decay (float, optional): Define the rate at which epsilon decay. Defaults to 0.99995.
        linear_decay (bool, optional): If True, substract the epsilon decay to the value of epsilon. If False, multiply the epsilon decay to the value of epsilon. Defaults to True.
        epsilon (float, optional): The probability for the agent to pick a action randomly.. Defaults to 1.0.
        epsilon_min (float, optional): The minimum value of epsilon. Defaults to 0.05.
        gamma (float, optional): The discount factor: Determine the importance of future rewards compared to the actual reward. Defaults to 0.99.
        betas (tuple[float, float], optional): Hyperparameters for the Adam optimizer. Defaults to (0.9, 0.999).
        eps (float, optional): Hyperparameter for the Adam optimizer. Defaults to 1e-8.
        cuda (bool, optional): If True, the training function will use cuda to accelerate the learning process. Defaults to False.

    Returns:
        list[DQNAgent]: The list of DQNAgent
    """
    agents = []
    for i in range(n):
        agents.append(DQNAgent(dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                               epsilon_decay=epsilon_decay, linear_decay=linear_decay, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma, 
                               betas=betas, eps=eps, random=True, cuda=cuda))
    return agents



















