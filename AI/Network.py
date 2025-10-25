# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 22:21:01 2025

@author: quent
"""

import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy
import pygame
import math


class DeepRLNetwork(nn.Module):
    """
    Fully connected feedforward network for Deep RL, adaptable to any layer dimensions.

    Parameters
    ----------
    dimensions : List[int]
        List of integers representing the number of neurons in each layer.
    """
    def __init__(self, dimensions: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation to all except last layer
                x = F.relu(x)
        return x


def save_network(network: DeepRLNetwork, path: str):
    """
    Save the network weights to a file.

    Parameters
    ----------
    network : DeepRLNetwork
        The network to save.
    path : str
        File path to save the network weights.
    """
    torch.save(network.state_dict(), path)
    return


def load_network(network: DeepRLNetwork, path: str, device: str = "cpu"):
    """
    Load network weights from a file.

    Parameters
    ----------
    network : DeepRLNetwork
        The network instance (with same architecture) to load weights into.
    path : str
        Path to the saved network weights.
    device : str
        Device to load the network on ("cpu" or "cuda").
    """
    network.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    network.to(device)
    return network


# optuna ?

def train_dqn_for_duration(
    players_number: tuple[int,int],
    models: list[torch.nn.Module],
    optimizer_cls: torch.optim.Optimizer,
    lr: float,
    simulate_episode: callable,
    scoring_function: callable,
    loss_fn: torch.nn.Module = torch.nn.MSELoss(),
    max_duration_s: int = 300,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.05,
    epsilon_decay: int = 10_000,
    initial_calls: int = 100,
    max_calls: int = 10000,
    steps_growth_rate: int = 1.01,
    delay_params: int = 20,
    batch_size: int = 64,
    min_batch_size: int = 32,
    nb_batches: int = 10,
    buffer_size: int = 50_000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cpu_sync_interval: int = 20,
    display: bool = False,
    start_display: callable=None,
) -> None:
    """
    Train a Deep Q-Network using experience replay for a given duration.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network that estimates Q-values.
    optimizer : torch.optim.Optimizer
        The optimizer for the model.
    simulate_episode : callable
        A function returning experience data as a dict with keys:
            - 'states'
            - 'actions'
            - 'rewards'
            - 'next_states'
    loss_fn : torch.nn.Module, optional
        Loss function, default is Mean Squared Error (MSELoss).
    max_duration_s : int, optional
        Maximum training duration in seconds.
    gamma : float, optional
        Discount factor for future rewards.
    batch_size : int, optional
        Number of samples per training batch.
    buffer_size : int, optional
        Maximum number of transitions stored in the replay buffer.
    device : str, optional
        Device to use ('cuda' or 'cpu').

    Returns
    -------
    None
        Trains the model in-place.
    """
    
    assert len(models) == players_number[0]+players_number[1]
    
    learning_models = []
    
    # Load models to GPU/CPU
    for model in models:
        model.to("cpu")
        
    if(device == "cuda"):
        for model in models:
            new_model = copy.deepcopy(model).to("cuda")
            learning_models.append(new_model)
    else:
        for model in models:
            new_model = copy.deepcopy(model).to("cpu")
            learning_models.append(new_model)
        
    replay_buffers = [deque(maxlen=buffer_size) for _ in range(len(models))]
    optimizers = [optimizer_cls(model.parameters(), lr=lr) for model in models]
    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_final) / epsilon_decay
    
    calls_per_game = initial_calls

    start_time = time.time()
    steps = 0
    
    temp_batch_size = batch_size
    
    screen = None
    draw_options = None
    if display:
        screen, draw_options = start_display()

    print(f"Starting training on {device.upper()} for {max_duration_s} seconds...\n")

    while (time.time() - start_time) < max_duration_s:
        # Simulate one full episode
        episode_data = simulate_episode(players_number, models=models, max_steps=calls_per_game, 
                        scoring_function=scoring_function, epsilon=epsilon,
                        display=display, screen=screen, draw_options = draw_options)
        
        for player_id in range(len(models)):
            player_data = episode_data[player_id]
            replay_buffer = replay_buffers[player_id]
            optimizer = optimizers[player_id]
            
            states = player_data["states"]
            actions = player_data["actions"]
            rewards = player_data["rewards"]
            next_states = player_data["next_states"]
            dones = player_data["dones"]
            
            # Store transitions
            for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                replay_buffer.append((s, a.argmax().item(), r, ns, d))
            
            # Skip until we have enough samples
            if len(replay_buffer) < batch_size*nb_batches:
                if(math.floor(len(replay_buffer)/nb_batches) < min_batch_size): continue
                temp_batch_size = math.floor(len(replay_buffer)/nb_batches)
            
            for i in range(nb_batches):
                # Sample a random minibatch
                batch = random.sample(replay_buffer, temp_batch_size)
                s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)
                
                # Convert to tensors
                s_batch = torch.tensor(np.array(s_batch), dtype=torch.float32, device=device)
                a_batch = torch.tensor(np.array(a_batch), dtype=torch.int64, device=device).unsqueeze(1)
                r_batch = torch.tensor(np.array(r_batch), dtype=torch.float32, device=device).unsqueeze(1)
                ns_batch = torch.tensor(np.array(ns_batch), dtype=torch.float32, device=device)
                d_batch = torch.tensor(d_batch, dtype=torch.float32, device=device).unsqueeze(1)
                
                learning_model = learning_models[player_id]
                target_model = models[player_id]
                
                # Forward passes
                q_values = learning_model(s_batch)
                with torch.no_grad():
                    next_actions = learning_model(ns_batch).argmax(dim=1, keepdim=True)  # shape [batch,1]
                    next_q_values = target_model(ns_batch.cpu()).gather(1, next_actions.cpu())    # shape [batch,1]
                    if(device == "cuda"):
                        next_q_values = next_q_values.cuda()
                    target_q = r_batch + gamma * (1 - d_batch) * next_q_values
        
                # Gather Q-values corresponding to chosen actions
                predicted_q = q_values.gather(1, a_batch)
        
                # Compute loss
                loss = loss_fn(predicted_q, target_q)
        
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        steps += 1
        temp_batch_size = batch_size
        if(steps > delay_params):
            epsilon = max(epsilon_final, epsilon - epsilon_decay_rate)
            calls_per_game = round(min(initial_calls * (steps_growth_rate ** (steps-delay_params)), max_calls))
            
        if(steps % cpu_sync_interval == 0):
            for model_id in range(len(models)):
                models[model_id].load_state_dict(learning_models[model_id].state_dict())
        
        if steps % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Steps: {steps}, Loss: {loss.item():.8f}")

    print("\nTraining finished.")
    
    if display:
        pygame.quit()
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    