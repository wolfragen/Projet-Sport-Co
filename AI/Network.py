# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 22:21:01 2025

@author: quent
"""

import torch
import torch.nn as nn


class DeepRLNetwork(nn.Module):
    """
    Feedforward network for Deep RL.

    Parameters
    ----------
    dimensions : List[int]
        List of integers representing the number of neurons in each layer.
    """
    def __init__(self, dimensions: list[int], activation=nn.LeakyReLU):
        super().__init__()
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

        # optional: initialize weights explicitly
        #self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

# optuna ?
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    