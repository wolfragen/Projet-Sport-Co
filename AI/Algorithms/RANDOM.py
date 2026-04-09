# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:39:52 2026

@author: quent
"""

import numpy as np

class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        
    def act(self, state, train):
        return np.random.randint(self.action_dim)

    def batch_act(self, states_batch):
        return np.random.randint(self.action_dim, size=len(states_batch))