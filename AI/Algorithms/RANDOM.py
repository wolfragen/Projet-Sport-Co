# -*- coding: utf-8 -*-
"""
Simple random policy baseline.
"""

import numpy as np


class RandomAgent:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def act(self, state, train=False):
        return np.random.randint(self.action_dim)
