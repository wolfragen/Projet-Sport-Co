# -*- coding: utf-8 -*-
"""
Lightweight, picklable agent proxy for multiprocessing workers.
Only carries the network state_dict — no replay buffer, optimizer, or training state.
The network is lazily reconstructed once per worker process.
"""

import torch
import numpy as np


class AgentProxy:
    """Inference-only agent that is always picklable."""

    def __init__(self, state_dict, dimensions, agent_type="dqn", device="cpu"):
        self.state_dict = state_dict
        self.dimensions = dimensions
        self.agent_type = agent_type
        self.device = device
        self._network = None

    def _build(self):
        from AI.Network import DeepRLNetwork
        if self.agent_type == "ppo":
            from torch import nn
            self._network = DeepRLNetwork(self.dimensions, last_layer=nn.Softmax(dim=1))
        else:
            self._network = DeepRLNetwork(self.dimensions)
        self._network.load_state_dict(self.state_dict)
        self._network.eval()
        self._network.to(self.device)

    @torch.no_grad()
    def act(self, state, train=False):
        if self._network is None:
            self._build()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        output = self._network(state_t)
        if self.agent_type == "ppo":
            dist = torch.distributions.Categorical(output)
            return dist.sample().item()
        return output.argmax(dim=1).item()

    @torch.no_grad()
    def batch_act(self, states_batch):
        """
        Batch inference: states_batch shape (batch_size, state_dim) → actions (batch_size,)
        """
        if self._network is None:
            self._build()
        states_t = torch.as_tensor(states_batch, dtype=torch.float32, device=self.device)
        output = self._network(states_t)
        if self.agent_type == "ppo":
            dist = torch.distributions.Categorical(output)
            return dist.sample().cpu().numpy()
        return output.argmax(dim=1).cpu().numpy()
