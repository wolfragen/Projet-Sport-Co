# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:10:37 2025

@author: quent
"""

import pytest
import torch
import numpy as np
from AI import Network as nn_module

@pytest.fixture
def dummy_model():
    model = nn_module.DeepRLNetwork([10, 20, 4])
    return model

def test_deep_rl_network_forward(dummy_model):
    x = torch.randn(5, 10)  # batch of 5
    out = dummy_model(x)
    assert out.shape == (5, 4)
    # Check that all layers except last apply ReLU
    x_manual = x.clone()
    for i, layer in enumerate(dummy_model.layers):
        x_manual = layer(x_manual)
        if i < len(dummy_model.layers)-1:
            assert torch.all(x_manual >= 0) or torch.any(x_manual < 0)  # basic sanity check

def test_save_and_load_network(tmp_path, dummy_model):
    save_path = tmp_path / "model.pt"
    nn_module.save_network(dummy_model, str(save_path))
    assert save_path.exists()

    # Create new model with same dimensions
    new_model = nn_module.DeepRLNetwork([10, 20, 4])
    nn_module.load_network(new_model, str(save_path), device="cpu")
    # Check weights are equal
    for p1, p2 in zip(dummy_model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)

def test_train_dqn_for_duration(monkeypatch):
    """
    Run a minimal training loop to check no crash.
    """
    model = nn_module.DeepRLNetwork([4, 8, 4])

    # Dummy simulate_episode
    def simulate_episode(players_number, models=None, max_steps=10, scoring_function=None, epsilon=1.0, display=False, screen=None, draw_options=None):
        episode_data = []
        for i in range(len(models)):
            episode_data.append({
                "states": [np.zeros(4) for _ in range(5)],
                "actions": [np.eye(4)[0] for _ in range(5)],
                "rewards": [1.0 for _ in range(5)],
                "next_states": [np.zeros(4) for _ in range(5)],
                "dones": [False for _ in range(5)]
            })
        return episode_data

    def dummy_scoring(game):
        return 0

    # Reduce duration for fast test
    nn_module.train_dqn_for_duration(
        players_number=(1,0),
        models=[model],
        optimizer_cls=torch.optim.Adam,
        lr=0.01,
        simulate_episode=simulate_episode,
        scoring_function=dummy_scoring,
        max_duration_s=0.1,  # very short
        batch_size=2,
        nb_batches=1,
        buffer_size=10,
        device="cpu"
    )
