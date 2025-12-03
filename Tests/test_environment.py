# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:56:08 2025

@author: quent
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

import Engine.Environment as Env

# -------------------------------------------------------------------
# Mock Settings
# -------------------------------------------------------------------
class MockSettings:
    DELTA_TIME = 2
    MAX_FPS = 60

# -------------------------------------------------------------------
# Patch Settings
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(Env, "Settings", MockSettings)

# -------------------------------------------------------------------
# Fixtures pour dépendances
# -------------------------------------------------------------------
@pytest.fixture
def mock_dependencies(monkeypatch):
    # Mock functions et classes
    monkeypatch.setattr(Env, "createSpace", lambda: MagicMock())
    monkeypatch.setattr(Env, "buildBoard", lambda space: ((0,0),(100,50)))
    monkeypatch.setattr(Env, "buildBall", lambda space: (MagicMock(), MagicMock()))
    monkeypatch.setattr(Env, "buildPlayers", lambda space, num, human=False: (
        [(MagicMock(), MagicMock()) for _ in range(sum(num))],
        [(MagicMock(), MagicMock()) for _ in range(num[0])],
        [(MagicMock(), MagicMock()) for _ in range(num[1])],
        0 if human else None
    ))
    monkeypatch.setattr(Env, "define_previous_pos", lambda players, ball: None)
    monkeypatch.setattr(Env, "checkPlayersCanShoot", lambda players, ball: None)
    monkeypatch.setattr(Env, "reset_movements", lambda players: None)
    monkeypatch.setattr(Env, "checkPlayersOut", lambda players: None)
    monkeypatch.setattr(Env, "checkIfGoal", lambda ball, score: False)
    monkeypatch.setattr(Env, "getVision", lambda space, player, ball, left, right: np.zeros(20))
    monkeypatch.setattr(Env, "play", lambda player, ball, action: action)
    monkeypatch.setattr(Env, "process_events", lambda: (False, -1))
    return monkeypatch

# -------------------------------------------------------------------
# Test initialisation
# -------------------------------------------------------------------
def test_environment_init(mock_dependencies):
    agents = [MagicMock() for _ in range(4)]
    env = Env.LearningEnvironment([2,2], agents, display=False, human=True)
    
    assert len(env.players) == 4
    assert len(env.players_left) == 2
    assert len(env.players_right) == 2
    assert env.selected_player == 0 if env.human else None
    assert env.agents == agents

# -------------------------------------------------------------------
# Test step
# -------------------------------------------------------------------
def test_step_calls(monkeypatch, mock_dependencies):
    agents = [MagicMock() for _ in range(2)]
    env = Env.LearningEnvironment([1,1], agents, display=False)
    
    rewards = env.step(human_events=False)
    # Vérifier type de retour
    assert isinstance(rewards, list)
    # Les rewards doivent appeler scoring_function de chaque agent
    for r, agent in zip(rewards, agents):
        agent.scoring_function.assert_called_once()

# -------------------------------------------------------------------
# Test playerAct
# -------------------------------------------------------------------
def test_playerAct(monkeypatch, mock_dependencies):
    agents = [MagicMock() for _ in range(1)]
    env = Env.LearningEnvironment([1,0], agents, display=False)
    
    action_result = env.playerAct(0, 5)
    assert action_result == 5

# -------------------------------------------------------------------
# Test getState
# -------------------------------------------------------------------
def test_getState(monkeypatch, mock_dependencies):
    agents = [MagicMock() for _ in range(1)]
    env = Env.LearningEnvironment([1,0], agents, display=False)
    
    state = env.getState(0)
    assert state.shape[0] == 20  # Mocked return

# -------------------------------------------------------------------
# Test isDone
# -------------------------------------------------------------------
def test_isDone(mock_dependencies):
    agents = [MagicMock() for _ in range(1)]
    env = Env.LearningEnvironment([1,0], agents, display=False)
    env.done = True
    assert env.isDone() == True
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    