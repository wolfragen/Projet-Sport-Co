# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:00:06 2025

@author: quent
"""

import numpy as np
import pymunk

import Engine.Utils as Utils

# -------------------------------------------------------------------
# Mock Settings
# -------------------------------------------------------------------
class MockSettings:
    GROUND_FRICTION = 0.9
    DIM_X = 100
    SCREEN_OFFSET = 5
    PLAYER_SHOOTING_RANGE = 10

# -------------------------------------------------------------------
# Mock des objets pymunk
# -------------------------------------------------------------------
class MockBody:
    def __init__(self, position=(0,0)):
        self.position = np.array(position, dtype=float)
        self.previous_position = np.array(position, dtype=float)
        self.canShoot = True

class MockShape:
    pass

# -------------------------------------------------------------------
# Tests createSpace
# -------------------------------------------------------------------
def test_createSpace(monkeypatch):
    monkeypatch.setattr(Utils, "Settings", MockSettings)
    space = Utils.createSpace()
    assert isinstance(space, pymunk.Space)
    assert space.damping == MockSettings.GROUND_FRICTION

# -------------------------------------------------------------------
# Tests checkIfGoal
# -------------------------------------------------------------------
def test_checkIfGoal_left(monkeypatch):
    monkeypatch.setattr(Utils, "Settings", MockSettings)
    ball = (MockBody(position=(-1,0)), MockShape())
    score = [0,0]
    result = Utils.checkIfGoal(ball, score)
    assert result == True
    assert score == [0,1]

def test_checkIfGoal_right(monkeypatch):
    monkeypatch.setattr(Utils, "Settings", MockSettings)
    ball = (MockBody(position=(MockSettings.DIM_X + MockSettings.SCREEN_OFFSET + 1, 0)), MockShape())
    score = [0,0]
    result = Utils.checkIfGoal(ball, score)
    assert result == True
    assert score == [1,0]

def test_checkIfGoal_none(monkeypatch):
    monkeypatch.setattr(Utils, "Settings", MockSettings)
    ball = (MockBody(position=(50,0)), MockShape())
    score = [0,0]
    result = Utils.checkIfGoal(ball, score)
    assert result == False
    assert score == [0,0]

# -------------------------------------------------------------------
# Tests checkPlayersOut
# -------------------------------------------------------------------
def test_checkPlayersOut(monkeypatch):
    monkeypatch.setattr(Utils, "Settings", MockSettings)
    players = [
        (MockBody(position=(-5, 10)), MockShape()),
        (MockBody(position=(110, 20)), MockShape()),
        (MockBody(position=(50, 30)), MockShape())
    ]
    Utils.checkPlayersOut(players)
    # Clamp left
    assert players[0][0].position[0] == MockSettings.SCREEN_OFFSET + 10
    # Clamp right
    assert players[1][0].position[0] == MockSettings.DIM_X + MockSettings.SCREEN_OFFSET - 10
    # Within bounds
    assert players[2][0].position[0] == 50

# -------------------------------------------------------------------
# Tests checkPlayersCanShoot
# -------------------------------------------------------------------
def test_checkPlayersCanShoot(monkeypatch):
    monkeypatch.setattr(Utils, "Settings", MockSettings)

    # Mock canShoot
    def mock_canShoot(player_body, ball_body, max_distance=None):
        if max_distance is None:
            max_distance = MockSettings.PLAYER_SHOOTING_RANGE
        # Renvoie True si player x < ball x
        return player_body.position[0] < ball_body.position[0]

    monkeypatch.setattr(Utils, "canShoot", mock_canShoot)

    ball = (MockBody(position=(10,0)), MockShape())
    players = [
        (MockBody(position=(5,0)), MockShape()),
        (MockBody(position=(15,0)), MockShape())
    ]

    Utils.checkPlayersCanShoot(players, ball)
    assert players[0][0].canShoot == True
    assert players[1][0].canShoot == False
    
    
    
    
    
    
    
    
    
    
