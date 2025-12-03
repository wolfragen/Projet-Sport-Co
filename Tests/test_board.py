# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:37:19 2025

@author: quent
"""

import pytest
from unittest.mock import MagicMock

import Engine.Entity.Board as Board

# -------------------------------------------------------------------
# Mock Settings
# -------------------------------------------------------------------
class MockSettings:
    SCREEN_OFFSET = 5
    DIM_X = 100
    DIM_Y = 50
    GOAL_LEN = 10
    WALL_ELASTICITY = 0.8
    WALL_FRICTION = 0.3
    LEFT_GOAL_COLLISION_TYPE = 2
    RIGHT_GOAL_COLLISION_TYPE = 3

# -------------------------------------------------------------------
# Patch Settings
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(Board, "Settings", MockSettings)

# -------------------------------------------------------------------
# Mock pymunk.Segment
# -------------------------------------------------------------------
class MockSegment:
    def __init__(self, body, a, b, radius):
        self.body = body
        self.a = a
        self.b = b
        self.radius = radius
        self.elasticity = None
        self.friction = None
        self.sensor = False
        self.collision_type = None
        self.color = None

# -------------------------------------------------------------------
# Test buildBoard
# -------------------------------------------------------------------
def test_buildBoard(monkeypatch):
    space_mock = MagicMock()
    space_mock.static_body = MagicMock()

    # Patch pymunk.Segment
    monkeypatch.setattr(Board.pymunk, "Segment", MockSegment)

    left_goal_pos, right_goal_pos = Board.buildBoard(space_mock)

    # Vérifier que les goals ont les bonnes positions
    assert left_goal_pos[0] == MockSettings.SCREEN_OFFSET
    assert right_goal_pos[0] == MockSettings.SCREEN_OFFSET + MockSettings.DIM_X
    assert left_goal_pos[1] == (MockSettings.DIM_Y/2 + MockSettings.SCREEN_OFFSET)
    assert right_goal_pos[1] == (MockSettings.DIM_Y/2 + MockSettings.SCREEN_OFFSET)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    