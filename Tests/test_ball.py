# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:35:09 2025

@author: quent
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

import Engine.Entity.Ball as Ball

# -------------------------------------------------------------------
# Mock Settings
# -------------------------------------------------------------------
class MockSettings:
    BALL_RADIUS = 2.0
    BALL_MASS = 1.0
    DIM_X = 100
    DIM_Y = 50
    SCREEN_OFFSET = 5
    BALL_ELASTICITY = 0.8
    BALL_FRICTION = 0.3
    BALL_COLOR = (255, 0, 0, 255)
    RANDOM_BALL_POSITION = False

# -------------------------------------------------------------------
# Patch Settings
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(Ball, "Settings", MockSettings)

# -------------------------------------------------------------------
# Test buildBall with RANDOM_BALL_POSITION = False (center)
# -------------------------------------------------------------------
def test_buildBall_center(monkeypatch):
    space_mock = MagicMock()

    # Patch pymunk.Body and Circle to MagicMock
    monkeypatch.setattr(Ball.pymunk, "Body", lambda mass, moment: MagicMock())
    monkeypatch.setattr(Ball.pymunk, "Circle", lambda body, radius: MagicMock())

    ball_tuple = Ball.buildBall(space_mock)
    body, shape = ball_tuple

    # Vérifier position au centre
    expected_pos = (MockSettings.SCREEN_OFFSET + MockSettings.DIM_X / 2,
                    MockSettings.SCREEN_OFFSET + MockSettings.DIM_Y / 2)
    np.testing.assert_array_equal(body.position, expected_pos)

    # Vérifier que shape.is_ball = True
    assert hasattr(shape, "is_ball")
    assert shape.is_ball == True

    # Vérifier que la balle a été ajoutée à l'espace
    space_mock.add.assert_called_once_with(body, shape)

# -------------------------------------------------------------------
# Test buildBall with RANDOM_BALL_POSITION = True
# -------------------------------------------------------------------
def test_buildBall_random(monkeypatch):
    space_mock = MagicMock()
    monkeypatch.setattr(Ball, "Settings", type("RandomSettings", (), dict(vars(MockSettings), RANDOM_BALL_POSITION=True))())

    monkeypatch.setattr(Ball.pymunk, "Body", lambda mass, moment: MagicMock())
    monkeypatch.setattr(Ball.pymunk, "Circle", lambda body, radius: MagicMock())

    ball_tuple = Ball.buildBall(space_mock)
    body, shape = ball_tuple

    # Vérifier que la position est dans les bornes
    x, y = body.position
    offset = MockSettings.SCREEN_OFFSET
    dim_x = MockSettings.DIM_X
    dim_y = MockSettings.DIM_Y
    assert offset + dim_x/10 <= x <= offset + dim_x*9/10
    assert offset + dim_y/10 <= y <= offset + dim_y*9/10
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    