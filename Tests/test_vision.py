# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:24:19 2025

@author: quent
"""

import pytest
import numpy as np
import math

import Engine.Vision as Vision

# -------------------------------------------------------------------
# Mock Settings
# -------------------------------------------------------------------
class MockSettings:
    ENTRY_NEURONS = 20  # Taille arbitraire pour tester..
    DIM_X = 100.0
    DIM_Y = 50.0

# -------------------------------------------------------------------
# Mock pymunk objects
# -------------------------------------------------------------------
class MockBody:
    def __init__(self, position=(0,0), angle=0.0, canShoot=True):
        self.position = np.array(position, dtype=float)
        self.angle = angle
        self.canShoot = canShoot

class MockShape:
    pass

# -------------------------------------------------------------------
# Patch Settings
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(Vision, "Settings", MockSettings)

# -------------------------------------------------------------------
# Test getVision
# -------------------------------------------------------------------
def test_getVision_basic():
    space = None  # TODO: Ignoré car rayTracing désactivé pour l'instant
    player = (MockBody(position=(10,5), angle=math.pi/2, canShoot=True), MockShape())
    ball = (MockBody(position=(50,25)), MockShape())
    left_goal_pos = (0,25)
    right_goal_pos = (100,25)

    vision = Vision.getVision(space, player, ball, left_goal_pos, right_goal_pos)
    
    # Vérifier type et taille
    assert isinstance(vision, np.ndarray)
    assert vision.shape[0] == MockSettings.ENTRY_NEURONS

    # Orientation
    assert np.isclose(vision[0], math.sin(player[0].angle))
    assert np.isclose(vision[1], math.cos(player[0].angle))

    # Positions normalisées
    expected_ball_x = (ball[0].position[0] - player[0].position[0]) / MockSettings.DIM_X
    expected_ball_y = (ball[0].position[1] - player[0].position[1]) / MockSettings.DIM_Y
    assert np.isclose(vision[2], expected_ball_x)
    assert np.isclose(vision[3], expected_ball_y)

    expected_left_goal_x = (left_goal_pos[0] - player[0].position[0]) / MockSettings.DIM_X
    expected_left_goal_y = (left_goal_pos[1] - player[0].position[1]) / MockSettings.DIM_Y
    assert np.isclose(vision[4], expected_left_goal_x)
    assert np.isclose(vision[5], expected_left_goal_y)

    expected_right_goal_x = (right_goal_pos[0] - player[0].position[0]) / MockSettings.DIM_X
    expected_right_goal_y = (right_goal_pos[1] - player[0].position[1]) / MockSettings.DIM_Y
    assert np.isclose(vision[6], expected_right_goal_x)
    assert np.isclose(vision[7], expected_right_goal_y)

    # canShoot
    assert vision[8] == True

# -------------------------------------------------------------------
# Test canShoot flag False
# -------------------------------------------------------------------
def test_getVision_canShoot_false():
    space = None
    player = (MockBody(position=(10,5), angle=0, canShoot=False), MockShape())
    ball = (MockBody(position=(50,25)), MockShape())
    left_goal_pos = (0,25)
    right_goal_pos = (100,25)

    vision = Vision.getVision(space, player, ball, left_goal_pos, right_goal_pos)

    assert vision[8] == False
    
    
