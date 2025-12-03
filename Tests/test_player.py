# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:42:28 2025

@author: quent
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

import Engine.Entity.Player as Player

# -------------------------------------------------------------------
# Mock Settings
# -------------------------------------------------------------------
class MockSettings:
    SCREEN_OFFSET = 5
    DIM_X = 100
    DIM_Y = 50
    PLAYER_LEN = 2
    PLAYER_MASS = 1.0
    PLAYER_ELASTICITY = 0.8
    PLAYER_FRICTION = 0.3
    PLAYER_LEFT_COLOR = (0,0,255,255)
    PLAYER_RIGHT_COLOR = (255,0,0,255)

# -------------------------------------------------------------------
# Patch Settings
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(Player, "Settings", MockSettings)

# -------------------------------------------------------------------
# Mock pymunk objects
# -------------------------------------------------------------------
class MockBody:
    def __init__(self):
        self.position = None
        self.previous_position = None
        self.angle = 0.0

class MockPoly:
    @staticmethod
    def create_box(body, size):
        shape = MagicMock()
        return shape

# -------------------------------------------------------------------
# Test buildPlayers basic
# -------------------------------------------------------------------
def test_buildPlayers(monkeypatch):
    space_mock = MagicMock()

    monkeypatch.setattr(Player.pymunk, "Body", lambda mass, moment: MockBody())
    monkeypatch.setattr(Player.pymunk.Poly, "create_box", MockPoly.create_box)
    monkeypatch.setattr(Player.np, "linspace", np.linspace)  # ensure np is used in spacing

    players_number = [2, 2]
    players, left_players, right_players, selected = Player.buildPlayers(space_mock, players_number, human=True)

    # Vérifier nombres de joueurs
    assert len(players) == 4
    assert len(left_players) == 2
    assert len(right_players) == 2

    # Vérifier selected_player
    assert selected == 0

    # Vérifier que chaque joueur a les attributs
    for body, shape in left_players + right_players:
        assert hasattr(shape, "is_player")
        assert shape.is_player == True
        assert hasattr(shape, "left_team")
        assert shape.left_team in [True, False]
















