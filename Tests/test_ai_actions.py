# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:07:31 2025

@author: quent
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import pymunk

import Settings
from AI import AIActions
from Engine import Actions
from Engine import Vision


@pytest.fixture(autouse=True)
def setup_settings():
    Settings.PLAYER_SPEED = 5
    Settings.PLAYER_ROT_SPEED = 2
    Settings.PLAYER_SHOOTING_RANGE = 10


def test_random_play_shape_and_bounds():
    """
    Test that random_play returns a 4-element array in [0,1].
    """
    player = (Mock(), Mock())
    game = {}
    arr = AIActions.random_play(game, player)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4,)
    assert np.all(arr >= 0) and np.all(arr <= 1)


@patch.object(Vision, "getVision")
@patch.object(Actions, "move")
@patch.object(Actions, "shoot")
@patch.object(Actions, "canShoot")
def test_play_action_selection(mock_canShoot, mock_shoot, mock_move, mock_getVision):
    """
    Test that play() chooses the correct action based on vision and canShoot.
    """

    # Mocks
    mock_getVision.return_value = np.zeros(10)  # dummy vision
    mock_canShoot.return_value = True

    # Mock player and ball
    player_body = Mock()
    player_shape = Mock()
    player = (player_body, player_shape)
    ball_body = Mock()
    ball_shape = Mock()
    game = {"ball": (ball_body, ball_shape)}

    # Force actions_array to choose move forward
    with patch("numpy.argmax", return_value=0):
        arr = AIActions.play(game, player)
        mock_move.assert_called_with(player, speed=Settings.PLAYER_SPEED)
        assert isinstance(arr, np.ndarray)

    # Force actions_array to choose rotate left
    with patch("numpy.argmax", return_value=1):
        arr = AIActions.play(game, player)
        mock_move.assert_called_with(player, rotation_speed=-Settings.PLAYER_ROT_SPEED)

    # Force actions_array to choose rotate right
    with patch("numpy.argmax", return_value=2):
        arr = AIActions.play(game, player)
        mock_move.assert_called_with(player, rotation_speed=Settings.PLAYER_ROT_SPEED)

    # Force actions_array to choose shoot
    with patch("numpy.argmax", return_value=3):
        arr = AIActions.play(game, player)
        mock_shoot.assert_called_with(player, game["ball"])


def test_play_with_cannot_shoot_minimizes_shoot_value():
    """
    Test that if canShoot returns False, the shoot action (index 3) is minimized.
    """
    # Mock player with real attributes
    player_body = pymunk.Body(1, 1)
    player_body.position = (0, 0)
    player_body.angle = 0.0
    player_shape = pymunk.Circle(player_body, radius=5)
    player = (player_body, player_shape)
    
    ball_body = pymunk.Body(1, 1)
    ball_body.position = (5, 0)
    ball_shape = pymunk.Circle(ball_body, radius=5)
    game = {"ball": (ball_body, ball_shape)}

    with patch.object(Vision, "getVision", return_value=np.zeros(10)):
        with patch.object(Actions, "canShoot", return_value=False):
            arr = AIActions.play(game, player)
            # Check index 3 has the minimal value
            assert arr[3] == np.min(arr)
