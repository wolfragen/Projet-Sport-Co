# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:03:10 2025

@author: quent
"""

import pytest
from unittest.mock import Mock
import pygame

import Settings
from Player import PlayerActions
from Engine import Actions


@pytest.fixture(autouse=True)
def setup_settings():
    # Mock settings
    Settings.PLAYER_SPEED = 5
    Settings.PLAYER_ROT_SPEED = 2


def test_process_events_quit_event_called(monkeypatch):
    """
    Test that stopGame is called when a QUIT event occurs.
    """

    game = {"selected_player": None}
    stopGame = Mock()

    # Mock pygame.event.get to return a QUIT event
    quit_event = pygame.event.Event(pygame.QUIT)
    monkeypatch.setattr(pygame, "event", Mock(get=Mock(return_value=[quit_event])))

    PlayerActions.process_events(game, stopGame)
    stopGame.assert_called_once()


def test_process_events_no_selected_player(monkeypatch):
    """
    If no player is selected, function should do nothing.
    """
    game = {"selected_player": None, "ball": None}
    stopGame = Mock()

    monkeypatch.setattr(pygame, "event", Mock(get=Mock(return_value=[])))
    monkeypatch.setattr(pygame, "key", Mock(get_pressed=Mock(return_value=[0]*512)))

    # Should not raise
    PlayerActions.process_events(game, stopGame)


def test_process_events_movement_and_shoot(monkeypatch):
    """
    Test that pressing keys calls Actions.move or Actions.shoot correctly.
    """

    # Create mock player
    player_body = Mock()
    player_shape = Mock()
    selected_player = (player_body, player_shape)

    # Create game dict
    ball = (Mock(), Mock())
    game = {"selected_player": selected_player, "ball": ball}

    stopGame = Mock()

    # Patch pygame.event.get to return no events
    monkeypatch.setattr(pygame, "event", Mock(get=Mock(return_value=[])))

    # Patch pygame.key.get_pressed to simulate keys pressed
    key_states = [0] * 512
    key_states[pygame.K_w] = 1       # Forward
    key_states[pygame.K_a] = 1       # Rotate left
    key_states[pygame.K_d] = 1       # Rotate right
    key_states[pygame.K_SPACE] = 1   # Shoot
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: key_states)

    # Patch Actions.move and Actions.shoot to track calls
    move_calls = []
    shoot_called = {"called": False}

    def fake_move(entity, speed=0, rotation_speed=0):
        move_calls.append((speed, rotation_speed))

    def fake_shoot(player, ball_arg, power=1.0):
        shoot_called["called"] = True

    monkeypatch.setattr(Actions, "move", fake_move)
    monkeypatch.setattr(Actions, "shoot", fake_shoot)

    # Call process_events
    PlayerActions.process_events(game, stopGame)

    # Check move called 3 times (forward + rotate left + rotate right)
    speeds = [call[0] for call in move_calls]
    rotations = [call[1] for call in move_calls]

    assert Settings.PLAYER_SPEED in speeds
    assert -Settings.PLAYER_ROT_SPEED in rotations
    assert Settings.PLAYER_ROT_SPEED in rotations

    # Check shoot called
    assert shoot_called["called"]
