# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:50:19 2025

@author: quent
"""

import math
import numpy as np
import pymunk

import Settings
from Engine.Actions import reset_movements, move, shoot, canShoot


def test_move_and_reset_movements_basic():
    """
    Test basic move and reset_movements behavior for players.
    """

    # --- 1. Mock Settings ---
    Settings.PLAYER_LEN = 10
    Settings.PLAYER_SHOOTING_RANGE = 5
    Settings.SHOOTING_SPEED = 10
    Settings.BALL_RADIUS = 1.0

    # --- 2. Create a minimal player and ball ---
    player_body = pymunk.Body(1, 1)
    player_body.position = (0, 0)
    player_body.angle = 0
    player_shape = pymunk.Circle(player_body, radius=5)

    player = (player_body, player_shape)

    # --- 3. Test move ---
    move(player, speed=5, rotation_speed=1)
    expected_vx = 5 * math.cos(player_body.angle)
    expected_vy = 5 * math.sin(player_body.angle)
    assert math.isclose(player_body.velocity.x, expected_vx, rel_tol=1e-6)
    assert math.isclose(player_body.velocity.y, expected_vy, rel_tol=1e-6)
    assert math.isclose(player_body.angular_velocity, 1, rel_tol=1e-6)
    assert player_body.previous_position == player_body.position
    assert player_body.previous_angle == player_body.angle

    # --- 4. Test reset_movements ---
    game = {"players": [player]}
    reset_movements(game)
    assert math.isclose(player_body.velocity.x, 0, rel_tol=1e-6)
    assert math.isclose(player_body.velocity.y, 0, rel_tol=1e-6)
    assert math.isclose(player_body.angular_velocity, 0, rel_tol=1e-6)


def test_canShoot_behavior():
    """
    Test canShoot with different distances between player and ball.
    """
    Settings.PLAYER_LEN = 10
    Settings.PLAYER_SHOOTING_RANGE = 5
    Settings.BALL_RADIUS = 1.0

    player_body = pymunk.Body(1, 1)
    player_body.position = (0, 0)
    player_body.angle = 0

    ball_body = pymunk.Body(1, 1)

    # Ball too far
    ball_body.position = (20, 0)
    assert not canShoot(player_body, ball_body)

    # Ball in range
    ball_body.position = (Settings.PLAYER_LEN/2 + Settings.BALL_RADIUS + 3, 0)
    assert canShoot(player_body, ball_body)

    # Ball just at the edge
    ball_body.position = (Settings.PLAYER_LEN/2 + Settings.BALL_RADIUS + Settings.PLAYER_SHOOTING_RANGE, 0)
    assert canShoot(player_body, ball_body)


def test_shoot_moves_ball(monkeypatch):
    """
    Test shoot() applies velocity to the ball correctly if canShoot returns True.
    """

    Settings.PLAYER_LEN = 10
    Settings.PLAYER_SHOOTING_RANGE = 5
    Settings.SHOOTING_SPEED = 10
    Settings.BALL_RADIUS = 1.0

    player_body = pymunk.Body(1, 1)
    player_body.position = (0, 0)
    player_body.angle = 0
    player_body.velocity = (3, 0)
    player_shape = pymunk.Circle(player_body, 5)
    player = (player_body, player_shape)

    ball_body = pymunk.Body(1, 1)
    ball_body.position = (7, 0)  # within shooting range
    ball_shape = pymunk.Circle(ball_body, Settings.BALL_RADIUS)
    ball = (ball_body, ball_shape)

    # Patch canShoot to force True
    monkeypatch.setattr("Engine.Actions.canShoot", lambda player_body, ball_body, max_distance=None: True)

    # Patch move to record arguments
    called_args = {}
    def fake_move(entity, speed=0, rotation_speed=0):
        called_args["entity"] = entity
        called_args["speed"] = speed
        called_args["rotation_speed"] = rotation_speed
    monkeypatch.setattr("Engine.Actions.move", fake_move)

    shoot(player, ball, power=2.0)

    # Check that move was called on the ball
    assert called_args["entity"] == ball
    expected_speed = np.linalg.norm(player_body.velocity) + Settings.SHOOTING_SPEED * 2.0
    assert math.isclose(called_args["speed"], expected_speed, rel_tol=1e-6)


def test_shoot_does_not_move_ball_if_cannot_shoot(monkeypatch):
    """
    Test shoot() does not move the ball if canShoot returns False.
    """
    Settings.PLAYER_LEN = 10
    Settings.PLAYER_SHOOTING_RANGE = 5
    Settings.SHOOTING_SPEED = 10
    Settings.BALL_RADIUS = 1.0

    player_body = pymunk.Body(1, 1)
    player_body.position = (0, 0)
    player_body.angle = 0
    player_shape = pymunk.Circle(player_body, 5)
    player = (player_body, player_shape)

    ball_body = pymunk.Body(1, 1)
    ball_body.position = (20, 0)
    ball_shape = pymunk.Circle(ball_body, Settings.BALL_RADIUS)
    ball = (ball_body, ball_shape)

    monkeypatch.setattr("Engine.Actions.canShoot", lambda player_body, ball_body, max_distance=None: False)
    called = {"moved": False}
    def fake_move(entity, speed=0, rotation_speed=0):
        called["moved"] = True
    monkeypatch.setattr("Engine.Actions.move", fake_move)

    shoot(player, ball)
    assert not called["moved"]
