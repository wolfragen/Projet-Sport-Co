# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:50:19 2025

@author: quent
"""

import pytest
import math
import numpy as np

from Engine import Actions


# -------------------------------------------------------------------
# Mock des objets pymunk
# -------------------------------------------------------------------
class MockBody:
    def __init__(self, position=(0, 0), angle=0):
        self.position = np.array(position, dtype=float)
        self.angle = angle
        self.velocity = np.array([0.0, 0.0])
        self.angular_velocity = 0.0
        self.canShoot = True  # utilisé dans shoot()

class MockShape:
    pass


# -------------------------------------------------------------------
# Mock Settings
# -------------------------------------------------------------------
class MockSettings:
    SHOOTING_SPEED = 10.0
    PLAYER_LEN = 2.0
    PLAYER_SHOOTING_RANGE = 3.0
    BALL_RADIUS = 0.5


# Remplace Settings
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setattr(Actions, "Settings", MockSettings)
    

# -------------------------------------------------------------------
# TEST move()
# -------------------------------------------------------------------
def test_move_basic():
    body = MockBody(position=(0, 0), angle=0)
    shape = MockShape()

    Actions.move((body, shape), speed=5, rotation_speed=2)

    assert np.allclose(body.velocity, [5, 0])
    assert body.angular_velocity == 2


def test_move_with_angle():
    body = MockBody(position=(0, 0), angle=math.pi / 2)
    shape = MockShape()

    Actions.move((body, shape), speed=4)

    assert np.allclose(body.velocity, [0, 4], atol=1e-6)


# -------------------------------------------------------------------
# TEST reset_movements()
# -------------------------------------------------------------------
def test_reset_movements():
    players = [(MockBody(angle=1.2), MockShape()), (MockBody(angle=0.5), MockShape())]

    # Vitesses initiales
    for p, _ in players:
        p.velocity = np.array([3, 4])
        p.angular_velocity = 5

    Actions.reset_movements(players)

    for p, _ in players:
        assert np.allclose(p.velocity, [0, 0])
        assert p.angular_velocity == 0


# -------------------------------------------------------------------
# TEST define_previous_pos()
# -------------------------------------------------------------------
def test_define_previous_pos():
    players = [(MockBody(position=(1, 2), angle=0.5), MockShape())]
    ball = (MockBody(position=(5, 5)), MockShape())

    Actions.define_previous_pos(players, ball)

    assert np.allclose(players[0][0].previous_position, [1, 2])
    assert players[0][0].previous_angle == 0.5
    assert np.allclose(ball[0].previous_position, [5, 5])


# -------------------------------------------------------------------
# TEST shoot()
# -------------------------------------------------------------------
def test_shoot_when_canShoot_true(monkeypatch):
    player = (MockBody(position=(0, 0), angle=0), MockShape())
    ball = (MockBody(position=(0, 0), angle=0), MockShape())

    player[0].velocity = np.array([3, 4])  # vitesse initiale → norme = 5

    Actions.shoot(player, ball, power=1.0)

    expected_speed = 5 + MockSettings.SHOOTING_SPEED
    assert np.allclose(ball[0].velocity, [expected_speed, 0])


def test_shoot_when_canShoot_false():
    player = (MockBody(position=(0, 0)), MockShape())
    ball = (MockBody(position=(0, 0)), MockShape())

    player[0].canShoot = False

    Actions.shoot(player, ball)

    # Aucune modification
    assert np.allclose(ball[0].velocity, [0, 0])


# -------------------------------------------------------------------
# TEST canShoot()
# -------------------------------------------------------------------
def test_canShoot_true():
    player = MockBody(position=(0, 0), angle=0)
    ball = MockBody(position=(2.2, 0))

    # FRONT du joueur = 1.0 unité devant lui (PLAYER_LEN/2)
    # Distance front → balle bord = (distance 2.2 - 0.5 BALL_RADIUS) = 1.7 <= 3.0 ⇒ TRUE
    assert Actions.canShoot(player, ball, Actions.Settings.PLAYER_SHOOTING_RANGE)


def test_canShoot_false():
    player = MockBody(position=(0, 0), angle=0)
    ball = MockBody(position=(10, 0))
    assert not Actions.canShoot(player, ball, Actions.Settings.PLAYER_SHOOTING_RANGE)


















