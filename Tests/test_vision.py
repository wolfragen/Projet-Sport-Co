# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:24:19 2025

@author: quent
"""
import numpy as np
import pymunk
import types

import Settings
from Engine.Vision import rayTracing, getVision


def test_rayTracing_and_getVision_basic():
    """
    Basic consistency test for rayTracing and getVision without real collisions.
    """

    # --- 1. Mock Settings values ---
    Settings.NUMBER_OF_RAYS = 8
    Settings.RAY_ANGLE = np.pi / 2
    Settings.VISION_RANGE = 100
    Settings.PLAYER_LEN = 10
    Settings.LEFT_GOAL_COLLISION_TYPE = 2
    Settings.RIGHT_GOAL_COLLISION_TYPE = 3
    Settings.DIM_X = 200
    Settings.DIM_Y = 100
    Settings.ENTRY_NEURONS = 7 + 8 * 8  # 7 + NUMBER_OF_RAYS*8

    # --- 2. Build minimal game environment ---
    space = pymunk.Space()
    player_body = pymunk.Body(1, 1)
    player_body.position = (50, 50)
    player_shape = pymunk.Circle(player_body, 5)
    space.add(player_body, player_shape)

    ball_body = pymunk.Body(1, 1)
    ball_body.position = (100, 50)
    ball_shape = pymunk.Circle(ball_body, 3)
    ball_shape.is_ball = True
    space.add(ball_body, ball_shape)

    game = {
        "space": space,
        "ball": (ball_body, ball_shape),
        "left_goal_position": (0, 50),
        "right_goal_position": (200, 50)
    }

    # --- 3. Test rayTracing ---
    ray_data = rayTracing(game, (player_body, player_shape))

    assert isinstance(ray_data, np.ndarray)
    assert ray_data.shape == (Settings.NUMBER_OF_RAYS * 8,)
    assert np.all(np.isfinite(ray_data) | np.isinf(ray_data))  # no NaN values
    assert ray_data.dtype == np.float32
    assert np.any(ray_data != 0)  # some distances should be filled

    # --- 4. Test getVision ---
    vision = getVision(game, (player_body, player_shape))

    assert isinstance(vision, np.ndarray)
    assert vision.dtype == np.float32
    assert vision.shape == (Settings.ENTRY_NEURONS,)

    # Orientation normalization between -1 and 1
    assert -1.0 <= vision[0] <= 1.0

    # Normalized coordinates should stay within [-1, 1]
    assert np.all(np.abs(vision[1:7]) <= 1.0)

    # Ray tracing data should exist
    ray_segment = vision[7:]
    assert len(ray_segment) == Settings.NUMBER_OF_RAYS * 8
    assert np.all(np.isfinite(ray_segment))

    # --- 5. Consistency check ---
    vision2 = getVision(game, (player_body, player_shape))
    assert np.allclose(vision, vision2, atol=1e-6)

    # --- 6. Robustness check: move player ---
    player_body.position = (150, 80)
    new_vision = getVision(game, (player_body, player_shape))
    assert not np.allclose(vision, new_vision, atol=1e-3)


def test_rayTracing_detects_ball_type(monkeypatch):
    """
    Targeted test: ensures that rayTracing correctly encodes 'ball' object type.
    """

    # --- 1. Mock Settings values ---
    Settings.NUMBER_OF_RAYS = 3
    Settings.RAY_ANGLE = np.pi / 3
    Settings.VISION_RANGE = 50
    Settings.PLAYER_LEN = 10
    Settings.LEFT_GOAL_COLLISION_TYPE = 2
    Settings.RIGHT_GOAL_COLLISION_TYPE = 3

    # --- 2. Create a minimal space with player and ball ---
    space = pymunk.Space()

    player_body = pymunk.Body(1, 1)
    player_body.position = (0, 0)
    player_shape = pymunk.Circle(player_body, 5)

    ball_body = pymunk.Body(1, 1)
    ball_body.position = (10, 0)
    ball_shape = pymunk.Circle(ball_body, 3)
    ball_shape.is_ball = True

    space.add(player_body, player_shape, ball_body, ball_shape)

    game = {"space": space, "ball": (ball_body, ball_shape)}

    # --- 3. Monkeypatch raycast to simulate a hit ---
    fake_hit = types.SimpleNamespace(
        shape=ball_shape,
        point=pymunk.Vec2d(5, 0)
    )
    monkeypatch.setattr(space, "segment_query_first", lambda *a, **kw: fake_hit)

    # --- 4. Run rayTracing and verify one-hot encoding ---
    data = rayTracing(game, (player_body, player_shape))
    one_hot = data.reshape(-1, 8)[:, 1:]
    assert np.any(one_hot[:, 4] == 1.0), "Type 'ball' not detected in one-hot encoding"
