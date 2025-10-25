# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:00:06 2025

@author: quent
"""

import pymunk
import Settings
from Engine.Utils import checkIfGoal, checkPlayersOut

def test_checkIfGoal_scoring_behavior():
    """
    Test checkIfGoal for different ball positions and scoring outcomes.
    """

    # --- 1. Mock settings ---
    Settings.DIM_X = 100
    Settings.SCREEN_OFFSET = 10

    # --- 2. Ball in play, no goal ---
    ball_body = pymunk.Body(1,1)
    ball_body.position = (50, 0)
    ball_shape = pymunk.Circle(ball_body, radius=5)
    game = {"ball": (ball_body, ball_shape), "score": [0,0]}

    scored, left_team = checkIfGoal(game)
    assert not scored
    assert left_team is None
    assert game["score"] == [0,0]

    # --- 3. Ball left goal ---
    ball_body.position = (5, 0)
    scored, left_team = checkIfGoal(game)
    assert scored
    assert left_team is False
    assert game["score"] == [0,1]

    # --- 4. Ball right goal ---
    ball_body.position = (Settings.DIM_X + Settings.SCREEN_OFFSET + 1, 0)
    scored, left_team = checkIfGoal(game)
    assert scored
    assert left_team is True
    assert game["score"] == [1,1]


def test_checkPlayersOut_clamping_behavior():
    """
    Test that checkPlayersOut correctly clamps players within horizontal bounds.
    """

    Settings.DIM_X = 100
    Settings.SCREEN_OFFSET = 10

    # Create 3 players: one inside bounds, one left out, one right out
    players = []
    p1 = pymunk.Body(1,1)
    p1.position = (50, 0)
    p1_shape = pymunk.Circle(p1, 5)
    players.append((p1, p1_shape))

    p2 = pymunk.Body(1,1)
    p2.position = (5, 0)  # too far left
    p2_shape = pymunk.Circle(p2, 5)
    players.append((p2, p2_shape))

    p3 = pymunk.Body(1,1)
    p3.position = (Settings.DIM_X + Settings.SCREEN_OFFSET + 50, 0)  # too far right
    p3_shape = pymunk.Circle(p3, 5)
    players.append((p3, p3_shape))

    checkPlayersOut(players)

    # Player inside bounds stays the same
    assert p1.position[0] == 50
    # Player clamped left
    assert p2.position[0] == Settings.SCREEN_OFFSET + 10
    # Player clamped right
    assert p3.position[0] == Settings.DIM_X + Settings.SCREEN_OFFSET - 10

    # Ensure previous_position updated
    for p, _ in players:
        assert p.previous_position == p.position
