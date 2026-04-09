# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:23:39 2025

@author: quent
"""

from Settings import Settings
from Engine.Actions import canShoot


def checkIfGoal(engine, score) -> bool:
    ball_x = engine.get_position(-1)[0]
    dim_x = Settings.DIM_X
    offset = Settings.SCREEN_OFFSET

    if ball_x < offset:
        score[1] += 1
        return True
    elif ball_x > dim_x + offset:
        score[0] += 1
        return True

    return False


def checkPlayersOut(engine) -> None:
    """
    Ensures that players stay within the horizontal bounds of the field.
    If a player moves outside the left or right boundary, their x-position is clamped.
    """
    dim_x = Settings.DIM_X
    offset = Settings.SCREEN_OFFSET

    for player_id in range(engine.get_n_players()):
        x, y = engine.get_position(player_id)

        if x < offset:
            x = offset + 10
            engine.set_position(player_id, x, y)
            engine.update_previous_position(player_id)
        elif x > dim_x + offset:
            x = dim_x + offset - 10
            engine.set_position(player_id, x, y)
            engine.update_previous_position(player_id)


def checkPlayersCanShoot(engine):
    for player_id in range(engine.get_n_players()):
        last_shoot = engine.get_last_player_shoot()
        engine.update_had_ball(player_id, engine.get_can_shoot(player_id))
        can = canShoot(engine, player_id)
        engine.update_can_shoot(player_id, can)
        if can and player_id != last_shoot:
            engine.set_last_player_shoot(None)
