# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:47:35 2025

@author: quent
"""

import Engine.Actions as Actions
from Settings import Settings


def play(engine, player_id, action):
    """
    Execute one AI action for a given player.

    Parameters
    ----------
    engine : PhysicsEngine
    player_id : int
    action : int or np.array
    """
    speed = Settings.PLAYER_SPEED
    rotation_speed = Settings.PLAYER_ROT_SPEED

    if type(action) != int:
        if action[2] > 0.1:
            Actions.shoot(engine, player_id, power=action[2])
        Actions.move(engine, player_id, speed=action[0] * speed, rotation_speed=action[1] * rotation_speed)
        return

    if action == 0:
        Actions.move(engine, player_id, speed=speed)
    elif action == 1:
        Actions.move(engine, player_id, rotation_speed=-rotation_speed)
    elif action == 2:
        Actions.move(engine, player_id, rotation_speed=rotation_speed)
    elif action == 3:
        Actions.shoot(engine, player_id)

    return
