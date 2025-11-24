# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:47:35 2025

@author: quent
"""

import numpy as np

import Engine.Actions as Actions
import Settings


def play(player, ball, action) -> np.array:
    """
    Execute one AI action for a given player based on its vision and a decision array.

    Parameters
    ----------
    game : dict
        Dictionary containing the current game state.
    player : tuple
        Tuple containing the player's body and shape (pymunk.Body, pymunk.Shape).

    Returns
    -------
    decision_array : np.array
        Returns the output vector of the AI.
    """
    
    body, shape = player
    ball_body, ball_shape = ball
    speed = Settings.PLAYER_SPEED
    rotation_speed = Settings.PLAYER_ROT_SPEED
    
    if action == 0:
        # Move forward
        Actions.move(player, speed=speed)
    elif action == 1:
        # Rotate left
        Actions.move(player, rotation_speed=-rotation_speed)
    elif action == 2:
        # Rotate right
        Actions.move(player, rotation_speed=rotation_speed)
        
    elif action == 3:
        # Move forward
        Actions.move(player, speed=speed*0.6)
    elif action == 4:
        # Rotate left
        Actions.move(player, rotation_speed=-rotation_speed*0.6)
    elif action == 5:
        # Rotate right
        Actions.move(player, rotation_speed=rotation_speed*0.6)
        
    elif action == 6:
        # Move forward
        Actions.move(player, speed=speed*0.3)
    elif action == 7:
        # Rotate left
        Actions.move(player, rotation_speed=-rotation_speed*0.3)
    elif action == 8:
        # Rotate right
        Actions.move(player, rotation_speed=rotation_speed*0.3)
        
    elif action == 9:
        # Shoot
        Actions.shoot(player, ball, power=1.0)
    elif action == 10:
        # Shoot
        Actions.shoot(player, ball, power=0.6)
    elif action == 11:
        # Shoot
        Actions.shoot(player, ball, power=0.3)
        
        
        """
    if action == 4:
        # Move backward
        Actions.move(player, speed=-speed)"""
    
    return

































