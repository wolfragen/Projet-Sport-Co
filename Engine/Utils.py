# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:23:39 2025

@author: quent
"""

import pymunk

import Settings
from Engine.Actions import canShoot


def createSpace():
    space = pymunk.Space() # Physics simulation space
    space.damping = Settings.GROUND_FRICTION # Simulate the friction with the ground
    return space


def checkIfGoal(ball, score) -> bool:
    
    body, shape = ball
    ball_x = body.position[0]
    dim_x = Settings.DIM_X
    offset = Settings.SCREEN_OFFSET
    
    if ball_x < offset:
        # Goal scored by right team
        score[1] += 1
        return True
    
    elif ball_x > dim_x + offset:
        # Goal scored by left team
        score[0] += 1
        return True
    
    return False


def checkPlayersOut(players: list[tuple]) -> None:
    """
    Ensures that players stay within the horizontal bounds of the field.  
    If a player moves outside the left or right boundary, their x-position is clamped.

    Parameters
    ----------
    players : list[tuple]
        List of player entities, each a tuple (body, shape) where `body` is a pymunk.Body.

    Returns
    -------
    None
        This function modifies the player positions in place.
    """
    
    dim_x = Settings.DIM_X
    offset = Settings.SCREEN_OFFSET
    
    for player in players:
        body, shape = player
        x, y = body.position
        
        # Clamp horizontal position
        if x < offset:
            x = offset + 10
        elif x > dim_x + offset:
            x = dim_x + offset - 10
            
        body.position = (x, y)
        body.previous_position = body.position
        
    return 

def checkPlayersCanShoot(players, ball):
    ball_body = ball[0]
    for player in players:
        player_body = player[0]
        player_body.hadBall = player_body.canShoot
        player_body.canShoot = canShoot(player_body, ball_body)
    return





































