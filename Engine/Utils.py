# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:23:39 2025

@author: quent
"""

import Settings


def checkIfGoal(game: dict) -> tuple[bool,bool]:
    """
    Checks if a goal has been scored and by which team.  

    Parameters
    ----------
    game : dict
        Current game state containing ball position, score, and other entities.

    Returns
    -------
    bool
        True if a goal was scored, false otherwise.
    bool
        True if left team scored, False if right team scored, None otherwise.
    """
    
    body, shape = game["ball"]
    ball_x = body.position[0]
    dim_x = Settings.DIM_X
    offset = Settings.SCREEN_OFFSET
    
    if ball_x < offset:
        # Goal scored by right team
        game["score"][1] += 1
        return (True, False)
    
    elif ball_x > dim_x + offset:
        # Goal scored by left team
        game["score"][0] += 1
        return (True, True)
    
    return (False, None)


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





































