# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:23:39 2025

@author: quent
"""

import numpy as np

import Settings


def checkIfGoal(game: dict, initGame: callable) -> np.uint8:
    """
    Checks if a goal has been scored and by which team.  
    If a goal is detected, resets the round by reinitializing the game state.

    Parameters
    ----------
    game : dict
        Current game state containing ball position, score, and other entities.
    initGame : callable
        Function to initialize a new game state, optionally accepting a score.

    Returns
    -------
    np.uint8
        0 : no goal
        1 : left team scored
        2 : right team scored
    """
    
    body, shape = game["ball"]
    ball_x = body.position[0]
    dim_x = Settings.DIM_X
    offset = Settings.SCREEN_OFFSET
    
    if ball_x < offset:
        # Goal scored by right team
        game["score"][1] += 1
        new_game = initGame(game["score"])
        game.update(new_game)
        print("Right team scored!")
        return np.uint8(2)
    
    elif ball_x > dim_x + offset:
        # Goal scored by left team
        game["score"][0] += 1
        new_game = initGame(game["score"])
        game.update(new_game)
        print("Left team scored!")
        return np.uint8(1)
    
    return np.uint8(0)


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
            x = offset
        elif x > dim_x + offset:
            x = dim_x + offset
            
        body.position = (x, y)
        
    return





































