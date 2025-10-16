# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:47:35 2025

@author: quent
"""

import numpy as np
import torch

import Engine.Vision as Vision
import Engine.Actions as Actions
import AI.Network as nn
import Settings


def play(game: dict, player: tuple, model: nn.DeepRLNetwork = None, vision: np.array = None) -> np.array:
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
    
    # Get the AI vision of the environment if not passed (graphic simulation)
    if(vision is None):
        vision = Vision.getVision(game, player)
    
    if(model is None):
        decision_array = random_play(game, player)
    else:
        decision_array = model(torch.from_numpy(vision).float()).detach().numpy()
    
    # Map decision outputs to main actions: move forward, rotate left, rotate right, shoot
    actions_array = [decision_array[0], decision_array[2], decision_array[4], decision_array[6]]
    
    body, shape = player
    speed = Settings.PLAYER_SPEED
    rotation_speed = Settings.PLAYER_ROT_SPEED
    
    # Determine the action with the highest value
    i_max = actions_array.index(max(actions_array))  # Final decision
    
    if(max(actions_array)) == 0:
        return decision_array
    
    if i_max == 0:
        # Move forward
        Actions.move(player, speed=speed)  # TODO variable speed? decision_array[1]
        
    if i_max == 1:
        # Rotate left
        Actions.move(player, rotation_speed=-rotation_speed)  # TODO variable speed? decision_array[3]
        
    if i_max == 2:
        # Rotate right
        Actions.move(player, rotation_speed=rotation_speed)  # TODO variable speed? decision_array[5]
        
    if i_max == 3:
        # Shoot
        Actions.shoot(player, game["ball"], power=decision_array[7])
    
    return decision_array

def random_play(game: dict, player: tuple) -> np.ndarray:
    """
    Generate a random action array for a player.
    This is a placeholder for AI decision-making.

    Parameters
    ----------
    game : dict
        Dictionary containing the current game state.
    player : tuple
        Tuple containing the player's body and shape (pymunk.Body, pymunk.Shape).

    Returns
    -------
    np.ndarray
        Array of 8 random floats between 0 and 1 representing action preferences.
    """
    
    # Generate random float values in [0, 1] for each possible action
    return np.random.rand(8)

































