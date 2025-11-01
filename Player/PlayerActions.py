# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 18:44:53 2025

@author: quent
"""

import pygame


def process_events() -> None:
    """
    Processes user input events and applies actions to the selected player.
    
    Handles quitting the game, movement (forward, rotation), and shooting
    based on key presses.

    Parameters
    ----------
    game : dict
        Game state dictionary containing at least the 'selected_player' key.
    stopGame : Callable[[], None]
        Function to call when quitting the game (e.g., closes Pygame).

    Returns
    -------
    None
        The function updates the selected player state but does not return anything.
    """

    # Handle quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True, -1

    keys = pygame.key.get_pressed()

    # Shoot the ball
    if keys[pygame.K_SPACE]:
        return False,3
    
    # Rotate left
    if keys[pygame.K_a]:
        return False,1

    # Rotate right
    if keys[pygame.K_d]:
        return False,2
    
    # Forward movement
    if keys[pygame.K_w]:
        return False,0
        
    return False, -1
        





















































