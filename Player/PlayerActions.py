# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 18:44:53 2025

@author: quent
"""

import time
import pygame

import Settings


time_last_action = 0

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
    
    global time_last_action

    # Handle quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True, -1

    if(time.time() - time_last_action < Settings.DELTA_TIME/1000):
        return False, -1

    keys = pygame.key.get_pressed()

    # Shoot the ball
    if keys[pygame.K_SPACE]:
        time_last_action = time.time()
        return False,3
    
    # Rotate left
    if keys[pygame.K_a]:
        time_last_action = time.time()
        return False,1

    # Rotate right
    if keys[pygame.K_d]:
        time_last_action = time.time()
        return False,2
    
    # Backward movement
    if keys[pygame.K_s]:
        time_last_action = time.time()
        return False,4
    
    # Forward movement
    if keys[pygame.K_w]:
        time_last_action = time.time()
        return False,0
        
    return False, -1
        





















































