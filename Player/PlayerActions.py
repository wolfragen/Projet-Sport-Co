# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 18:44:53 2025

@author: quent
"""

import pygame
from typing import Callable

import Settings
from Engine import Actions


def process_events(game: dict, stopGame: Callable[[], None]) -> None:
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
            stopGame()
            return

    keys = pygame.key.get_pressed()
    selected_player = game["selected_player"]
    if(selected_player is None): return
    body, shape = selected_player
    speed = Settings.PLAYER_SPEED
    rotation_speed = Settings.PLAYER_ROT_SPEED

    # Forward movement
    if keys[pygame.K_w]:
        Actions.move(selected_player, speed=speed)

    # Rotate left
    if keys[pygame.K_a]:
        Actions.move(selected_player, rotation_speed=-rotation_speed)

    # Rotate right
    if keys[pygame.K_d]:
        Actions.move(selected_player, rotation_speed=rotation_speed)

    # Shoot the ball
    if keys[pygame.K_SPACE]:
        Actions.shoot(selected_player, game["ball"])
        
    return
        





















































