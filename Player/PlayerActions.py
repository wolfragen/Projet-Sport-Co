# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 18:44:53 2025

@author: quent
"""

import pygame

import Settings
from Engine import Actions


def process_events(game, stopGame):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stopGame()
            return

    keys = pygame.key.get_pressed()    
    selected_player = game["selected_player"]
    body, shape = selected_player
    speed = Settings.PLAYER_SPEED
    rotation_speed = Settings.PLAYER_ROT_SPEED
    
    if keys[pygame.K_w]:
        # On avance
        Actions.move(selected_player, speed=speed)
        
    if keys[pygame.K_a]:
        # Rotation à gauche
        Actions.move(selected_player, rotation_speed=-rotation_speed)
        
    if keys[pygame.K_d]:
        # Rotation à droite
        Actions.move(selected_player, rotation_speed=+rotation_speed)
        
    if keys[pygame.K_SPACE]:
        # Tir !
        Actions.shoot(selected_player, game["ball"])





















































