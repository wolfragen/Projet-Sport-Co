# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 19:32:28 2025

@author: quent
"""

import math
import numpy as np

import Settings

def reset_movements(game):
    for player in game["players"]:
        move(player)
    return

def move(entity, speed=0, rotation_speed=0):
    body, shape = entity
    angle = body.angle
    
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    
    body.velocity = vx, vy
    
    body.angular_velocity = rotation_speed
    return

def shoot(player, ball):
    ball_body, ball_shape = ball
    player_body, player_shape = player
    
    if not canShoot(player_body, ball_body):
        return
    else:
        shooting_speed = np.linalg.norm(player_body.velocity) + Settings.SHOOTING_SPEED
        ball_body.angle = player_body.angle
        move(ball, speed=shooting_speed)
        
    return
        
def canShoot(player_body, ball_body, max_distance=Settings.PLAYER_SHOOTING_RANGE):
    """
    Vérifie si le joueur peut tirer sur la balle en fonction de la distance entre
    le centre de la face avant du joueur et le bord de la balle.
    
    Parameters
    ----------
    player_body : pymunk.Body
        Body du joueur
    ball_body : pymunk.Body
        Body de la balle
    player_length : float
        Longueur du joueur (taille du carré)
    ball_radius : float
        Rayon de la balle
    max_distance : float
        Distance maximale (à partir de la face avant jusqu'au bord de la balle) pour pouvoir tirer
    
    Returns
    -------
    bool
        True si le joueur peut tirer, False sinon
    """

    # Centre de la face avant du joueur
    front_offset = Settings.PLAYER_LEN / 2
    angle = player_body.angle
    front_pos = player_body.position + np.array([front_offset * np.cos(angle),
                                                 front_offset * np.sin(angle)])
    
    # Distance entre la face avant et le centre de la balle
    ball_pos = np.array(ball_body.position)
    distance = np.linalg.norm(front_pos - ball_pos)
    
    # Distance jusqu'au bord de la balle
    distance_to_edge = distance - Settings.BALL_RADIUS
    
    return distance_to_edge <= max_distance














































    