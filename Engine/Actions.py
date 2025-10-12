# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 19:32:28 2025

@author: quent
"""

import math
import numpy as np
import pymunk

import Settings

def reset_movements(game: dict) -> None:
    """
    Reset all player movements by stopping their current velocity.

    Parameters
    ----------
    game : dict
        The game dictionary containing the list of players.

    Returns
    -------
    None
        This function does not return any value.
    """
    
    # Iterate over all players and reset their movement
    for player in game.get("players", []):
        move(player, speed=0, rotation_speed=0)
    
    return

def move(entity: tuple[pymunk.Body, pymunk.Shape], speed: float = 0, rotation_speed: float = 0) -> None:
    """
    Apply linear and angular velocity to an entity (player or object).

    Parameters
    ----------
    entity : tuple[pymunk.Body, pymunk.Shape]
        Tuple containing the Pymunk body and shape of the entity.
    speed : float, optional
        Forward speed to apply along the entity's current angle (default is 0).
    rotation_speed : float, optional
        Angular velocity to apply (default is 0).

    Returns
    -------
    None
        The entity's velocity and angular_velocity are updated in-place.
    """
    
    body, shape = entity
    angle = body.angle

    # Compute velocity components along the entity's facing direction
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)

    # Apply linear velocity
    body.velocity = vx, vy

    # Apply angular velocity
    body.angular_velocity = rotation_speed
    
    return

def shoot(player: tuple[pymunk.Body, pymunk.Shape],
          ball: tuple[pymunk.Body, pymunk.Shape],
          power: float = 1.0) -> None:
    """
    Make a player shoot the ball in the direction the player is facing.

    Parameters
    ----------
    player : tuple[pymunk.Body, pymunk.Shape]
        Tuple containing the player's Pymunk body and shape.
    ball : tuple[pymunk.Body, pymunk.Shape]
        Tuple containing the ball's Pymunk body and shape.
    power : float, optional
        Multiplier for the shot's speed (default is 1.0).

    Returns
    -------
    None
        Updates the ball's velocity and angle in-place.
    """
    
    ball_body, ball_shape = ball
    player_body, player_shape = player
    
    # Only allow shooting if ball is reachable
    if not canShoot(player_body, ball_body):
        return
    
    # Compute the shooting speed: current player speed + base shooting speed scaled by power
    shooting_speed = np.linalg.norm(player_body.velocity) + Settings.SHOOTING_SPEED * power

    # Align the ball angle with the player
    ball_body.angle = player_body.angle

    # Apply velocity to the ball in the player's facing direction
    move(ball, speed=shooting_speed)
    
    return
        
def canShoot(player_body: pymunk.Body,
             ball_body: pymunk.Body,
             max_distance: float = Settings.PLAYER_SHOOTING_RANGE) -> bool:
    """
    Check if a player can shoot the ball based on the distance from the front face of the player 
    to the edge of the ball.

    Parameters
    ----------
    player_body : pymunk.Body
        The Pymunk body of the player.
    ball_body : pymunk.Body
        The Pymunk body of the ball.
    max_distance : float, optional
        Maximum distance allowed from the player's front face to the ball's edge to be able to shoot 
        (default is Settings.PLAYER_SHOOTING_RANGE).

    Returns
    -------
    bool
        True if the player can shoot, False otherwise.
    """
    
    # Calculate the center position of the player's front face
    front_offset = Settings.PLAYER_LEN / 2
    angle = player_body.angle
    front_pos = player_body.position + np.array([
        front_offset * np.cos(angle),
        front_offset * np.sin(angle)
    ])

    # Vector to the ball and distance
    ball_pos = np.array(ball_body.position)
    distance = np.linalg.norm(front_pos - ball_pos)

    # Distance from front face to the edge of the ball
    distance_to_edge = distance - Settings.BALL_RADIUS

    return distance_to_edge <= max_distance














































    