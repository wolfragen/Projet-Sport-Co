# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 19:32:28 2025

@author: quent
"""

import math

from Settings import Settings


def reset_movements(engine) -> None:
    """Reset all player movements by stopping their current velocity."""
    for player_id in range(engine.get_n_players()):
        move(engine, player_id, speed=0, rotation_speed=0)


def move(engine, entity_id: int, speed: float = 0, rotation_speed: float = 0) -> None:
    """Apply linear and angular velocity to an entity (player or ball)."""
    angle = engine.get_angle(entity_id)

    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)

    engine.set_velocity(entity_id, vx, vy)
    engine.set_angular_velocity(entity_id, rotation_speed)


def shoot(engine, player_id: int, power: float = 1.0) -> None:
    """Make a player shoot the ball in the direction the player is facing."""
    if not engine.get_can_shoot(player_id):
        return

    vx, vy = engine.get_velocity(player_id)
    player_speed = math.sqrt(vx * vx + vy * vy)
    shooting_speed = player_speed + Settings.SHOOTING_SPEED * power

    player_angle = engine.get_angle(player_id)
    engine.set_angle(-1, player_angle)

    move(engine, -1, speed=shooting_speed)

    engine.set_last_player_shoot(player_id)


def canShoot(engine, player_id: int, max_distance: float = Settings.PLAYER_SHOOTING_RANGE) -> bool:
    """Check if a player can shoot the ball based on distance from front face to ball edge."""
    front_offset = Settings.PLAYER_LEN / 2
    px, py = engine.get_position(player_id)
    angle = engine.get_angle(player_id)

    front_x = px + front_offset * math.cos(angle)
    front_y = py + front_offset * math.sin(angle)

    bx, by = engine.get_position(-1)

    dx = front_x - bx
    dy = front_y - by
    distance = math.sqrt(dx * dx + dy * dy)

    distance_to_edge = distance - engine.get_ball_radius()

    return distance_to_edge <= max_distance
