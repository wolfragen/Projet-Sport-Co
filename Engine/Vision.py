# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:49:47 2025

@author: quent


Pour les types en rayTracing :
    - 0: rien trouvé
    - 1: mur
    - 2: goal gauche
    - 3: goal droit
    - 4: balle
    - 5: joueur gauche
    - 6: joueur droit

autres paramètres de vision : 
    - orientation cos, sin
    - position de la balle % joueur
    - position du goal gauche % joueur
    - position du goal droit % joueur
    
Total de 520 entrées.

"""

import math
import numpy as np
import pymunk

from Settings import Settings


def rayTracing(space, player: tuple[pymunk.Body, pymunk.Shape]) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform raycasting from a player's position and convert it into a flattened vision array.

    Each ray detects the distance to the first object hit and encodes the object type
    as a one-hot vector. The final output is a flattened array containing distances
    and one-hot encodings for all rays.

    Parameters
    ----------
    game : dict
        The current game state containing 'space' and all entities.
    player : tuple (pymunk.Body, pymunk.Shape)
        The player body and shape to cast rays from.

    Returns
    -------
    vision_array : np.ndarray, shape (NUMBER_OF_RAYS*8,)
        Flattened observation vector where each ray contributes 2 position value (x,y) 
        followed by a 6-element one-hot vector representing the entity type:
            distance : float
                Distance from the player to the first hit along the ray.
                np.inf if no collision detected.
            one-hot entity type : array of length 6
                0: nothing
                1: wall
                2: left goal
                3: right goal
                4: ball
                5: left team player
                6: right team player
    """
    
    body, shape = player

    number_of_rays = Settings.NUMBER_OF_RAYS
    fov = Settings.RAY_ANGLE
    max_dist = Settings.VISION_RANGE

    # Starting angle for the first ray
    start_angle = body.angle - fov / 2
    step = fov / (number_of_rays - 1)
    origin = body.position

    positions = np.full(number_of_rays*2, max_dist, dtype=np.float32)
    types = np.zeros(number_of_rays, dtype=np.int8)

    # Save original filter and ignore the player itself
    original_filter = shape.filter
    shape.filter = pymunk.ShapeFilter(mask=0)

    for i in range(number_of_rays):
        angle = start_angle + i * step
        direction = pymunk.Vec2d(math.cos(angle), math.sin(angle))
        end_point = origin + direction * max_dist

        # Raycasting using pymunk
        ray_radius = max(1, round(Settings.PLAYER_LEN * 0.05))
        hit = space.segment_query_first(origin, end_point, ray_radius, pymunk.ShapeFilter())

        if hit is not None:
            hit_shape = hit.shape
            positions[2*i:2*i+2] = hit.point - origin  # fraction of max distance

            # Identify object type
            if hasattr(hit_shape, "is_ball") and hit_shape.is_ball:
                types[i] = np.uint8(4)
            elif hasattr(hit_shape, "is_player") and hit_shape.is_player:
                types[i] = np.uint8(5 if hit_shape.left_team else 6)
            elif hit_shape.collision_type == Settings.LEFT_GOAL_COLLISION_TYPE:
                types[i] = np.uint8(2)
            elif hit_shape.collision_type == Settings.RIGHT_GOAL_COLLISION_TYPE:
                types[i] = np.uint8(3)
            else:
                types[i] = np.uint8(1)  # default: wall

    # Restore original filter
    shape.filter = original_filter

    # Preallocate array
    vision_array = np.zeros(number_of_rays * 8, dtype=np.float32)

    # Vectorized assignment for one-hot encoding
    indices = np.nonzero(types > 2) # TODO: à vérifier
    for i in indices:
        vision_array[8*i+1 + types[i]] = 1.0

    # Fill distances for all rays
    for i in range(number_of_rays):
        vision_array[8*i:8*i+2] = positions[i]
        
    return vision_array


def getVision(space, players: list[tuple[pymunk.Body, pymunk.Shape]], player_id, ball, left_goal_position, right_goal_position, phantom_player=None) -> np.ndarray:

    vision_array = np.zeros(Settings.ENTRY_NEURONS, dtype=np.float32)

    player = players[player_id]
    body, shape = player
    ball_body, _ = ball

    dim_x = Settings.DIM_X
    dim_y = Settings.DIM_Y
    shooting_speed = Settings.SHOOTING_SPEED
    player_speed = Settings.PLAYER_SPEED

    body_pos_x, body_pos_y = body.position
    ball_pos_x, ball_pos_y = ball_body.position

    dx_ball = (ball_pos_x - body_pos_x) / dim_x
    dy_ball = (ball_pos_y - body_pos_y) / dim_y

    dx_left_goal  = (left_goal_position[0]  - body_pos_x) / dim_x
    dy_left_goal  = (left_goal_position[1]  - body_pos_y) / dim_y
    dx_right_goal = (right_goal_position[0] - body_pos_x) / dim_x
    dy_right_goal = (right_goal_position[1] - body_pos_y) / dim_y

    if shape.left_team:
        sin_a = math.sin(body.angle)
        cos_a = math.cos(body.angle)

        own_goal_dx, own_goal_dy = dx_left_goal, dy_left_goal
        opp_goal_dx, opp_goal_dy = dx_right_goal, dy_right_goal
        ball_dx, ball_dy = dx_ball, dy_ball
    else:
        angle = body.angle + math.pi
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        own_goal_dx, own_goal_dy = -dx_right_goal, -dy_right_goal
        opp_goal_dx, opp_goal_dy = -dx_left_goal,  -dy_left_goal
        ball_dx, ball_dy = -dx_ball, -dy_ball

    vision_array[0] = sin_a
    vision_array[1] = cos_a
    vision_array[2:4] = (ball_dx, ball_dy)
    vision_array[4:6] = (own_goal_dx, own_goal_dy)
    vision_array[6:8] = (opp_goal_dx, opp_goal_dy)

    if not Settings.COMPETITIVE_VISION:

        if Settings.ENTRY_NEURONS == 9:
            vision_array[8] = int(body.canShoot)

    else:

        body_vel_x, body_vel_y = body.velocity
        ball_vel_x, ball_vel_y = ball_body.velocity

        denom = shooting_speed + player_speed
        ball_vx_rel = (ball_vel_x - body_vel_x) / denom
        ball_vy_rel = (ball_vel_y - body_vel_y) / denom

        if not shape.left_team:
            ball_vx_rel = -ball_vx_rel
            ball_vy_rel = -ball_vy_rel

        vision_array[8:10] = (ball_vx_rel, ball_vy_rel)

        insert_index = 10

        for other_id, (other_body, other_shape) in enumerate(players):

            if other_id == player_id:
                continue

            other_pos_x, other_pos_y = other_body.position

            dx_other_player_ball = (ball_pos_x - other_pos_x) / dim_x
            dy_other_player_ball = (ball_pos_y - other_pos_y) / dim_y

            if not shape.left_team:
                dx_other_player_ball = -dx_other_player_ball
                dy_other_player_ball = -dy_other_player_ball

            vision_array[insert_index:insert_index+3] = (
                dx_other_player_ball,
                dy_other_player_ball,
                other_shape.left_team,
            )

            insert_index += 3
            
        """
        # Old version : 
        for other_id, (other_body, other_shape) in enumerate(players):

            if other_id == player_id:
                continue

            other_pos_x, other_pos_y = other_body.position

            dx_other_player_ball = (ball_pos_x - other_pos_x) / dim_x
            dy_other_player_ball = (ball_pos_y - other_pos_y) / dim_y

            if not shape.left_team:
                dx_other_player_ball = -dx_other_player_ball
                dy_other_player_ball = -dy_other_player_ball

            vision_array[insert_index:insert_index+2] = (
                dx_other_player_ball,
                dy_other_player_ball,
            )

            insert_index += 2
        """

    return vision_array


def getGlobalVision(space, players: list[tuple[pymunk.Body, pymunk.Shape]], ball, left_goal_position, right_goal_position, phantom_player=None) -> np.ndarray:

    ball_body, _ = ball

    dim_x = Settings.DIM_X
    dim_y = Settings.DIM_Y

    ball_pos_x, ball_pos_y = ball_body.position
    ball_vel_x, ball_vel_y = ball_body.velocity

    vision = [
        ball_pos_x / dim_x,
        ball_pos_y / dim_y,
        ball_vel_x / (Settings.SHOOTING_SPEED + Settings.PLAYER_SPEED),
        ball_vel_y / (Settings.SHOOTING_SPEED + Settings.PLAYER_SPEED),
    ]

    for body, shape in players:
        pos_x, pos_y = body.position

        vision.extend([
            pos_x / dim_x,
            pos_y / dim_y,
            math.sin(body.angle),
            math.cos(body.angle),
            1.0 if shape.left_team else -1.0
        ])

    return np.asarray(vision, dtype=np.float32)










































