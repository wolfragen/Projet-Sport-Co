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

import Settings


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


def getVision(space, players: list[tuple[pymunk.Body, pymunk.Shape]], player_id, ball, left_goal_position, right_goal_position) -> np.ndarray:

    vision_array = np.zeros(Settings.ENTRY_NEURONS, dtype=np.float32)

    # Player, ball, and goals positions
    player = players[player_id]
    body, shape = player
    ball_body, _ = ball
    dim_x = Settings.DIM_X
    dim_y = Settings.DIM_Y
    
    # Normalize positions by the field dimensions
    dx_ball = (ball_body.position[0] - body.position[0]) / dim_x
    dy_ball = (ball_body.position[1] - body.position[1]) / dim_y
    
    dx_left_goal  = (left_goal_position[0]  - body.position[0]) / dim_x
    dy_left_goal  = (left_goal_position[1]  - body.position[1]) / dim_y
    dx_right_goal = (right_goal_position[0] - body.position[0]) / dim_x
    dy_right_goal = (right_goal_position[1] - body.position[1]) / dim_y
    
    if shape.left_team:
        sin_a = math.sin(body.angle)
        cos_a = math.cos(body.angle)
    
        own_goal_dx, own_goal_dy = dx_left_goal, dy_left_goal
        opp_goal_dx, opp_goal_dy = dx_right_goal, dy_right_goal
        ball_dx, ball_dy = dx_ball, dy_ball
        
    else:
        sin_a = math.sin(body.angle + math.pi)
        cos_a = math.cos(body.angle + math.pi)
    
        own_goal_dx, own_goal_dy = -dx_right_goal, -dy_right_goal
        opp_goal_dx, opp_goal_dy = -dx_left_goal,  -dy_left_goal
        ball_dx, ball_dy = -dx_ball, -dy_ball
    
    if not Settings.COMPETITIVE_VISION :
        vision_array[0] = sin_a
        vision_array[1] = cos_a
        vision_array[2:4] = (ball_dx, ball_dy)
        vision_array[4:6] = (own_goal_dx, own_goal_dy)
        vision_array[6:8] = (opp_goal_dx, opp_goal_dy)
        
        if Settings.ENTRY_NEURONS == 9: # TODO à changer
            vision_array[8] = int(body.canShoot)
            
    else:
        # Compétitif, vision joueur adverse également !
        assert len(players) == 2 # TODO ajuster pour +
        opponent = players[0]
        if(player_id == 0):
            opponent = players[1]
        opp_body, opp_shape = opponent
        
        ball_vx_rel = (ball_body.velocity[0] - body.velocity[0]) / (Settings.SHOOTING_SPEED + Settings.PLAYER_SPEED)
        ball_vy_rel = (ball_body.velocity[1] - body.velocity[1]) / (Settings.SHOOTING_SPEED + Settings.PLAYER_SPEED)
        
        dx_opp_ball = (ball_body.position[0] - opp_body.position[0]) / dim_x
        dy_opp_ball = (ball_body.position[1] - opp_body.position[1]) / dim_y
        dist_ball_opponent = math.sqrt(dx_opp_ball**2 + dy_opp_ball**2)
        
        vision_array[0] = sin_a
        vision_array[1] = cos_a
        vision_array[2:4] = (ball_dx, ball_dy)
        vision_array[4:6] = (own_goal_dx, own_goal_dy)
        vision_array[6:8] = (opp_goal_dx, opp_goal_dy)
        vision_array[8:10] = (ball_vx_rel, ball_vy_rel)
        vision_array[10] = dist_ball_opponent
        

    """ # TODO: remettre si on remet le ray Tracing
    # Normalize ray distances and copy one-hot info
    ray_data = rayTracing(space, player)
    ray_data[::8] = ray_data[::8] / dim_x
    ray_data[1::8] = ray_data[1::8] / dim_y
    vision_array[8:] = ray_data.flatten()
    """
    
    return vision_array










































