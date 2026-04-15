# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:49:47 2025

@author: quent
"""

import math
import numpy as np

from Settings import Settings


def rayTracing(engine, player_id: int) -> np.ndarray:
    """
    Perform raycasting from a player's position and convert it into a flattened vision array.
    """
    number_of_rays = Settings.NUMBER_OF_RAYS
    fov = Settings.RAY_ANGLE
    max_dist = Settings.VISION_RANGE

    origin = engine.get_position(player_id)
    player_angle = engine.get_angle(player_id)

    start_angle = player_angle - fov / 2
    step = fov / (number_of_rays - 1)

    positions = np.full(number_of_rays * 2, max_dist, dtype=np.float32)
    types = np.zeros(number_of_rays, dtype=np.int8)

    for i in range(number_of_rays):
        angle = start_angle + i * step
        direction = (math.cos(angle), math.sin(angle))
        end_point = (origin[0] + direction[0] * max_dist, origin[1] + direction[1] * max_dist)

        ray_radius = max(1, round(Settings.PLAYER_LEN * 0.05))
        hit = engine.raycast(origin, end_point, ray_radius, exclude_entity_id=player_id)

        if hit is not None:
            hit_point = hit["point"]
            positions[2 * i] = hit_point[0] - origin[0]
            positions[2 * i + 1] = hit_point[1] - origin[1]
            types[i] = np.uint8(hit["entity_type"])

    vision_array = np.zeros(number_of_rays * 8, dtype=np.float32)

    indices = np.nonzero(types > 2)
    for i in indices:
        vision_array[8 * i + 1 + types[i]] = 1.0

    for i in range(number_of_rays):
        vision_array[8 * i:8 * i + 2] = positions[i]

    return vision_array


def getVision(engine, player_id, left_goal_position, right_goal_position, phantom_player=None) -> np.ndarray:

    vision_array = np.zeros(Settings.ENTRY_NEURONS, dtype=np.float32)

    dim_x = Settings.DIM_X
    dim_y = Settings.DIM_Y
    shooting_speed = Settings.SHOOTING_SPEED
    player_speed = Settings.PLAYER_SPEED

    body_pos_x, body_pos_y = engine.get_position(player_id)
    ball_pos_x, ball_pos_y = engine.get_position(-1)
    player_angle = engine.get_angle(player_id)
    left_team = engine.get_left_team(player_id)

    dx_ball = (ball_pos_x - body_pos_x) / dim_x
    dy_ball = (ball_pos_y - body_pos_y) / dim_y

    dx_left_goal  = (left_goal_position[0]  - body_pos_x) / dim_x
    dy_left_goal  = (left_goal_position[1]  - body_pos_y) / dim_y
    dx_right_goal = (right_goal_position[0] - body_pos_x) / dim_x
    dy_right_goal = (right_goal_position[1] - body_pos_y) / dim_y

    if left_team:
        sin_a = math.sin(player_angle)
        cos_a = math.cos(player_angle)

        own_goal_dx, own_goal_dy = dx_left_goal, dy_left_goal
        opp_goal_dx, opp_goal_dy = dx_right_goal, dy_right_goal
        ball_dx, ball_dy = dx_ball, dy_ball
    else:
        angle = player_angle + math.pi
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
            vision_array[8] = int(engine.get_can_shoot(player_id))

    else:

        body_vel_x, body_vel_y = engine.get_velocity(player_id)
        ball_vel_x, ball_vel_y = engine.get_velocity(-1)

        denom = shooting_speed + player_speed
        ball_vx_rel = (ball_vel_x - body_vel_x) / denom
        ball_vy_rel = (ball_vel_y - body_vel_y) / denom

        if not left_team:
            ball_vx_rel = -ball_vx_rel
            ball_vy_rel = -ball_vy_rel

        vision_array[8:10] = (ball_vx_rel, ball_vy_rel)

        insert_index = 10
        n_players = engine.get_n_players()

        # Teammates first, then opponents — keeps the slot index of teammates
        # constant across players for the shared policy.
        for pass_teammates in (True, False):
            for other_id in range(n_players):
                if other_id == player_id:
                    continue
                other_is_teammate = (engine.get_left_team(other_id) == left_team)
                if other_is_teammate != pass_teammates:
                    continue

                other_pos_x, other_pos_y = engine.get_position(other_id)

                dx_other_player_ball = (ball_pos_x - other_pos_x) / dim_x
                dy_other_player_ball = (ball_pos_y - other_pos_y) / dim_y

                if not left_team:
                    dx_other_player_ball = -dx_other_player_ball
                    dy_other_player_ball = -dy_other_player_ball

                vision_array[insert_index:insert_index + 2] = (
                    dx_other_player_ball,
                    dy_other_player_ball,
                )
                insert_index += 2

                if n_players > 2:
                    vision_array[insert_index] = 1.0 if other_is_teammate else -1.0
                    insert_index += 1

        score = engine._score if hasattr(engine, "_score") else (0.0, 0.0)
        score_left, score_right = float(score[0]), float(score[1])
        if left_team:
            my_s, opp_s = score_left, score_right
        else:
            my_s, opp_s = score_right, score_left
        vision_array[insert_index]     = my_s / 10.0
        vision_array[insert_index + 1] = opp_s / 10.0

    return vision_array


def getGlobalVision(engine, left_goal_position, right_goal_position, phantom_player=None) -> np.ndarray:

    dim_x = Settings.DIM_X
    dim_y = Settings.DIM_Y

    ball_pos_x, ball_pos_y = engine.get_position(-1)
    ball_vel_x, ball_vel_y = engine.get_velocity(-1)

    vision = [
        ball_pos_x / dim_x,
        ball_pos_y / dim_y,
        ball_vel_x / (Settings.SHOOTING_SPEED + Settings.PLAYER_SPEED),
        ball_vel_y / (Settings.SHOOTING_SPEED + Settings.PLAYER_SPEED),
    ]

    for player_id in range(engine.get_n_players()):
        pos_x, pos_y = engine.get_position(player_id)
        a = engine.get_angle(player_id)
        lt = engine.get_left_team(player_id)

        vision.extend([
            pos_x / dim_x,
            pos_y / dim_y,
            math.sin(a),
            math.cos(a),
            1.0 if lt else -1.0
        ])

    score = engine._score if hasattr(engine, "_score") else (0.0, 0.0)
    vision.append(float(score[0]) / 10.0)
    vision.append(float(score[1]) / 10.0)

    return np.asarray(vision, dtype=np.float32)
