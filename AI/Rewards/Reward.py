# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 20:27:17 2025

@author: quent
"""

import numpy as np
import math

import Settings
from Engine.Actions import canShoot


def computeReward(*args, **kwargs):
    """
    Backward-compatible reward function.
    Accepts either:
      computeReward(player, action, ball, left_goal_position, right_goal_position, score, training_progression=0.0)
    or:
      computeReward(coeff_dict, player, action, ball, left_goal_position, right_goal_position, score, training_progression=0.0)
    """
    if len(args) >= 7 and isinstance(args[0], dict):
        coeff_dict = args[0]
        player, action, ball, left_goal_position, right_goal_position, score = args[1:7]
        training_progression = args[7] if len(args) > 7 else kwargs.get("training_progression", 0.0)
    else:
        coeff_dict = {
            "static_reward": -0.005,
            "delta_ball_player_coeff": 0.05,
            "delta_ball_goal_coeff": 0.1,
            "can_shoot_coeff": 0.01,
            "goal_coeff": 5.0,
            "wrong_goal_coeff": -5.0,
            "align_ball_coeff": 0.0,
            "align_goal_coeff": 0.0,
            "shoot_when_can_coeff": 0.0,
            "shoot_without_ball_coeff": 0.0,
        }
        player, action, ball, left_goal_position, right_goal_position, score = args[:6]
        training_progression = args[6] if len(args) > 6 else kwargs.get("training_progression", 0.0)

    body, shape = player
    ball_body, _ = ball

    demi_goal_len = math.floor(Settings.GOAL_LEN / 2)
    delta_time = Settings.DELTA_TIME
    player_speed = Settings.PLAYER_SPEED
    shooting_speed = Settings.SHOOTING_SPEED

    alpha, beta = 1.0, 1.0

    static_reward = coeff_dict.get("static_reward", -0.005)
    delta_ball_player_coeff = coeff_dict.get("delta_ball_player_coeff", 0.05)
    delta_ball_goal_coeff = coeff_dict.get("delta_ball_goal_coeff", 0.1)
    can_shoot_coeff = coeff_dict.get("can_shoot_coeff", 0.01)
    goal_coeff = coeff_dict.get("goal_coeff", 5.0)
    wrong_goal_coeff = coeff_dict.get("wrong_goal_coeff", -goal_coeff)
    align_ball_coeff = coeff_dict.get("align_ball_coeff", 0.0)
    align_goal_coeff = coeff_dict.get("align_goal_coeff", 0.0)
    shoot_center_coeff = coeff_dict.get("shoot_center_coeff", 0.0)
    shoot_when_can_coeff = coeff_dict.get("shoot_when_can_coeff", 0.0)
    shoot_without_ball_coeff = coeff_dict.get("shoot_without_ball_coeff", 0.0)

    if score[0] != 0 or score[1] != 0:
        if shape.left_team and score[0] == 1:
            return goal_coeff
        if (not shape.left_team) and score[1] == 1:
            return goal_coeff
        return wrong_goal_coeff

    # Distance to ball (delta)
    prev_dx = ball_body.previous_position[0] - body.previous_position[0]
    prev_dy = ball_body.previous_position[1] - body.previous_position[1]
    prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)

    curr_dx = ball_body.position[0] - body.position[0]
    curr_dy = ball_body.position[1] - body.position[1]
    curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)

    delta = prev_dist - curr_dist
    delta_ball_player_reward = 0.0
    if abs(delta) > (player_speed * delta_time / 1000) / 1000:
        delta_ball_player_reward = delta_ball_player_coeff * delta / (player_speed * delta_time / 1000)
        delta_ball_player_reward = max(-delta_ball_player_coeff, min(delta_ball_player_reward, delta_ball_player_coeff))

    # Distance of ball to opponent goal (delta)
    if shape.left_team:
        gx, gy = right_goal_position
        gy_top, gy_bottom = gy - demi_goal_len, gy + demi_goal_len
    else:
        gx, gy = left_goal_position
        gy_top, gy_bottom = gy - demi_goal_len, gy + demi_goal_len

    prev_dist = point_to_segment_distance(
        ball_body.previous_position[0], ball_body.previous_position[1], gx, gy_top, gx, gy_bottom
    )
    curr_dist = point_to_segment_distance(
        ball_body.position[0], ball_body.position[1], gx, gy_top, gx, gy_bottom
    )

    delta = prev_dist - curr_dist
    delta_ball_goal_reward = 0.0
    if abs(delta) > (shooting_speed * delta_time / 1000) / 1000:
        delta_ball_goal_reward = delta_ball_goal_coeff * delta / (shooting_speed * delta_time / 1000) * 4
        delta_ball_goal_reward = max(-delta_ball_goal_coeff, min(delta_ball_goal_reward, delta_ball_goal_coeff))

    # Shooting and alignment shaping
    can_shoot_now = canShoot(body, ball_body)
    can_shoot_reward = can_shoot_coeff if can_shoot_now else 0.0
    is_shoot = (isinstance(action, np.ndarray) and float(action[2]) > 0.1) or action == 3
    if is_shoot:
        can_shoot_reward += shoot_when_can_coeff if can_shoot_now else -shoot_without_ball_coeff

    align_ball_reward = get_alignment_reward(body, ball_body.position, align_ball_coeff)
    align_goal_reward = 0.0
    if can_shoot_now:
        align_goal_reward = get_goal_alignment_reward(body, shape, left_goal_position, right_goal_position, align_goal_coeff)

    shoot_center_reward = 0.0
    if is_shoot and can_shoot_now and shoot_center_coeff > 0.0:
        shoot_center_reward = shoot_center_coeff * get_goal_alignment_cos(body, shape, left_goal_position, right_goal_position)

    reward = (
        static_reward
        + delta_ball_goal_reward
        + delta_ball_player_reward
        + can_shoot_reward
        + align_ball_reward
        + align_goal_reward
        + shoot_center_reward
    )

    return reward

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Returns shortest distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
    # Vector projection approach
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1

    c1 = wx * vx + wy * vy
    if c1 <= 0:
        return np.hypot(px - x1, py - y1)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return np.hypot(px - x2, py - y2)

    b = c1 / c2
    bx, by = x1 + b * vx, y1 + b * vy
    return np.hypot(px - bx, py - by)


def get_alignment_reward(body, target_pos, align_coeff):
    if align_coeff == 0.0:
        return 0.0
    vec = np.array([target_pos[0] - body.position[0], target_pos[1] - body.position[1]], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return 0.0
    vec /= norm
    facing = np.array([np.cos(body.angle), np.sin(body.angle)], dtype=np.float32)
    cos_sim = np.dot(facing, vec)
    return align_coeff * max(0.0, float(cos_sim))


def get_goal_alignment_reward(body, shape, left_goal_position, right_goal_position, align_coeff):
    if align_coeff == 0.0:
        return 0.0
    target = right_goal_position if shape.left_team else left_goal_position
    return get_alignment_reward(body, target, align_coeff)


def get_goal_alignment_cos(body, shape, left_goal_position, right_goal_position):
    target = right_goal_position if shape.left_team else left_goal_position
    vec = np.array([target[0] - body.position[0], target[1] - body.position[1]], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return 0.0
    vec /= norm
    facing = np.array([np.cos(body.angle), np.sin(body.angle)], dtype=np.float32)
    cos_sim = np.dot(facing, vec)
    return max(0.0, float(cos_sim))



def completeComputeReward(player, action, ball, left_goal_position, right_goal_position, score, training_progression=0.0):
    
    body, shape = player
    if(score[0] != 0 or score[1] != 0):
        if shape.left_team and score[0] == 1:
            return 1
        elif (not shape.left_team) and score[1] == 1:
            return 1
        else:
            return -1
        
        
    body, shape = player
    ball_body, ball_shape = ball
    
    offset = Settings.SCREEN_OFFSET
    max_dist = Settings.MAX_DIST
    
    demi_goal_len = math.floor(Settings.GOAL_LEN/2)
    
    delta_time = Settings.DELTA_TIME
    player_speed = Settings.PLAYER_SPEED
    player_rotating_speed = Settings.PLAYER_ROT_SPEED
    shooting_speed = Settings.SHOOTING_SPEED
    
    reward = 0.0
    
    """
    if(action == 3 and not canShoot(body, ball_body)):
        return -0.5 * (1 - training_progression) -0.05"""
        
    alpha, beta = 1.0, 1.0  # relative weights
    
    static_reward = -0.001
    delta_ball_player_coeff = 0.05
    dist_ball_player_coeff = 0
    delta_ball_goal_coeff = 0.1 + delta_ball_player_coeff # éliminer la récompense négative de l'éloignement de la balle % joueur
    dist_ball_goal_coeff = 0
    out_of_bound_coeff = 0
    delta_angle_coeff = 0
    angle_diff_coeff = 0
    can_shoot_coeff = 0
    
    # Delta position of player
    dx = (body.position[0] - body.previous_position[0])
    dy = (body.position[1] - body.previous_position[1])
    delta = np.sqrt(alpha * dx**2 + beta * dy**2)
    
    # Distance to ball
    prev_dx = (ball_body.previous_position[0] - body.previous_position[0])
    prev_dy = (ball_body.previous_position[1] - body.previous_position[1])
    prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
    
    curr_dx = (ball_body.position[0] - body.position[0])
    curr_dy = (ball_body.position[1] - body.position[1])
    curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    
    delta = prev_dist - curr_dist
    delta_ball_player_reward = 0
    if(abs(delta) > (player_speed * delta_time/1000)/1000):
        delta_ball_player_reward = delta_ball_player_coeff* delta / (player_speed * delta_time/1000)
        delta_ball_player_reward = max(-delta_ball_player_coeff, min(delta_ball_player_reward, delta_ball_player_coeff))
        print(f"{delta=}")
        print(f"{delta_ball_player_reward=}")
        
    dist_ball_player_reward = -dist_ball_player_coeff* curr_dist/max_dist
    
    
    # Distance of ball to opponent goal
    if shape.left_team:
        gx, gy = right_goal_position
        gy_top, gy_bottom = gy - demi_goal_len, gy + demi_goal_len
    else:
        gx, gy = left_goal_position
        gy_top, gy_bottom = gy - demi_goal_len, gy + demi_goal_len
    
    prev_dist = point_to_segment_distance(
        ball_body.previous_position[0], ball_body.previous_position[1],
        gx, gy_top, gx, gy_bottom
    )
    curr_dist = point_to_segment_distance(
        ball_body.position[0], ball_body.position[1],
        gx, gy_top, gx, gy_bottom
    )
    
    delta = prev_dist - curr_dist
    delta_ball_goal_reward = 0
    if abs(delta) > (shooting_speed * delta_time / 1000) / 1000:
        delta_ball_goal_reward = delta_ball_goal_coeff * delta / (shooting_speed * delta_time / 1000) * 4
        delta_ball_goal_reward = max(-delta_ball_goal_coeff, min(delta_ball_goal_reward, delta_ball_goal_coeff))
        print(f"{delta=}")
        print(f"{delta_ball_goal_reward=}")
        
    dist_ball_goal_reward = -dist_ball_goal_coeff* curr_dist/max_dist
        
        
    # Angle between player and ball
    vec_ball = np.array([ball_body.position[0] - body.position[0], ball_body.position[1] - body.position[1]])
    angle_to_ball = np.arctan2(vec_ball[1], vec_ball[0])
    angle_diff = (angle_to_ball - body.angle + np.pi) % (2*np.pi) - np.pi # back to -pi ; +pi
    max_angle = np.deg2rad(10)
    angle_diff_reward = 0
    if abs(angle_diff) <= max_angle:
        angle_diff_reward = angle_diff_coeff
    else:
        angle_diff_reward = - angle_diff_coeff * abs(angle_diff/np.pi)**0.25 * 0
        
    previous_vec_ball = np.array([ball_body.previous_position[0] - body.previous_position[0], ball_body.previous_position[1] - body.previous_position[1]])
    previous_angle_to_ball = np.arctan2(previous_vec_ball[1], previous_vec_ball[0])
    previous_angle_diff = (previous_angle_to_ball - body.previous_angle + np.pi) % (2*np.pi) - np.pi # back to -pi ; +pi
    
    delta = (abs(previous_angle_diff) - abs(angle_diff) + np.pi) % (2*np.pi) - np.pi
    delta_angle_reward = 0
    if(abs(delta) > (player_rotating_speed * delta_time/1000)/np.pi):
        delta_angle_reward = delta_angle_coeff * delta / (player_rotating_speed * delta_time/1000)
    
    # Penalize for being out of bounds
    x, y = body.position
    out_of_bound_reward = 0
    if (x < offset or x > Settings.DIM_X + offset):
        out_of_bound_reward = -out_of_bound_coeff
        
    can_shoot_reward = 0
    if(canShoot(body, ball_body)):
        can_shoot_reward = can_shoot_coeff
    
    reward = (static_reward + out_of_bound_reward + delta_angle_reward 
            + angle_diff_reward + dist_ball_player_reward + delta_ball_goal_reward 
            + delta_ball_player_reward + dist_ball_goal_reward + can_shoot_reward)

    return reward
