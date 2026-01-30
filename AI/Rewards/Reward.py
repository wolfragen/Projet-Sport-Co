# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 20:27:17 2025

@author: quent
"""

import numpy as np
import math

import Settings
from Engine.Actions import canShoot

def sigmoid(x):
    return 1/(1+math.exp(-x))


def computeReward(coeff_dict, player, action, ball, left_goal_position, right_goal_position, score, mean_steps, training_progression=0.0, debug=False):
        
    body, shape = player
    ball_body, ball_shape = ball
    
    reward = 0.0
        
    alpha, beta = 1.0, 1.0  # relative weights
    
    static_reward = coeff_dict["static_reward"]
    starting_static_reward = coeff_dict["starting_static_reward"]
    ending_static_reward = coeff_dict["ending_static_reward"]
    delta_ball_player_coeff = coeff_dict["delta_ball_player_coeff"]
    delta_ball_goal_coeff = coeff_dict["delta_ball_goal_coeff"]
    can_shoot_coeff = coeff_dict["can_shoot_coeff"]
    goal_coeff = coeff_dict["goal_coeff"]
    wrong_goal_coeff = coeff_dict["wrong_goal_coeff"]
    has_ball_coeff = coeff_dict["has_ball_coeff"]
    
    
    body, shape = player
    goal_reward = get_goal_reward(score, shape, goal_coeff, wrong_goal_coeff)
    if(static_reward == 0):
        static_reward = get_static_reward(starting_static_reward, ending_static_reward, mean_steps)
    
    # Delta position of player
    delta_ball_player_reward = get_delta_ball_player_reward(delta_ball_player_coeff, body, ball_body, alpha, beta)
    
    
    # Distance of ball to opponent goal
    delta_ball_goal_reward = get_delta_ball_goal_reward(shape, ball_body, right_goal_position, left_goal_position, delta_ball_goal_coeff)
        
    can_shoot_reward = get_shooting_reward(action, body, ball_body, shape, left_goal_position, right_goal_position, can_shoot_coeff)
    has_ball_reward = get_has_ball_reward(action, body, has_ball_coeff)
    
    reward = (static_reward + delta_ball_goal_reward + delta_ball_player_reward + can_shoot_reward + has_ball_reward + 
              goal_reward)

    if(debug):
        reward_dict = {
            "static_reward": static_reward,
            "delta_ball_goal_reward": delta_ball_goal_reward,
            "delta_ball_player_reward": delta_ball_player_reward,
            "can_shoot_reward": can_shoot_reward,
            "has_ball_reward": has_ball_reward,
            "goal_reward": goal_reward,
            "total_reward": reward,
            }
        return reward, reward_dict

    return reward

def get_static_reward(starting_coeff, ending_coeff, mean_steps):
    """
    Docstring for get_static_reward
    
    :param starting_coeff: Coefficient pour mean_steps = 2000
    :param ending_coeff: Coefficient pour mean_steps = 75
    :param mean_steps: moyenne de steps par partie
    :return: static_reward
    """
    sig = sigmoid((1000 - mean_steps) / 200)

    coeff = (starting_coeff - ending_coeff) / 0.9903
    static_reward = starting_coeff - coeff * sig
    return static_reward

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

def get_goal_reward(score, shape, goal_coeff, wrong_goal_coeff):
    if(score[0] != 0 or score[1] != 0):
        if shape.left_team and score[0] == 1:
            return goal_coeff
        elif (not shape.left_team) and score[1] == 1:
            return goal_coeff
        else:
            return wrong_goal_coeff
    return 0

def get_has_ball_reward(action, body, has_ball_coeff):
    if(body.canShoot and not body.hadBall):
        return has_ball_coeff
    elif(not body.canShoot and body.hadBall and action != 3):
        return -has_ball_coeff
    return 0
        
def get_shooting_reward(action, body, ball_body, shape, left_goal_position, right_goal_position, can_shoot_coeff):
    if isinstance(action, np.ndarray):
        is_shoot = float(action[2]) > 0.1
    else:
        is_shoot = action == 3

    if not (body.hadBall and is_shoot):
        return 0.0

    demi_goal_len = Settings.GOAL_LEN / 2
    r = Settings.BALL_RADIUS

    if shape.left_team:
        gx, gy = right_goal_position
    else:
        gx, gy = left_goal_position

    # Expand goal vertically by ball radius (post bounce tolerance)
    seg_top = np.array([gx, gy - demi_goal_len + r/2])
    seg_bottom = np.array([gx, gy + demi_goal_len - r/2])

    ray_origin = np.array(ball_body.position)
    ray_dir = np.array([np.cos(body.angle), np.sin(body.angle)])

    if ray_intersects_segment(ray_origin, ray_dir, seg_top, seg_bottom):
        return can_shoot_coeff

    return 0.0


def get_delta_ball_player_reward(delta_ball_player_coeff, body, ball_body, alpha, beta):
    delta_time = Settings.DELTA_TIME
    player_speed = Settings.PLAYER_SPEED
    
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
        
    return delta_ball_player_reward

def get_delta_ball_goal_reward(shape, ball_body, right_goal_position, left_goal_position, delta_ball_goal_coeff):
    demi_goal_len = math.floor(Settings.GOAL_LEN/2)
    shooting_speed = Settings.SHOOTING_SPEED
    delta_time = Settings.DELTA_TIME
    
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
        
    return delta_ball_goal_reward



def ray_intersects_segment(ray_origin, ray_dir, seg_a, seg_b):
    """
    Ray–segment intersection in 2D.
    Ray: p = ray_origin + t * ray_dir, t >= 0
    Segment: between seg_a and seg_b
    """
    v1 = ray_origin - seg_a
    v2 = seg_b - seg_a
    v3 = np.array([-ray_dir[1], ray_dir[0]])

    dot = np.dot(v2, v3)
    if abs(dot) < 1e-8:
        return False  # parallel

    t1 = np.cross(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot

    return (t1 >= 0.0) and (0.0 <= t2 <= 1.0)




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
