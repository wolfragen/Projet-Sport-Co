# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 20:27:17 2025

@author: quent
"""

import numpy as np

import Settings
from Engine.Actions import canShoot


def computeReward(player, ball, left_goal_position, right_goal_position, score, training_progression=0.0):
    
    body, shape = player
    
    if(score[0] != 0 and score[1] != 0):
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
    
    delta_time = Settings.DELTA_TIME
    player_speed = Settings.PLAYER_SPEED
    player_rotating_speed = Settings.PLAYER_ROT_SPEED
    shooting_speed = Settings.SHOOTING_SPEED
    
    reward = 0.0
        
    alpha, beta = 1.0, 0.5  # relative weights
    
    static_reward = -0.001
    delta_ball_player_coeff = 0
    dist_ball_player_coeff = 0.002 * (1 - training_progression) *0 # mettre au carré ?
    delta_ball_goal_coeff = 0
    dist_ball_goal_coeff = 0.02 * (1 - training_progression) *0
    out_of_bound_coeff = 0
    delta_angle_coeff = 0
    angle_diff_coeff = 0.01
    can_shoot_coeff = 0.02 * np.sqrt(1 - training_progression) *0
    
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
        
    dist_ball_player_reward = -dist_ball_player_coeff* curr_dist/max_dist
    
    
    # Distance of ball to opponent goal
    if shape.left_team:
        prev_dx = (right_goal_position[0] - ball_body.previous_position[0])
        prev_dy = (right_goal_position[1] - ball_body.previous_position[1])
        prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
        
        curr_dx = (right_goal_position[0] - ball_body.previous_position[0])
        curr_dy = (right_goal_position[1] - ball_body.previous_position[1])
        curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    else:
        prev_dx = (left_goal_position[0] - ball_body.previous_position[0])
        prev_dy = (left_goal_position[1] - ball_body.previous_position[1])
        prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
        
        curr_dx = (left_goal_position[0] - ball_body.previous_position[0])
        curr_dy = (left_goal_position[1] - ball_body.previous_position[1])
        curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    
    delta = prev_dist - curr_dist
    delta_ball_goal_reward = 0
    if(abs(delta) > (shooting_speed * delta_time/1000)/1000):
        delta_ball_goal_reward = delta_ball_goal_coeff* delta / (shooting_speed * delta_time/1000)
        
    dist_ball_goal_reward = -dist_ball_goal_coeff* curr_dist/max_dist
        
        
    # Angle between player and ball
    vec_ball = np.array([ball_body.position[0] - body.position[0], ball_body.position[1] - body.position[1]])
    angle_to_ball = np.arctan2(vec_ball[1], vec_ball[0])
    angle_diff = (angle_to_ball - body.angle + np.pi) % (2*np.pi) - np.pi # back to -pi ; +pi
    max_angle = np.deg2rad(5)
    angle_diff_reward = 0
    if abs(angle_diff) <= max_angle:
        angle_diff_reward = angle_diff_coeff
    else:
        angle_diff_reward = - angle_diff_coeff * abs(angle_diff/np.pi)**0.25
        
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


"""
def compute_reward(game: dict, player: tuple[pymunk.Body, pymunk.Shape], scored: tuple[bool,bool], debug: bool =False) -> float:
    
    Compute the reward for a given player based on the current game state.
    
    Parameters
   -------
    game : dict
        Current game state.
    player : tuple(pymunk.Body, pymunk.Shape)
        The player to compute reward for.
    scored : tuple[bool,bool]
        (has_scored, left_team_scored)
    
    Returns
   ----
    float
        The reward value for this player at this timestep.
    
    body, shape = player
    ball_body, ball_shape = game["ball"]
    has_scored, left_team_scored = scored
    
    left_goal = game["left_goal_position"]
    right_goal = game["right_goal_position"]
    
    offset = Settings.SCREEN_OFFSET
    max_dist = Settings.MAX_DIST
    
    delta_time = Settings.DELTA_TIME
    player_speed = Settings.PLAYER_SPEED
    player_rotating_speed = Settings.PLAYER_ROT_SPEED
    shooting_speed = Settings.SHOOTING_SPEED
    
    reward = 0.0
        
    alpha, beta = 1.0, 0.5  # relative weights
    
    static_reward = 0
    delta_ball_player_coeff = 0
    dist_ball_player_coeff = 0
    delta_ball_goal_coeff = 0
    dist_ball_goal_coeff = 0
    goal_coeff = 0
    out_of_bound_coeff = 0
    delta_angle_coeff = 0
    angle_diff_coeff = 1
    can_shoot_coeff = 0
    
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
        
    dist_ball_player_reward = -dist_ball_player_coeff* curr_dist/max_dist
    
    
    # Distance of ball to opponent goal
    if shape.left_team:
        prev_dx = (right_goal[0] - ball_body.previous_position[0])
        prev_dy = (right_goal[1] - ball_body.previous_position[1])
        prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
        
        curr_dx = (right_goal[0] - ball_body.previous_position[0])
        curr_dy = (right_goal[1] - ball_body.previous_position[1])
        curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    else:
        prev_dx = (left_goal[0] - ball_body.previous_position[0])
        prev_dy = (left_goal[1] - ball_body.previous_position[1])
        prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
        
        curr_dx = (left_goal[0] - ball_body.previous_position[0])
        curr_dy = (left_goal[1] - ball_body.previous_position[1])
        curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    
    delta = prev_dist - curr_dist
    delta_ball_goal_reward = 0
    if(abs(delta) > (shooting_speed * delta_time/1000)/1000):
        delta_ball_goal_reward = delta_ball_goal_coeff* delta / (shooting_speed * delta_time/1000)
        
    dist_ball_goal_reward = -dist_ball_goal_coeff* curr_dist/max_dist
        
        
    # Angle between player and ball
    vec_ball = np.array([ball_body.position[0] - body.position[0], ball_body.position[1] - body.position[1]])
    angle_to_ball = np.arctan2(vec_ball[1], vec_ball[0])
    angle_diff = (angle_to_ball - body.angle + np.pi) % (2*np.pi) - np.pi # back to -pi ; +pi
    max_angle = np.deg2rad(10)
    angle_diff_reward = 0
    if abs(angle_diff) <= max_angle:
        angle_diff_reward = angle_diff_coeff*5
    else:
        angle_diff_reward = angle_diff_coeff * (1 - abs(angle_diff/np.pi)*2)
        
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
    if(Actions.canShoot(body, ball_body)):
        can_shoot_reward = can_shoot_coeff
    
    # Scoring
    goal_reward = 0
    if(has_scored):
        # goal
        if shape.left_team and left_team_scored:
            goal_reward = goal_coeff
        elif (not shape.left_team) and (not left_team_scored):
            goal_reward = goal_coeff
        else:
            goal_reward = -goal_coeff
    
    reward = (static_reward + goal_reward + out_of_bound_reward + delta_angle_reward 
            + angle_diff_reward + dist_ball_player_reward + delta_ball_goal_reward 
            + delta_ball_player_reward + dist_ball_goal_reward + can_shoot_reward)
    if(debug):
        print("------------")
        if(static_reward != 0): print(f"{static_reward=}")
        if(goal_reward != 0): print(f"{goal_reward=}")
        if(out_of_bound_reward != 0): print(f"{out_of_bound_reward=}")
        if(delta_angle_reward != 0): print(f"{delta_angle_reward=}")
        if(angle_diff_reward != 0): print(f"{angle_diff_reward=}")
        if(dist_ball_player_reward != 0): print(f"{dist_ball_player_reward=}")
        if(delta_ball_goal_reward != 0): print(f"{delta_ball_goal_reward=}")
        if(dist_ball_goal_reward != 0): print(f"{dist_ball_goal_reward=}")
        if(delta_ball_player_reward != 0): print(f"{delta_ball_player_reward=}")
        if(can_shoot_reward != 0): print(f"{can_shoot_reward=}")
        print(f"TOTAL REWARD = {reward}")
        print("------------")
    
    
    return reward/10
"""
