# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 20:27:17 2025

@author: quent
"""

import numpy as np
import math

from Settings import Settings


def sigmoid(x):
    return 1/(1+math.exp(-x))


def computeReward(coeff_dict, player_state, players_number, action, ball_state, left_goal_position, right_goal_position, score, previous_score, mean_steps, training_progression=0.0, debug=False):

    reward = 0.0

    alpha, beta = 1.0, 1.0  # relative weights

    static_reward = coeff_dict.get("static_reward", None)
    static_lead_reward = coeff_dict.get("static_lead_reward", None)
    static_draw_reward = coeff_dict.get("static_draw_reward", None)
    starting_static_reward = coeff_dict.get("starting_static_reward", None)
    ending_static_reward = coeff_dict.get("ending_static_reward", None)
    delta_ball_player_coeff = coeff_dict.get("delta_ball_player_coeff", None)
    delta_ball_goal_coeff = coeff_dict.get("delta_ball_goal_coeff", None)
    can_shoot_coeff = coeff_dict.get("can_shoot_coeff", None)
    goal_coeff = coeff_dict.get("goal_coeff", None)
    wrong_goal_coeff = coeff_dict.get("wrong_goal_coeff", None)
    has_ball_coeff = coeff_dict.get("has_ball_coeff", None)


    goal_reward = get_goal_reward(score, previous_score, player_state, goal_coeff, wrong_goal_coeff)
    if(static_reward == None):
        static_reward = 0
    if(starting_static_reward is not None and ending_static_reward is not None):
        static_reward = get_static_reward(starting_static_reward, ending_static_reward, mean_steps)
    if(static_lead_reward is not None and static_draw_reward is not None):
        static_reward = get_score_diff_static_reward(score, player_state, static_draw_reward, static_lead_reward)

    # Delta position of player
    delta_ball_player_reward = 0
    if(delta_ball_player_coeff is not None):
        delta_ball_player_reward = get_delta_ball_player_reward(delta_ball_player_coeff, player_state, ball_state, alpha, beta)


    # Distance of ball to opponent goal
    delta_ball_goal_reward = 0
    if(delta_ball_goal_coeff is not None):
        delta_ball_goal_reward = get_delta_ball_goal_reward(player_state, ball_state, right_goal_position, left_goal_position, delta_ball_goal_coeff)

    can_shoot_reward = 0
    if(can_shoot_coeff is not None):
        can_shoot_reward = get_shooting_reward(players_number, action, player_state, ball_state, left_goal_position, right_goal_position, can_shoot_coeff, score, previous_score)

    has_ball_reward = 0
    if(has_ball_coeff is not None):
        has_ball_reward = get_has_ball_reward(action, player_state, has_ball_coeff)

    reward = (static_reward + delta_ball_goal_reward + delta_ball_player_reward + can_shoot_reward + has_ball_reward +
              goal_reward)

    if(debug): # Attention, reward_dict noté en brut dans DQN => testing game et runtests
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
    sig = sigmoid((1000 - mean_steps) / 200)
    coeff = (starting_coeff - ending_coeff) / 0.9903
    static_reward = starting_coeff - coeff * sig
    return static_reward

def get_score_diff_static_reward(
    score,
    player_state,
    draw_penalty=-0.002,
    lead_coeff=0.004,
    max_goal_diff=5
):
    my_score = score[0] if player_state.left_team else score[1]
    opp_score = score[1] if player_state.left_team else score[0]

    diff = my_score - opp_score
    diff = max(-max_goal_diff, min(diff, max_goal_diff))

    return lead_coeff * math.tanh(diff) + draw_penalty

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Returns shortest distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
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

def get_goal_reward(score, prev_score, player_state, goal_coeff, wrong_goal_coeff):
    my_idx = 0 if player_state.left_team else 1
    opp_idx = 1 - my_idx

    if score[my_idx] > prev_score[my_idx]:
        return goal_coeff

    if score[opp_idx] > prev_score[opp_idx]:
        return wrong_goal_coeff

    return 0.0

def get_has_ball_reward(action, player_state, has_ball_coeff):
    if(player_state.can_shoot and not player_state.had_ball):
        return has_ball_coeff
    elif(not player_state.can_shoot and player_state.had_ball and action != 3):
        return -has_ball_coeff
    return 0

def get_shooting_reward(players_number, action, player_state, ball_state, left_goal_position,
                        right_goal_position, can_shoot_coeff, score, previous_score):
    n_players_left = players_number[0]
    if(ball_state.last_player_shoot is not None):
        if(score[0]>previous_score[0] and player_state.left_team and ball_state.last_player_shoot < n_players_left):
            return can_shoot_coeff
        elif(score[1]>previous_score[1] and not player_state.left_team and ball_state.last_player_shoot >= n_players_left):
            return can_shoot_coeff

    return 0.0


def get_delta_ball_player_reward(delta_ball_player_coeff, player_state, ball_state, alpha, beta):
    delta_time = Settings.DELTA_TIME
    player_speed = Settings.PLAYER_SPEED

    # Distance to ball
    prev_dx = (ball_state.previous_position[0] - player_state.previous_position[0])
    prev_dy = (ball_state.previous_position[1] - player_state.previous_position[1])
    prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)

    curr_dx = (ball_state.position[0] - player_state.position[0])
    curr_dy = (ball_state.position[1] - player_state.position[1])
    curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)

    delta = prev_dist - curr_dist
    delta_ball_player_reward = 0
    if(abs(delta) > (player_speed * delta_time/1000)/1000):
        delta_ball_player_reward = delta_ball_player_coeff* delta / (player_speed * delta_time/1000)
        delta_ball_player_reward = max(-delta_ball_player_coeff, min(delta_ball_player_reward, delta_ball_player_coeff))

    return delta_ball_player_reward

def get_delta_ball_goal_reward(player_state, ball_state, right_goal_position, left_goal_position, delta_ball_goal_coeff):
    demi_goal_len = math.floor(Settings.GOAL_LEN/2)
    shooting_speed = Settings.SHOOTING_SPEED
    delta_time = Settings.DELTA_TIME

    if player_state.left_team:
        gx, gy = right_goal_position
        gy_top, gy_bottom = gy - demi_goal_len, gy + demi_goal_len
    else:
        gx, gy = left_goal_position
        gy_top, gy_bottom = gy - demi_goal_len, gy + demi_goal_len

    prev_dist = point_to_segment_distance(
        ball_state.previous_position[0], ball_state.previous_position[1],
        gx, gy_top, gx, gy_bottom
    )
    curr_dist = point_to_segment_distance(
        ball_state.position[0], ball_state.position[1],
        gx, gy_top, gx, gy_bottom
    )

    delta = prev_dist - curr_dist
    delta_ball_goal_reward = 0
    if abs(delta) > (shooting_speed * delta_time / 1000) / 1000:
        delta_ball_goal_reward = delta_ball_goal_coeff * delta / (shooting_speed * delta_time / 1000) * 4
        delta_ball_goal_reward = max(-delta_ball_goal_coeff, min(delta_ball_goal_reward, delta_ball_goal_coeff))

    return delta_ball_goal_reward


def ray_intersects_segment(ray_origin, ray_dir, seg_a, seg_b):
    v1 = ray_origin - seg_a
    v2 = seg_b - seg_a
    v3 = np.array([-ray_dir[1], ray_dir[0]])

    dot = np.dot(v2, v3)
    if abs(dot) < 1e-8:
        return False

    t1 = np.cross(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot

    return (t1 >= 0.0) and (0.0 <= t2 <= 1.0)
