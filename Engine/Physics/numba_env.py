# -*- coding: utf-8 -*-
"""
Numba-compiled full game step: physics + actions + vision + rewards in one @njit call.
Eliminates ALL Python object creation between agent.act() calls.

Reward coefficients array layout:
    [0] static_reward        [1] starting_static_reward  [2] ending_static_reward
    [3] delta_ball_player    [4] delta_ball_goal          [5] can_shoot
    [6] goal                 [7] wrong_goal               [8] has_ball
    [9] static_lead          [10] static_draw
    Use NaN to indicate "not active".
"""

from numba import njit
from math import sqrt, cos, sin, exp, tanh, floor, pi, isnan
from Engine.Physics.numba_core import step_kernel

# ============================================================
# Reward sub-functions (all @njit)
# ============================================================

@njit(cache=True)
def _sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


@njit(cache=True)
def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = wx * vx + wy * vy
    if c1 <= 0:
        return sqrt((px - x1) ** 2 + (py - y1) ** 2)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return sqrt((px - x2) ** 2 + (py - y2) ** 2)
    b = c1 / c2
    bx, by = x1 + b * vx, y1 + b * vy
    return sqrt((px - bx) ** 2 + (py - by) ** 2)


@njit(cache=True)
def _compute_reward_for_player(
    # Player data
    p_pos_x, p_pos_y, p_prev_pos_x, p_prev_pos_y,
    p_can_shoot, p_had_ball, p_left_team,
    # Ball data
    b_pos_x, b_pos_y, b_prev_pos_x, b_prev_pos_y,
    b_last_player_shoot,
    # Game state
    score_0, score_1, prev_score_0, prev_score_1,
    # Goals
    left_goal_x, left_goal_y, right_goal_x, right_goal_y,
    # Config
    players_number_left, players_number_right,
    action, mean_steps,
    # Reward coefficients (NaN = not active)
    rc_static, rc_starting_static, rc_ending_static,
    rc_delta_ball_player, rc_delta_ball_goal, rc_can_shoot,
    rc_goal, rc_wrong_goal, rc_has_ball,
    rc_static_lead, rc_static_draw,
    # Constants
    delta_time, player_speed, shooting_speed, goal_len,
):
    reward = 0.0
    my_idx = 0 if p_left_team else 1
    opp_idx = 1 - my_idx

    # --- Static reward ---
    static_reward = 0.0
    if not isnan(rc_static):
        static_reward = rc_static
    if not isnan(rc_starting_static) and not isnan(rc_ending_static):
        sig = _sigmoid((1000.0 - mean_steps) / 200.0)
        coeff = (rc_starting_static - rc_ending_static) / 0.9903
        static_reward = rc_starting_static - coeff * sig
    if not isnan(rc_static_lead) and not isnan(rc_static_draw):
        my_score = score_0 if p_left_team else score_1
        opp_score = score_1 if p_left_team else score_0
        diff = my_score - opp_score
        if diff < -5.0:
            diff = -5.0
        elif diff > 5.0:
            diff = 5.0
        static_reward = rc_static_lead * tanh(diff) + rc_static_draw
    reward += static_reward

    # --- Goal reward ---
    if not isnan(rc_goal):
        my_score_arr = score_0 if my_idx == 0 else score_1
        my_prev = prev_score_0 if my_idx == 0 else prev_score_1
        opp_score_arr = score_0 if opp_idx == 0 else score_1
        opp_prev = prev_score_0 if opp_idx == 0 else prev_score_1
        if my_score_arr > my_prev:
            reward += rc_goal
        elif opp_score_arr > opp_prev:
            reward += rc_wrong_goal

    # --- Delta ball-player reward ---
    if not isnan(rc_delta_ball_player):
        prev_dx = b_prev_pos_x - p_prev_pos_x
        prev_dy = b_prev_pos_y - p_prev_pos_y
        prev_dist = sqrt(prev_dx * prev_dx + prev_dy * prev_dy)
        curr_dx = b_pos_x - p_pos_x
        curr_dy = b_pos_y - p_pos_y
        curr_dist = sqrt(curr_dx * curr_dx + curr_dy * curr_dy)
        delta = prev_dist - curr_dist
        threshold = (player_speed * delta_time / 1000.0) / 1000.0
        if abs(delta) > threshold:
            dbp = rc_delta_ball_player * delta / (player_speed * delta_time / 1000.0)
            if dbp < -rc_delta_ball_player:
                dbp = -rc_delta_ball_player
            elif dbp > rc_delta_ball_player:
                dbp = rc_delta_ball_player
            reward += dbp

    # --- Delta ball-goal reward ---
    if not isnan(rc_delta_ball_goal):
        demi = floor(goal_len / 2.0)
        if p_left_team:
            gx, gy = right_goal_x, right_goal_y
        else:
            gx, gy = left_goal_x, left_goal_y
        gy_top = gy - demi
        gy_bottom = gy + demi
        prev_dist = _point_to_segment_distance(b_prev_pos_x, b_prev_pos_y, gx, gy_top, gx, gy_bottom)
        curr_dist = _point_to_segment_distance(b_pos_x, b_pos_y, gx, gy_top, gx, gy_bottom)
        delta = prev_dist - curr_dist
        threshold = (shooting_speed * delta_time / 1000.0) / 1000.0
        if abs(delta) > threshold:
            dbg = rc_delta_ball_goal * delta / (shooting_speed * delta_time / 1000.0) * 4.0
            if dbg < -rc_delta_ball_goal:
                dbg = -rc_delta_ball_goal
            elif dbg > rc_delta_ball_goal:
                dbg = rc_delta_ball_goal
            reward += dbg

    # --- Shooting reward ---
    if not isnan(rc_can_shoot):
        n_left = players_number_left
        if b_last_player_shoot >= 0:  # -1 means None
            if score_0 > prev_score_0 and p_left_team and b_last_player_shoot < n_left:
                reward += rc_can_shoot
            elif score_1 > prev_score_1 and (not p_left_team) and b_last_player_shoot >= n_left:
                reward += rc_can_shoot

    # --- Has ball reward ---
    if not isnan(rc_has_ball):
        if p_can_shoot and not p_had_ball:
            reward += rc_has_ball
        elif not p_can_shoot and p_had_ball and action != 3:
            reward -= rc_has_ball

    return reward


# ============================================================
# Vision computation (@njit)
# ============================================================

@njit(cache=True)
def _compute_vision(
    player_idx,  # 1-based index in arrays (player_id + 1)
    pos_x, pos_y, vel_x, vel_y, angle, left_team,
    n_bodies, n_players,
    left_goal_x, left_goal_y, right_goal_x, right_goal_y,
    dim_x, dim_y, shooting_speed, player_speed,
    competitive_vision, entry_neurons,
    score_left, score_right,
    vision_out,  # pre-allocated output array (entry_neurons,)
):
    bx, by = pos_x[0], pos_y[0]  # ball
    px, py = pos_x[player_idx], pos_y[player_idx]
    pa = angle[player_idx]
    lt = left_team[player_idx]

    dx_ball = (bx - px) / dim_x
    dy_ball = (by - py) / dim_y
    dx_lg = (left_goal_x - px) / dim_x
    dy_lg = (left_goal_y - py) / dim_y
    dx_rg = (right_goal_x - px) / dim_x
    dy_rg = (right_goal_y - py) / dim_y

    if lt:
        sin_a = sin(pa)
        cos_a = cos(pa)
        own_gx, own_gy = dx_lg, dy_lg
        opp_gx, opp_gy = dx_rg, dy_rg
        ball_dx, ball_dy = dx_ball, dy_ball
    else:
        a2 = pa + pi
        sin_a = sin(a2)
        cos_a = cos(a2)
        own_gx, own_gy = -dx_rg, -dy_rg
        opp_gx, opp_gy = -dx_lg, -dy_lg
        ball_dx, ball_dy = -dx_ball, -dy_ball

    vision_out[0] = sin_a
    vision_out[1] = cos_a
    vision_out[2] = ball_dx
    vision_out[3] = ball_dy
    vision_out[4] = own_gx
    vision_out[5] = own_gy
    vision_out[6] = opp_gx
    vision_out[7] = opp_gy

    if competitive_vision:
        bvx, bvy = vel_x[0], vel_y[0]
        pvx, pvy = vel_x[player_idx], vel_y[player_idx]
        denom = shooting_speed + player_speed
        rel_vx = (bvx - pvx) / denom
        rel_vy = (bvy - pvy) / denom
        if not lt:
            rel_vx = -rel_vx
            rel_vy = -rel_vy
        vision_out[8] = rel_vx
        vision_out[9] = rel_vy

        insert = 10
        # Teammates first, then opponents — keeps the slot index of teammates
        # constant across players for the shared policy.
        for pass_idx in range(2):
            want_teammate = (pass_idx == 0)
            for other_idx in range(1, n_players + 1):  # 1-based
                if other_idx == player_idx:
                    continue
                other_lt = left_team[other_idx]
                is_teammate = (other_lt == lt)
                if is_teammate != want_teammate:
                    continue
                ox, oy = pos_x[other_idx], pos_y[other_idx]
                dx_ob = (bx - ox) / dim_x
                dy_ob = (by - oy) / dim_y
                if not lt:
                    dx_ob = -dx_ob
                    dy_ob = -dy_ob
                vision_out[insert] = dx_ob
                vision_out[insert + 1] = dy_ob
                insert += 2
                if n_players > 2:
                    vision_out[insert] = 1.0 if is_teammate else -1.0
                    insert += 1

        if lt:
            my_s, opp_s = score_left, score_right
        else:
            my_s, opp_s = score_right, score_left
        vision_out[insert] = my_s / 10.0
        vision_out[insert + 1] = opp_s / 10.0


@njit(cache=True)
def _compute_global_vision(
    pos_x, pos_y, vel_x, vel_y, angle, left_team,
    n_bodies, n_players,
    dim_x, dim_y, shooting_speed, player_speed,
    score_left, score_right,
    global_vision_out,  # pre-allocated
):
    denom = shooting_speed + player_speed
    global_vision_out[0] = pos_x[0] / dim_x
    global_vision_out[1] = pos_y[0] / dim_y
    global_vision_out[2] = vel_x[0] / denom
    global_vision_out[3] = vel_y[0] / denom

    idx = 4
    for i in range(1, n_players + 1):
        global_vision_out[idx] = pos_x[i] / dim_x
        global_vision_out[idx + 1] = pos_y[i] / dim_y
        global_vision_out[idx + 2] = sin(angle[i])
        global_vision_out[idx + 3] = cos(angle[i])
        global_vision_out[idx + 4] = 1.0 if left_team[i] else -1.0
        idx += 5

    global_vision_out[idx] = score_left / 10.0
    global_vision_out[idx + 1] = score_right / 10.0


# ============================================================
# Mega-kernel: full game step
# ============================================================

@njit(cache=True)
def full_game_step(
    # === Engine state arrays (modified in-place) ===
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel,
    prev_pos_x, prev_pos_y, prev_angle,
    mass, inv_mass, moment, inv_moment,
    elasticity, friction, shape_type, radius, half_w, half_h,
    can_shoot, had_ball, left_team,
    n_bodies, n_players,
    # === Walls ===
    wall_ax, wall_ay, wall_bx, wall_by,
    wall_radius, wall_elasticity, wall_friction, n_walls,
    # === Actions (one per player, discrete int) ===
    actions,  # int32 array (n_players,) : 0=forward, 1=left, 2=right, 3=shoot
    # === Game state ===
    score,       # float64 array (2,) — modified in-place
    prev_score,  # float64 array (2,)
    last_player_shoot,  # int: -1 = None, >=0 = player_id
    # === Goals ===
    left_goal_x, left_goal_y, right_goal_x, right_goal_y,
    # === Reward coefficients (NaN = inactive) ===
    reward_coeffs,  # float64 array (11,)
    # === Settings constants ===
    dt_ms, n_substeps, damping,
    player_speed, player_rot_speed, shooting_speed,
    player_len, ball_radius, goal_len,
    dim_x, dim_y, screen_offset,
    players_number_left, players_number_right,
    mean_steps,
    competitive_vision, entry_neurons,
    # === Output arrays (pre-allocated) ===
    vision_out,         # float32 (n_players, entry_neurons)
    global_vision_out,  # float32 (global_vision_size,)
    rewards_out,        # float64 (n_players,)
):
    """
    Complete game step: actions → physics → game logic → rewards → vision.
    Returns: (done, updated_last_player_shoot)
    """

    # ----- 1. Snapshot previous positions -----
    for i in range(n_bodies):
        prev_pos_x[i] = pos_x[i]
        prev_pos_y[i] = pos_y[i]
        prev_angle[i] = angle[i]

    # ----- 2. Apply actions -----
    half_player = player_len / 2.0
    for pid in range(n_players):
        idx = pid + 1  # 1-based index in arrays
        a = actions[pid]

        if a == 0:  # Move forward
            ang = angle[idx]
            vel_x[idx] = player_speed * cos(ang)
            vel_y[idx] = player_speed * sin(ang)
            ang_vel[idx] = 0.0
        elif a == 1:  # Rotate left
            vel_x[idx] = 0.0
            vel_y[idx] = 0.0
            ang_vel[idx] = -player_rot_speed
        elif a == 2:  # Rotate right
            vel_x[idx] = 0.0
            vel_y[idx] = 0.0
            ang_vel[idx] = player_rot_speed
        elif a == 3:  # Shoot
            vel_x[idx] = 0.0
            vel_y[idx] = 0.0
            ang_vel[idx] = 0.0
            if can_shoot[idx]:
                pvx, pvy = vel_x[idx], vel_y[idx]
                spd = sqrt(pvx * pvx + pvy * pvy) + shooting_speed
                pang = angle[idx]
                angle[0] = pang
                vel_x[0] = spd * cos(pang)
                vel_y[0] = spd * sin(pang)
                ang_vel[0] = 0.0
                last_player_shoot = pid
        else:  # No action / invalid
            vel_x[idx] = 0.0
            vel_y[idx] = 0.0
            ang_vel[idx] = 0.0

    # ----- 3. Physics step -----
    step_kernel(
        pos_x, pos_y, vel_x, vel_y, angle, ang_vel,
        mass, inv_mass, moment, inv_moment,
        elasticity, friction, shape_type, radius, half_w, half_h,
        n_bodies,
        wall_ax, wall_ay, wall_bx, wall_by,
        wall_radius, wall_elasticity, wall_friction, n_walls,
        dt_ms, n_substeps, damping
    )

    # ----- 4. Save last_player_shoot before goal check -----
    temp_last_shoot = last_player_shoot

    # ----- 5. Check if goal -----
    done = False
    ball_x = pos_x[0]
    if ball_x < screen_offset:
        score[1] += 1.0
        done = True
    elif ball_x > dim_x + screen_offset:
        score[0] += 1.0
        done = True

    # Restore last_player_shoot (goal check may have side effects in original code)
    last_player_shoot = temp_last_shoot

    # ----- 6. Reset movements -----
    for idx in range(1, n_players + 1):
        vel_x[idx] = 0.0
        vel_y[idx] = 0.0
        ang_vel[idx] = 0.0

    # ----- 7. Check players can shoot -----
    for pid in range(n_players):
        idx = pid + 1
        had_ball[idx] = can_shoot[idx]

        # canShoot: distance from front face to ball edge
        px, py = pos_x[idx], pos_y[idx]
        ang = angle[idx]
        front_x = px + half_player * cos(ang)
        front_y = py + half_player * sin(ang)
        dx = front_x - pos_x[0]
        dy = front_y - pos_y[0]
        dist = sqrt(dx * dx + dy * dy)
        dist_to_edge = dist - ball_radius
        can = dist_to_edge <= player_len / 3.0  # PLAYER_SHOOTING_RANGE, scales with entity size

        can_shoot[idx] = can
        if can and pid != last_player_shoot:
            last_player_shoot = -1  # None

    # ----- 8. Compute rewards -----
    for pid in range(n_players):
        idx = pid + 1
        rewards_out[pid] = _compute_reward_for_player(
            pos_x[idx], pos_y[idx], prev_pos_x[idx], prev_pos_y[idx],
            can_shoot[idx], had_ball[idx], bool(left_team[idx]),
            pos_x[0], pos_y[0], prev_pos_x[0], prev_pos_y[0],
            last_player_shoot,
            score[0], score[1], prev_score[0], prev_score[1],
            left_goal_x, left_goal_y, right_goal_x, right_goal_y,
            players_number_left, players_number_right,
            actions[pid], mean_steps,
            reward_coeffs[0], reward_coeffs[1], reward_coeffs[2],
            reward_coeffs[3], reward_coeffs[4], reward_coeffs[5],
            reward_coeffs[6], reward_coeffs[7], reward_coeffs[8],
            reward_coeffs[9], reward_coeffs[10],
            dt_ms, player_speed, shooting_speed, goal_len,
        )

    # ----- 9. Check players out of bounds -----
    # Margin must keep the player's body fully inside the wall (half body +
    # wall_radius + epsilon), otherwise the next collision step can re-eject
    # the player outward through a nearby corner or goal opening.
    margin = player_len * 0.5 + 5.0
    x_min = screen_offset + margin
    x_max = dim_x + screen_offset - margin
    y_min = screen_offset + margin
    y_max = dim_y + screen_offset - margin
    for idx in range(1, n_players + 1):
        x = pos_x[idx]
        y = pos_y[idx]
        oob = False
        if x < x_min:
            pos_x[idx] = x_min
            vel_x[idx] = 0.0
            oob = True
        elif x > x_max:
            pos_x[idx] = x_max
            vel_x[idx] = 0.0
            oob = True
        if y < y_min:
            pos_y[idx] = y_min
            vel_y[idx] = 0.0
            oob = True
        elif y > y_max:
            pos_y[idx] = y_max
            vel_y[idx] = 0.0
            oob = True
        if oob:
            prev_pos_x[idx] = pos_x[idx]
            prev_pos_y[idx] = pos_y[idx]

    # ----- 10. Compute vision for all players -----
    for pid in range(n_players):
        idx = pid + 1
        _compute_vision(
            idx, pos_x, pos_y, vel_x, vel_y, angle, left_team,
            n_bodies, n_players,
            left_goal_x, left_goal_y, right_goal_x, right_goal_y,
            dim_x, dim_y, shooting_speed, player_speed,
            competitive_vision, entry_neurons,
            score[0], score[1],
            vision_out[pid],
        )

    # ----- 11. Compute global vision -----
    _compute_global_vision(
        pos_x, pos_y, vel_x, vel_y, angle, left_team,
        n_bodies, n_players,
        dim_x, dim_y, shooting_speed, player_speed,
        score[0], score[1],
        global_vision_out,
    )

    return done, last_player_shoot
