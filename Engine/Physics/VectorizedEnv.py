# -*- coding: utf-8 -*-
"""
Vectorized environment: N games running in parallel in a single process.
All state is stored in batched numpy arrays of shape (batch_size, n_bodies).
One Numba call steps all environments simultaneously.
"""

import numpy as np
import math
from random import randint, random

from Settings import Settings
from Engine.Physics import numba_env


class VectorizedEnv:
    """N environments batched in flat numpy arrays."""

    def __init__(self, batch_size, players_number, reward_coeff_dict):
        self.batch_size = batch_size
        self.players_number = players_number
        n_left, n_right = players_number
        self.n_players = n_left + n_right
        self.n_bodies = self.n_players + 1  # +1 for ball

        B = batch_size
        N = self.n_bodies

        # Physics state: (batch, n_bodies)
        self.pos_x = np.zeros((B, N), dtype=np.float64)
        self.pos_y = np.zeros((B, N), dtype=np.float64)
        self.vel_x = np.zeros((B, N), dtype=np.float64)
        self.vel_y = np.zeros((B, N), dtype=np.float64)
        self.angle = np.zeros((B, N), dtype=np.float64)
        self.ang_vel = np.zeros((B, N), dtype=np.float64)
        self.prev_pos_x = np.zeros((B, N), dtype=np.float64)
        self.prev_pos_y = np.zeros((B, N), dtype=np.float64)
        self.prev_angle = np.zeros((B, N), dtype=np.float64)

        # Physics properties (shared across batch, but stored per-env for kernel compat)
        self.mass = np.zeros(N, dtype=np.float64)
        self.inv_mass = np.zeros(N, dtype=np.float64)
        self.moment = np.zeros(N, dtype=np.float64)
        self.inv_moment = np.zeros(N, dtype=np.float64)
        self.elasticity = np.zeros(N, dtype=np.float64)
        self.friction = np.zeros(N, dtype=np.float64)
        self.shape_type = np.zeros(N, dtype=np.int32)
        self.radius = np.zeros(N, dtype=np.float64)
        self.half_w = np.zeros(N, dtype=np.float64)
        self.half_h = np.zeros(N, dtype=np.float64)

        # Game logic: (batch, n_bodies)
        self.can_shoot = np.zeros((B, N), dtype=np.int8)
        self.had_ball = np.zeros((B, N), dtype=np.int8)
        self.left_team = np.zeros(N, dtype=np.int8)  # shared

        # Scores: (batch, 2)
        self.scores = np.zeros((B, 2), dtype=np.float64)
        self.prev_scores = np.zeros((B, 2), dtype=np.float64)
        self.last_player_shoot = np.full(B, -1, dtype=np.int64)

        # Walls (shared across batch)
        n_walls = 10 if Settings.ROUND_CORNER else 6
        self.wall_ax = np.zeros(n_walls, dtype=np.float64)
        self.wall_ay = np.zeros(n_walls, dtype=np.float64)
        self.wall_bx = np.zeros(n_walls, dtype=np.float64)
        self.wall_by = np.zeros(n_walls, dtype=np.float64)
        self.wall_radius = np.zeros(n_walls, dtype=np.float64)
        self.wall_elasticity = np.zeros(n_walls, dtype=np.float64)
        self.wall_friction = np.zeros(n_walls, dtype=np.float64)
        self.n_walls = 0

        # Goals
        self.left_goal_x = 0.0
        self.left_goal_y = 0.0
        self.right_goal_x = 0.0
        self.right_goal_y = 0.0

        # Output buffers
        NP = self.n_players
        EN = Settings.ENTRY_NEURONS
        self.vision_buf = np.zeros((B, NP, EN), dtype=np.float32)
        self.global_vision_buf = np.zeros((B, 4 + NP * 5), dtype=np.float32)
        self.rewards_buf = np.zeros((B, NP), dtype=np.float64)
        self.done_buf = np.zeros(B, dtype=np.bool_)
        self.actions_buf = np.zeros((B, NP), dtype=np.int32)

        # Reward coefficients
        self.reward_coeffs = np.full(11, np.nan, dtype=np.float64)
        self._set_reward_coeffs(reward_coeff_dict)

        # Build shared properties
        self._build_properties()
        self._build_walls()

        # Initial positions template (for resets)
        self._init_positions()

    def _set_reward_coeffs(self, coeff_dict):
        mapping = [
            "static_reward", "starting_static_reward", "ending_static_reward",
            "delta_ball_player_coeff", "delta_ball_goal_coeff", "can_shoot_coeff",
            "goal_coeff", "wrong_goal_coeff", "has_ball_coeff",
            "static_lead_reward", "static_draw_reward",
        ]
        for i, key in enumerate(mapping):
            val = coeff_dict.get(key, None)
            self.reward_coeffs[i] = float('nan') if val is None else float(val)

    def _build_properties(self):
        """Set physics properties for ball (index 0) and players (1..N)."""
        # Ball
        r = Settings.BALL_RADIUS
        m = Settings.BALL_MASS
        mom = 0.5 * m * r * r
        self.shape_type[0] = 0
        self.radius[0] = r
        self.mass[0] = m
        self.inv_mass[0] = 1.0 / m
        self.moment[0] = mom
        self.inv_moment[0] = 1.0 / mom
        self.elasticity[0] = Settings.BALL_ELASTICITY
        self.friction[0] = Settings.BALL_FRICTION

        # Players
        size = Settings.PLAYER_LEN
        pm = Settings.PLAYER_MASS
        pmom = pm * (size * size + size * size) / 12.0
        half = size / 2.0
        n_left = self.players_number[0]
        for pid in range(self.n_players):
            idx = pid + 1
            self.shape_type[idx] = 1
            self.mass[idx] = pm
            self.inv_mass[idx] = 1.0 / pm
            self.moment[idx] = pmom
            self.inv_moment[idx] = 1.0 / pmom
            self.elasticity[idx] = Settings.PLAYER_ELASTICITY
            self.friction[idx] = Settings.PLAYER_FRICTION
            self.half_w[idx] = half
            self.half_h[idx] = half
            self.left_team[idx] = 1 if pid < n_left else 0

    def _build_walls(self):
        """Build wall geometry from Settings (same as NumbaEngine.build_board)."""
        offset = Settings.SCREEN_OFFSET
        width = Settings.DIM_X
        height = Settings.DIM_Y
        goal_len = Settings.GOAL_LEN
        wall_r = Settings.WALL_RADIUS
        player_len = Settings.PLAYER_LEN

        lt = offset + (height - goal_len) / 2
        lb = offset + (height + goal_len) / 2

        if Settings.ROUND_CORNER:
            segs = [
                ((offset + player_len, offset), (offset + width - player_len, offset)),
                ((offset + player_len, offset + height), (offset + width - player_len, offset + height)),
                ((offset, offset + player_len), (offset, lt)),
                ((offset, lb), (offset, offset + height - player_len)),
                ((offset + width, offset + player_len), (offset + width, lt)),
                ((offset + width, lb), (offset + width, offset + height - player_len)),
                ((offset + player_len, offset), (offset, offset + player_len)),
                ((offset + width - player_len, offset), (offset + width, offset + player_len)),
                ((offset + player_len, offset + height), (offset, offset + height - player_len)),
                ((offset + width - player_len, offset + height), (offset + width, offset + height - player_len)),
            ]
        else:
            segs = [
                ((offset, offset), (offset + width, offset)),
                ((offset, offset + height), (offset + width, offset + height)),
                ((offset, offset), (offset, lt)),
                ((offset, lb), (offset, offset + height)),
                ((offset + width, offset), (offset + width, lt)),
                ((offset + width, lb), (offset + width, offset + height)),
            ]

        self.n_walls = len(segs)
        for i, (a, b) in enumerate(segs):
            self.wall_ax[i], self.wall_ay[i] = a
            self.wall_bx[i], self.wall_by[i] = b
            self.wall_radius[i] = wall_r
            self.wall_elasticity[i] = Settings.WALL_ELASTICITY
            self.wall_friction[i] = Settings.WALL_FRICTION

        self.left_goal_x = offset
        self.left_goal_y = (lt + lb) / 2
        self.right_goal_x = offset + width
        self.right_goal_y = (lt + lb) / 2

    def _init_positions(self):
        """Compute initial player positions (reused on reset)."""
        from Engine.Entity.Player import spacing
        offset = Settings.SCREEN_OFFSET
        dim_x, dim_y = Settings.DIM_X, Settings.DIM_Y
        n_left, n_right = self.players_number

        self._init_player_pos = []
        self._init_player_angles = []

        left_pos = spacing(n_left, Settings.PLAYER_LEN, offset, offset, dim_x / 2, dim_y)
        for i in range(n_left):
            self._init_player_pos.append((left_pos[i][0], left_pos[i][1]))
            self._init_player_angles.append(0.0)

        if n_right > 0:
            right_pos = spacing(n_right, Settings.PLAYER_LEN, dim_x / 2 + offset, offset, dim_x / 2, dim_y, revert_x=True)
            for i in range(n_right):
                self._init_player_pos.append((right_pos[-i][0], right_pos[-i][1]))
                self._init_player_angles.append(math.pi)

    def reset_all(self):
        """Reset all environments."""
        for env_idx in range(self.batch_size):
            self._reset_single(env_idx)

    def reset_env(self, env_idx):
        """Reset a single environment (full reset including scores)."""
        self._reset_single(env_idx)

    def reset_after_goal(self, env_idx):
        """Reset positions only after a goal — preserve scores for multi-goal episodes."""
        offset = Settings.SCREEN_OFFSET
        dim_x, dim_y = Settings.DIM_X, Settings.DIM_Y

        # Ball
        if Settings.RANDOM_BALL_POSITION:
            self.pos_x[env_idx, 0] = randint(round(offset + dim_x * 0.1), round(offset + dim_x * 0.9))
            self.pos_y[env_idx, 0] = randint(round(offset + dim_y * 0.1), round(offset + dim_y * 0.9))
        else:
            self.pos_x[env_idx, 0] = offset + dim_x / 2
            self.pos_y[env_idx, 0] = offset + dim_y / 2
        self.vel_x[env_idx, 0] = 0.0
        self.vel_y[env_idx, 0] = 0.0
        self.angle[env_idx, 0] = 0.0
        self.ang_vel[env_idx, 0] = 0.0

        # Players
        for pid in range(self.n_players):
            idx = pid + 1
            px, py = self._init_player_pos[pid]
            base_angle = self._init_player_angles[pid]
            self.pos_x[env_idx, idx] = px
            self.pos_y[env_idx, idx] = py
            self.vel_x[env_idx, idx] = 0.0
            self.vel_y[env_idx, idx] = 0.0
            self.angle[env_idx, idx] = random() * 2 * math.pi if Settings.RANDOM_PLAYER_INIT else base_angle
            self.ang_vel[env_idx, idx] = 0.0
            self.can_shoot[env_idx, idx] = 0
            self.had_ball[env_idx, idx] = 0

        # Copy to prev
        self.prev_pos_x[env_idx] = self.pos_x[env_idx]
        self.prev_pos_y[env_idx] = self.pos_y[env_idx]
        self.prev_angle[env_idx] = self.angle[env_idx]

        # Scores are NOT reset — game continues
        self.last_player_shoot[env_idx] = -1
        self.done_buf[env_idx] = False

    def _reset_single(self, env_idx):
        offset = Settings.SCREEN_OFFSET
        dim_x, dim_y = Settings.DIM_X, Settings.DIM_Y

        # Ball
        if Settings.RANDOM_BALL_POSITION:
            self.pos_x[env_idx, 0] = randint(round(offset + dim_x * 0.1), round(offset + dim_x * 0.9))
            self.pos_y[env_idx, 0] = randint(round(offset + dim_y * 0.1), round(offset + dim_y * 0.9))
        else:
            self.pos_x[env_idx, 0] = offset + dim_x / 2
            self.pos_y[env_idx, 0] = offset + dim_y / 2
        self.vel_x[env_idx, 0] = 0.0
        self.vel_y[env_idx, 0] = 0.0
        self.angle[env_idx, 0] = 0.0
        self.ang_vel[env_idx, 0] = 0.0

        # Players
        for pid in range(self.n_players):
            idx = pid + 1
            px, py = self._init_player_pos[pid]
            base_angle = self._init_player_angles[pid]
            self.pos_x[env_idx, idx] = px
            self.pos_y[env_idx, idx] = py
            self.vel_x[env_idx, idx] = 0.0
            self.vel_y[env_idx, idx] = 0.0
            self.angle[env_idx, idx] = random() * 2 * math.pi if Settings.RANDOM_PLAYER_INIT else base_angle
            self.ang_vel[env_idx, idx] = 0.0
            self.can_shoot[env_idx, idx] = 0
            self.had_ball[env_idx, idx] = 0

        # Copy to prev
        self.prev_pos_x[env_idx] = self.pos_x[env_idx]
        self.prev_pos_y[env_idx] = self.pos_y[env_idx]
        self.prev_angle[env_idx] = self.angle[env_idx]

        # Score
        self.scores[env_idx] = 0.0
        self.prev_scores[env_idx] = 0.0
        self.last_player_shoot[env_idx] = -1
        self.done_buf[env_idx] = False

    def step(self, actions, active_mask=None, mean_steps=2000.0):
        """
        Step all environments at once.
        actions: (batch_size, n_players) int32 array
        active_mask: (batch_size,) bool array — only step active envs (None = all active)
        Returns: (visions, global_visions, rewards, dones)
        """
        np.copyto(self.actions_buf, actions)
        np.copyto(self.prev_scores, self.scores)  # snapshot scores for reward computation
        if active_mask is None:
            active_mask = np.ones(self.batch_size, dtype=np.bool_)

        _batched_step(
            self.pos_x, self.pos_y, self.vel_x, self.vel_y,
            self.angle, self.ang_vel,
            self.prev_pos_x, self.prev_pos_y, self.prev_angle,
            self.mass, self.inv_mass, self.moment, self.inv_moment,
            self.elasticity, self.friction, self.shape_type,
            self.radius, self.half_w, self.half_h,
            self.can_shoot, self.had_ball, self.left_team,
            self.n_bodies, self.n_players,
            self.wall_ax, self.wall_ay, self.wall_bx, self.wall_by,
            self.wall_radius, self.wall_elasticity, self.wall_friction,
            self.n_walls,
            self.actions_buf,
            self.scores, self.prev_scores, self.last_player_shoot,
            self.left_goal_x, self.left_goal_y,
            self.right_goal_x, self.right_goal_y,
            self.reward_coeffs,
            float(Settings.DELTA_TIME), int(Settings.DELTA_SIMU), float(Settings.GROUND_FRICTION),
            float(Settings.PLAYER_SPEED), float(Settings.PLAYER_ROT_SPEED), float(Settings.SHOOTING_SPEED),
            float(Settings.PLAYER_LEN), float(Settings.BALL_RADIUS), float(Settings.GOAL_LEN),
            float(Settings.DIM_X), float(Settings.DIM_Y), float(Settings.SCREEN_OFFSET),
            int(self.players_number[0]), int(self.players_number[1]),
            float(mean_steps),
            bool(Settings.COMPETITIVE_VISION), int(Settings.ENTRY_NEURONS),
            self.batch_size,
            self.vision_buf, self.global_vision_buf, self.rewards_buf, self.done_buf,
            active_mask,
        )

        return self.vision_buf, self.global_vision_buf, self.rewards_buf, self.done_buf

    def get_states(self, player_id):
        """Get vision states for a specific player across all envs. Shape: (batch, ENTRY_NEURONS)."""
        return self.vision_buf[:, player_id]


# ============================================================
# Batched kernel
# ============================================================

from numba import njit, prange

@njit(parallel=True, cache=True)
def _batched_step(
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel,
    prev_pos_x, prev_pos_y, prev_angle,
    mass, inv_mass, moment, inv_moment,
    elasticity, friction, shape_type, radius, half_w, half_h,
    can_shoot, had_ball, left_team,
    n_bodies, n_players,
    wall_ax, wall_ay, wall_bx, wall_by,
    wall_radius, wall_elasticity, wall_friction, n_walls,
    actions,
    scores, prev_scores, last_player_shoot,
    left_goal_x, left_goal_y, right_goal_x, right_goal_y,
    reward_coeffs,
    dt_ms, n_substeps, damping,
    player_speed, player_rot_speed, shooting_speed,
    player_len, ball_radius, goal_len,
    dim_x, dim_y, screen_offset,
    players_number_left, players_number_right,
    mean_steps,
    competitive_vision, entry_neurons,
    batch_size,
    vision_out, global_vision_out, rewards_out, done_out,
    active_mask,  # bool array (batch_size,) — skip inactive envs
):
    for env_idx in prange(batch_size):
        if not active_mask[env_idx]:
            continue
        done_out[env_idx], last_player_shoot[env_idx] = numba_env.full_game_step(
            pos_x[env_idx], pos_y[env_idx], vel_x[env_idx], vel_y[env_idx],
            angle[env_idx], ang_vel[env_idx],
            prev_pos_x[env_idx], prev_pos_y[env_idx], prev_angle[env_idx],
            mass, inv_mass, moment, inv_moment,
            elasticity, friction, shape_type, radius, half_w, half_h,
            can_shoot[env_idx], had_ball[env_idx], left_team,
            n_bodies, n_players,
            wall_ax, wall_ay, wall_bx, wall_by,
            wall_radius, wall_elasticity, wall_friction, n_walls,
            actions[env_idx],
            scores[env_idx], prev_scores[env_idx],
            last_player_shoot[env_idx],
            left_goal_x, left_goal_y, right_goal_x, right_goal_y,
            reward_coeffs,
            dt_ms, n_substeps, damping,
            player_speed, player_rot_speed, shooting_speed,
            player_len, ball_radius, goal_len,
            dim_x, dim_y, screen_offset,
            players_number_left, players_number_right,
            mean_steps,
            competitive_vision, entry_neurons,
            vision_out[env_idx], global_vision_out[env_idx], rewards_out[env_idx],
        )
