# -*- coding: utf-8 -*-
"""
NumbaEngine: Custom physics engine compiled with Numba.
Implements PhysicsEngine ABC with flat numpy arrays.

Index convention: slot 0 = ball, slots 1..N = players.
"""

import math
from typing import Optional
from random import randint, random

import numpy as np

from Settings import Settings
from Engine.Physics.PhysicsEngine import PhysicsEngine
from Engine.Physics.EntityState import PlayerState, BallState
from Engine.Physics import numba_core
from Engine.Physics import numba_env


class NumbaEngine(PhysicsEngine):

    def __init__(self):
        self._n_bodies = 0
        self._n_players = 0
        self._n_walls = 0

        # Arrays will be allocated in create_world()
        self._pos_x = None
        self._pos_y = None
        self._vel_x = None
        self._vel_y = None
        self._angle = None
        self._ang_vel = None
        self._prev_pos_x = None
        self._prev_pos_y = None
        self._prev_angle = None
        self._mass = None
        self._inv_mass = None
        self._moment = None
        self._inv_moment = None
        self._elasticity = None
        self._friction = None
        self._shape_type = None
        self._radius = None
        self._half_w = None
        self._half_h = None

        # Walls
        self._wall_ax = None
        self._wall_ay = None
        self._wall_bx = None
        self._wall_by = None
        self._wall_radius = None
        self._wall_elasticity = None
        self._wall_friction = None
        self._wall_segments_render = []

        # Game logic (not in kernel)
        self._can_shoot = None
        self._had_ball = None
        self._left_team = None
        self._colors = None
        self._last_player_shoot = -2  # -2 = None

        self._jit_warmed = False

    def _entity_to_index(self, entity_id: int) -> int:
        return 0 if entity_id == -1 else entity_id + 1

    # --- Lifecycle ---

    def create_world(self) -> None:
        n = sum(Settings.PLAYERS_NUMBER) + 1
        nw = 10 if Settings.ROUND_CORNER else 6

        self._n_bodies = n
        self._n_players = sum(Settings.PLAYERS_NUMBER)
        self._n_walls = 0  # filled by build_board

        self._pos_x = np.zeros(n, dtype=np.float64)
        self._pos_y = np.zeros(n, dtype=np.float64)
        self._vel_x = np.zeros(n, dtype=np.float64)
        self._vel_y = np.zeros(n, dtype=np.float64)
        self._angle = np.zeros(n, dtype=np.float64)
        self._ang_vel = np.zeros(n, dtype=np.float64)
        self._prev_pos_x = np.zeros(n, dtype=np.float64)
        self._prev_pos_y = np.zeros(n, dtype=np.float64)
        self._prev_angle = np.zeros(n, dtype=np.float64)
        self._mass = np.zeros(n, dtype=np.float64)
        self._inv_mass = np.zeros(n, dtype=np.float64)
        self._moment = np.zeros(n, dtype=np.float64)
        self._inv_moment = np.zeros(n, dtype=np.float64)
        self._elasticity = np.zeros(n, dtype=np.float64)
        self._friction = np.zeros(n, dtype=np.float64)
        self._shape_type = np.zeros(n, dtype=np.int32)
        self._radius = np.zeros(n, dtype=np.float64)
        self._half_w = np.zeros(n, dtype=np.float64)
        self._half_h = np.zeros(n, dtype=np.float64)

        self._wall_ax = np.zeros(nw, dtype=np.float64)
        self._wall_ay = np.zeros(nw, dtype=np.float64)
        self._wall_bx = np.zeros(nw, dtype=np.float64)
        self._wall_by = np.zeros(nw, dtype=np.float64)
        self._wall_radius = np.zeros(nw, dtype=np.float64)
        self._wall_elasticity = np.zeros(nw, dtype=np.float64)
        self._wall_friction = np.zeros(nw, dtype=np.float64)
        self._wall_segments_render = []

        self._can_shoot = np.zeros(n, dtype=np.int8)
        self._had_ball = np.zeros(n, dtype=np.int8)
        self._left_team = np.zeros(n, dtype=np.int8)
        self._colors = np.zeros((n, 4), dtype=np.int32)
        self._last_player_shoot = -1  # -1 = None

        # Pre-allocated output buffers for full_step
        n_players = n - 1
        self._actions_buf = np.zeros(n_players, dtype=np.int32)
        self._prev_score_buf = np.zeros(2, dtype=np.float64)
        self._vision_buf = np.zeros((n_players, Settings.ENTRY_NEURONS), dtype=np.float32)
        gv_size = 6 + n_players * 5
        self._global_vision_buf = np.zeros(gv_size, dtype=np.float32)
        self._rewards_buf = np.zeros(n_players, dtype=np.float64)
        self._reward_coeffs = np.full(11, np.nan, dtype=np.float64)

        # Goal positions (set by build_board)
        self._left_goal_x = 0.0
        self._left_goal_y = 0.0
        self._right_goal_x = 0.0
        self._right_goal_y = 0.0

        # Score array
        self._score = np.zeros(2, dtype=np.float64)

        # JIT warmup (first call compiles, subsequent use disk cache)
        if not self._jit_warmed:
            _dummy_n = 1
            _dummy_nw = 1
            _d = np.zeros(_dummy_n, dtype=np.float64)
            _di = np.zeros(_dummy_n, dtype=np.int32)
            _dw = np.zeros(_dummy_nw, dtype=np.float64)
            numba_core.step_kernel(
                _d.copy(), _d.copy(), _d.copy(), _d.copy(), _d.copy(), _d.copy(),
                _d.copy(), _d.copy(), _d.copy(), _d.copy(),
                _d.copy(), _d.copy(), _di.copy(), _d.copy(), _d.copy(), _d.copy(),
                _dummy_n,
                _dw.copy(), _dw.copy(), _dw.copy(), _dw.copy(),
                _dw.copy(), _dw.copy(), _dw.copy(), _dummy_nw,
                1.0, 1, 0.5
            )
            self._jit_warmed = True

    def build_board(self) -> tuple[tuple, tuple]:
        offset = Settings.SCREEN_OFFSET
        width = Settings.DIM_X
        height = Settings.DIM_Y
        goal_len = Settings.GOAL_LEN
        wall_r = Settings.WALL_RADIUS
        player_len = Settings.PLAYER_LEN
        round_corner = Settings.ROUND_CORNER

        left_top_goal = offset + (height - goal_len) / 2
        left_bottom_goal = offset + (height + goal_len) / 2
        right_top_goal = offset + (height - goal_len) / 2
        right_bottom_goal = offset + (height + goal_len) / 2

        if round_corner:
            segments = [
                ((offset + player_len, offset), (offset + width - player_len, offset)),
                ((offset + player_len, offset + height), (offset + width - player_len, offset + height)),
                ((offset, offset + player_len), (offset, left_top_goal)),
                ((offset, left_bottom_goal), (offset, offset + height - player_len)),
                ((offset + width, offset + player_len), (offset + width, right_top_goal)),
                ((offset + width, right_bottom_goal), (offset + width, offset + height - player_len)),
                ((offset + player_len, offset), (offset, offset + player_len)),
                ((offset + width - player_len, offset), (offset + width, offset + player_len)),
                ((offset + player_len, offset + height), (offset, offset + height - player_len)),
                ((offset + width - player_len, offset + height), (offset + width, offset + height - player_len)),
            ]
        else:
            segments = [
                ((offset, offset), (offset + width, offset)),
                ((offset, offset + height), (offset + width, offset + height)),
                ((offset, offset), (offset, left_top_goal)),
                ((offset, left_bottom_goal), (offset, offset + height)),
                ((offset + width, offset), (offset + width, right_top_goal)),
                ((offset + width, right_bottom_goal), (offset + width, offset + height)),
            ]

        self._n_walls = len(segments)
        for idx, (a, b) in enumerate(segments):
            self._wall_ax[idx] = a[0]
            self._wall_ay[idx] = a[1]
            self._wall_bx[idx] = b[0]
            self._wall_by[idx] = b[1]
            self._wall_radius[idx] = wall_r
            self._wall_elasticity[idx] = Settings.WALL_ELASTICITY
            self._wall_friction[idx] = Settings.WALL_FRICTION

        self._wall_segments_render = [(a, b, wall_r) for a, b in segments]

        left_goal_position = (offset, (left_bottom_goal + left_top_goal) / 2)
        right_goal_position = (offset + width, (right_bottom_goal + right_top_goal) / 2)
        self._left_goal_x = left_goal_position[0]
        self._left_goal_y = left_goal_position[1]
        self._right_goal_x = right_goal_position[0]
        self._right_goal_y = right_goal_position[1]
        return left_goal_position, right_goal_position

    def build_ball(self) -> None:
        r = Settings.BALL_RADIUS
        m = Settings.BALL_MASS
        mom = 0.5 * m * r * r  # moment_for_circle

        self._shape_type[0] = 0
        self._radius[0] = r
        self._mass[0] = m
        self._inv_mass[0] = 1.0 / m
        self._moment[0] = mom
        self._inv_moment[0] = 1.0 / mom
        self._elasticity[0] = Settings.BALL_ELASTICITY
        self._friction[0] = Settings.BALL_FRICTION

        offset = Settings.SCREEN_OFFSET
        dim_x = Settings.DIM_X
        dim_y = Settings.DIM_Y

        if Settings.RANDOM_BALL_POSITION:
            self._pos_x[0] = randint(round(offset + dim_x * 0.1), round(offset + dim_x * 0.9))
            self._pos_y[0] = randint(round(offset + dim_y * 0.1), round(offset + dim_y * 0.9))
        else:
            self._pos_x[0] = offset + dim_x / 2
            self._pos_y[0] = offset + dim_y / 2

        self._prev_pos_x[0] = self._pos_x[0]
        self._prev_pos_y[0] = self._pos_y[0]
        self._colors[0] = Settings.BALL_COLOR

    def build_players(self, players_number: tuple[int, int], human: bool = False,
                      phantom_player=None) -> Optional[int]:
        from Engine.Entity.Player import spacing

        offset = Settings.SCREEN_OFFSET
        dim_x = Settings.DIM_X
        dim_y = Settings.DIM_Y
        size = Settings.PLAYER_LEN
        m = Settings.PLAYER_MASS
        mom = m * (size * size + size * size) / 12.0
        half = size / 2.0

        n_left, n_right = players_number
        idx = 1  # slot 0 is ball

        left_positions = spacing(n_left, size, offset, offset, dim_x / 2, dim_y)
        for i in range(n_left):
            pos = left_positions[i]
            a = random() * 2 * math.pi if Settings.RANDOM_PLAYER_INIT else 0.0
            self._setup_player(idx, pos[0], pos[1], a, True, idx - 1, m, mom, half)
            idx += 1

        if n_right != 0:
            right_positions = spacing(n_right, size, dim_x / 2 + offset, offset, dim_x / 2, dim_y, revert_x=True)
            for i in range(n_right):
                pos = right_positions[-i]
                a = random() * 2 * math.pi if Settings.RANDOM_PLAYER_INIT else math.pi
                self._setup_player(idx, pos[0], pos[1], a, False, idx - 1, m, mom, half)
                idx += 1

        if phantom_player is not None:
            a = random() * 2 * math.pi if Settings.RANDOM_PLAYER_INIT else math.pi
            self._setup_player(idx, phantom_player["position_x"], phantom_player["position_y"],
                               a, False, idx - 1, m, mom, half)
            idx += 1

        self._n_bodies = idx
        self._n_players = idx - 1

        return 0 if human else None

    def _setup_player(self, idx, x, y, angle, left_team, player_id, m, mom, half):
        self._pos_x[idx] = x
        self._pos_y[idx] = y
        self._prev_pos_x[idx] = x
        self._prev_pos_y[idx] = y
        self._angle[idx] = angle
        self._prev_angle[idx] = angle
        self._shape_type[idx] = 1
        self._mass[idx] = m
        self._inv_mass[idx] = 1.0 / m
        self._moment[idx] = mom
        self._inv_moment[idx] = 1.0 / mom
        self._elasticity[idx] = Settings.PLAYER_ELASTICITY
        self._friction[idx] = Settings.PLAYER_FRICTION
        self._half_w[idx] = half
        self._half_h[idx] = half
        self._left_team[idx] = 1 if left_team else 0
        self._can_shoot[idx] = 0
        self._had_ball[idx] = 0
        color = Settings.PLAYER_LEFT_COLOR if left_team else Settings.PLAYER_RIGHT_COLOR
        self._colors[idx] = color

    # --- Hot Path ---

    def step(self, dt: float, n_substeps: int) -> None:
        numba_core.step_kernel(
            self._pos_x, self._pos_y, self._vel_x, self._vel_y,
            self._angle, self._ang_vel,
            self._mass, self._inv_mass, self._moment, self._inv_moment,
            self._elasticity, self._friction,
            self._shape_type, self._radius, self._half_w, self._half_h,
            self._n_bodies,
            self._wall_ax, self._wall_ay, self._wall_bx, self._wall_by,
            self._wall_radius, self._wall_elasticity, self._wall_friction,
            self._n_walls,
            dt, n_substeps, Settings.GROUND_FRICTION
        )

    # --- State Access ---

    def get_ball_state(self) -> BallState:
        return BallState(
            entity_id=-1,
            position=(self._pos_x[0], self._pos_y[0]),
            velocity=(self._vel_x[0], self._vel_y[0]),
            angle=self._angle[0],
            previous_position=(self._prev_pos_x[0], self._prev_pos_y[0]),
            last_player_shoot=None if self._last_player_shoot == -2 else self._last_player_shoot,
            radius=Settings.BALL_RADIUS,
        )

    def get_player_state(self, player_id: int) -> PlayerState:
        i = player_id + 1
        return PlayerState(
            entity_id=player_id,
            position=(self._pos_x[i], self._pos_y[i]),
            velocity=(self._vel_x[i], self._vel_y[i]),
            angle=self._angle[i],
            angular_velocity=self._ang_vel[i],
            left_team=bool(self._left_team[i]),
            can_shoot=bool(self._can_shoot[i]),
            had_ball=bool(self._had_ball[i]),
            previous_position=(self._prev_pos_x[i], self._prev_pos_y[i]),
            previous_angle=self._prev_angle[i],
            color=tuple(self._colors[i]),
        )

    def get_all_player_states(self) -> list[PlayerState]:
        return [self.get_player_state(pid) for pid in range(self._n_players)]

    def get_n_players(self) -> int:
        return self._n_players

    # --- State Mutation ---

    def set_velocity(self, entity_id: int, vx: float, vy: float) -> None:
        i = self._entity_to_index(entity_id)
        self._vel_x[i] = vx
        self._vel_y[i] = vy

    def set_angular_velocity(self, entity_id: int, omega: float) -> None:
        i = self._entity_to_index(entity_id)
        self._ang_vel[i] = omega

    def set_angle(self, entity_id: int, angle: float) -> None:
        i = self._entity_to_index(entity_id)
        self._angle[i] = angle

    def set_position(self, entity_id: int, x: float, y: float) -> None:
        i = self._entity_to_index(entity_id)
        self._pos_x[i] = x
        self._pos_y[i] = y

    # --- Bookkeeping ---

    def snapshot_previous_positions(self) -> None:
        n = self._n_bodies
        self._prev_pos_x[:n] = self._pos_x[:n]
        self._prev_pos_y[:n] = self._pos_y[:n]
        self._prev_angle[:n] = self._angle[:n]

    def update_previous_position(self, entity_id: int) -> None:
        i = self._entity_to_index(entity_id)
        self._prev_pos_x[i] = self._pos_x[i]
        self._prev_pos_y[i] = self._pos_y[i]

    def update_can_shoot(self, player_id: int, value: bool) -> None:
        self._can_shoot[player_id + 1] = int(value)

    def update_had_ball(self, player_id: int, value: bool) -> None:
        self._had_ball[player_id + 1] = int(value)

    def set_last_player_shoot(self, value: Optional[int]) -> None:
        self._last_player_shoot = -2 if value is None else value

    # --- Direct field access (no dataclass allocation) ---

    def get_position(self, entity_id: int) -> tuple[float, float]:
        i = self._entity_to_index(entity_id)
        return (self._pos_x[i], self._pos_y[i])

    def get_velocity(self, entity_id: int) -> tuple[float, float]:
        i = self._entity_to_index(entity_id)
        return (self._vel_x[i], self._vel_y[i])

    def get_angle(self, entity_id: int) -> float:
        i = self._entity_to_index(entity_id)
        return float(self._angle[i])

    def get_left_team(self, player_id: int) -> bool:
        return bool(self._left_team[player_id + 1])

    def get_can_shoot(self, player_id: int) -> bool:
        return bool(self._can_shoot[player_id + 1])

    def get_had_ball(self, player_id: int) -> bool:
        return bool(self._had_ball[player_id + 1])

    def get_previous_position(self, entity_id: int) -> tuple[float, float]:
        i = self._entity_to_index(entity_id)
        return (self._prev_pos_x[i], self._prev_pos_y[i])

    def get_last_player_shoot(self) -> Optional[int]:
        return None if self._last_player_shoot == -2 else self._last_player_shoot

    def get_ball_radius(self) -> float:
        return float(self._radius[0])

    # --- Full game step (mega-kernel) ---

    def set_reward_coeffs(self, coeff_dict: dict) -> None:
        """Pre-extract reward coefficients from dict into flat array for Numba."""
        mapping = [
            "static_reward", "starting_static_reward", "ending_static_reward",
            "delta_ball_player_coeff", "delta_ball_goal_coeff", "can_shoot_coeff",
            "goal_coeff", "wrong_goal_coeff", "has_ball_coeff",
            "static_lead_reward", "static_draw_reward",
        ]
        for i, key in enumerate(mapping):
            val = coeff_dict.get(key, None)
            self._reward_coeffs[i] = float('nan') if val is None else float(val)

    def set_score(self, score):
        """Set the score array reference."""
        self._score = score

    def full_step(self, actions, mean_steps=2000.0):
        """
        Complete game step in one Numba call.
        actions: list/array of int actions per player.
        Returns: (visions, global_vision, rewards, done)
        """
        n_p = self._n_players
        for i in range(n_p):
            self._actions_buf[i] = int(actions[i])

        self._prev_score_buf[0] = self._score[0]
        self._prev_score_buf[1] = self._score[1]

        # Zero output buffers
        self._vision_buf[:] = 0.0
        self._global_vision_buf[:] = 0.0
        self._rewards_buf[:] = 0.0

        done, lps = numba_env.full_game_step(
            self._pos_x, self._pos_y, self._vel_x, self._vel_y,
            self._angle, self._ang_vel,
            self._prev_pos_x, self._prev_pos_y, self._prev_angle,
            self._mass, self._inv_mass, self._moment, self._inv_moment,
            self._elasticity, self._friction, self._shape_type,
            self._radius, self._half_w, self._half_h,
            self._can_shoot, self._had_ball, self._left_team,
            self._n_bodies, self._n_players,
            self._wall_ax, self._wall_ay, self._wall_bx, self._wall_by,
            self._wall_radius, self._wall_elasticity, self._wall_friction,
            self._n_walls,
            self._actions_buf,
            self._score, self._prev_score_buf,
            self._last_player_shoot,
            self._left_goal_x, self._left_goal_y,
            self._right_goal_x, self._right_goal_y,
            self._reward_coeffs,
            float(Settings.DELTA_TIME), int(Settings.DELTA_SIMU), float(Settings.GROUND_FRICTION),
            float(Settings.PLAYER_SPEED), float(Settings.PLAYER_ROT_SPEED), float(Settings.SHOOTING_SPEED),
            float(Settings.PLAYER_LEN), float(Settings.BALL_RADIUS), float(Settings.GOAL_LEN),
            float(Settings.DIM_X), float(Settings.DIM_Y), float(Settings.SCREEN_OFFSET),
            int(Settings.PLAYERS_NUMBER[0]), int(Settings.PLAYERS_NUMBER[1]),
            float(mean_steps),
            bool(Settings.COMPETITIVE_VISION), int(Settings.ENTRY_NEURONS),
            self._vision_buf, self._global_vision_buf, self._rewards_buf,
        )

        self._last_player_shoot = lps
        return self._vision_buf, self._global_vision_buf, self._rewards_buf, done

    # --- Raycasting ---

    def raycast(self, origin: tuple, end: tuple, radius: float,
                exclude_entity_id: Optional[int] = None) -> Optional[dict]:
        return None  # NUMBER_OF_RAYS=0, never called during training

    # --- Rendering ---

    def get_render_data(self) -> dict:
        players = []
        for pid in range(self._n_players):
            i = pid + 1
            px, py = self._pos_x[i], self._pos_y[i]
            a = self._angle[i]
            hw, hh = self._half_w[i], self._half_h[i]
            ca, sa = math.cos(a), math.sin(a)
            local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            verts = [(px + lx * ca - ly * sa, py + lx * sa + ly * ca) for lx, ly in local]
            players.append({
                "position": (px, py),
                "angle": a,
                "color": tuple(self._colors[i]),
                "vertices": verts,
            })

        ball = {
            "position": (self._pos_x[0], self._pos_y[0]),
            "radius": Settings.BALL_RADIUS,
            "color": Settings.BALL_COLOR,
            "angle": self._angle[0],
        }

        return {
            "players": players,
            "ball": ball,
            "walls": self._wall_segments_render,
        }
