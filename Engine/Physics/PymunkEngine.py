# -*- coding: utf-8 -*-

import math
from typing import Optional
from random import randint, random

import numpy as np
import pymunk

from Settings import Settings
from Engine.Physics.PhysicsEngine import PhysicsEngine
from Engine.Physics.EntityState import PlayerState, BallState


class PymunkEngine(PhysicsEngine):
    """Pymunk-based physics backend."""

    def __init__(self):
        self.space: Optional[pymunk.Space] = None
        self.ball_body: Optional[pymunk.Body] = None
        self.ball_shape: Optional[pymunk.Circle] = None
        self.player_bodies: list[pymunk.Body] = []
        self.player_shapes: list[pymunk.Shape] = []
        self.player_left_team: list[bool] = []
        self.player_colors: list[tuple] = []
        self.wall_segments: list[tuple] = []  # list of ((x1,y1), (x2,y2), radius)

    # --- Lifecycle ---

    def create_world(self) -> None:
        self.space = pymunk.Space()
        self.space.damping = Settings.GROUND_FRICTION
        self.ball_body = None
        self.ball_shape = None
        self.player_bodies = []
        self.player_shapes = []
        self.player_left_team = []
        self.player_colors = []
        self.wall_segments = []

    def build_board(self) -> tuple[tuple, tuple]:
        static_body = self.space.static_body

        offset = Settings.SCREEN_OFFSET
        width = Settings.DIM_X
        height = Settings.DIM_Y
        goal_len = Settings.GOAL_LEN
        wall_radius = Settings.WALL_RADIUS
        player_len = Settings.PLAYER_LEN
        round_corner = Settings.ROUND_CORNER

        if round_corner:
            top_line = (offset + player_len, offset), (offset + width - player_len, offset)
            bottom_line = (offset + player_len, offset + height), (offset + width - player_len, offset + height)
        else:
            top_line = (offset, offset), (offset + width, offset)
            bottom_line = (offset, offset + height), (offset + width, offset + height)

        left_top_goal = offset + (height - goal_len) / 2
        left_bottom_goal = offset + (height + goal_len) / 2
        right_top_goal = offset + (height - goal_len) / 2
        right_bottom_goal = offset + (height + goal_len) / 2

        if round_corner:
            static_lines = [
                pymunk.Segment(static_body, top_line[0], top_line[1], wall_radius),
                pymunk.Segment(static_body, bottom_line[0], bottom_line[1], wall_radius),
                pymunk.Segment(static_body, (offset, offset + player_len), (offset, left_top_goal), wall_radius),
                pymunk.Segment(static_body, (offset, left_bottom_goal), (offset, offset + height - player_len), wall_radius),
                pymunk.Segment(static_body, (offset + width, offset + player_len), (offset + width, right_top_goal), wall_radius),
                pymunk.Segment(static_body, (offset + width, right_bottom_goal), (offset + width, offset + height - player_len), wall_radius),
                pymunk.Segment(static_body, (offset + player_len, offset), (offset, offset + player_len), wall_radius),
                pymunk.Segment(static_body, (offset + width - player_len, offset), (offset + width, offset + player_len), wall_radius),
                pymunk.Segment(static_body, (offset + player_len, offset + height), (offset, offset + height - player_len), wall_radius),
                pymunk.Segment(static_body, (offset + width - player_len, offset + height), (offset + width, offset + height - player_len), wall_radius),
            ]
        else:
            static_lines = [
                pymunk.Segment(static_body, top_line[0], top_line[1], wall_radius),
                pymunk.Segment(static_body, bottom_line[0], bottom_line[1], wall_radius),
                pymunk.Segment(static_body, (offset, offset), (offset, left_top_goal), wall_radius),
                pymunk.Segment(static_body, (offset, left_bottom_goal), (offset, offset + height), wall_radius),
                pymunk.Segment(static_body, (offset + width, offset), (offset + width, right_top_goal), wall_radius),
                pymunk.Segment(static_body, (offset + width, right_bottom_goal), (offset + width, offset + height), wall_radius),
            ]

        for line in static_lines:
            line.elasticity = Settings.WALL_ELASTICITY
            line.friction = Settings.WALL_FRICTION

        self.space.add(*static_lines)

        # Store wall segments for rendering
        for seg in static_lines:
            self.wall_segments.append((tuple(seg.a), tuple(seg.b), seg.radius))

        # Goal sensors (invisible, for AI vision raycasting)
        left_goal_sensor = pymunk.Segment(static_body, (offset, left_top_goal), (offset, left_bottom_goal), 1.0)
        right_goal_sensor = pymunk.Segment(static_body, (offset + width, right_top_goal), (offset + width, right_bottom_goal), 1.0)

        left_goal_sensor.sensor = True
        right_goal_sensor.sensor = True
        left_goal_sensor.collision_type = Settings.LEFT_GOAL_COLLISION_TYPE
        right_goal_sensor.collision_type = Settings.RIGHT_GOAL_COLLISION_TYPE
        left_goal_sensor.color = (0, 0, 0, 0)
        right_goal_sensor.color = (0, 0, 0, 0)

        self.space.add(left_goal_sensor, right_goal_sensor)

        left_goal_position = (offset, (left_bottom_goal + left_top_goal) / 2)
        right_goal_position = (offset + width, (right_bottom_goal + right_top_goal) / 2)

        return left_goal_position, right_goal_position

    def build_ball(self) -> None:
        radius = Settings.BALL_RADIUS
        mass = Settings.BALL_MASS
        dim_x = Settings.DIM_X
        dim_y = Settings.DIM_Y
        offset = Settings.SCREEN_OFFSET

        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)

        if Settings.RANDOM_BALL_POSITION:
            body.position = (
                randint(round(offset + dim_x * 1 / 10), round(offset + dim_x * 9 / 10)),
                randint(round(offset + dim_y * 1 / 10), round(offset + dim_y * 9 / 10))
            )
        else:
            body.position = (offset + dim_x / 2, offset + dim_y / 2)

        body.previous_position = body.position
        body.last_player_shoot = None

        shape = pymunk.Circle(body, radius)
        shape.elasticity = Settings.BALL_ELASTICITY
        shape.friction = Settings.BALL_FRICTION
        shape.color = Settings.BALL_COLOR
        shape.is_ball = True

        self.space.add(body, shape)
        self.ball_body = body
        self.ball_shape = shape

    def build_players(self, players_number: tuple[int, int], human: bool = False,
                      phantom_player=None) -> Optional[int]:
        from Engine.Entity.Player import spacing

        offset = Settings.SCREEN_OFFSET
        dim_x = Settings.DIM_X
        dim_y = Settings.DIM_Y
        size = Settings.PLAYER_LEN
        mass = Settings.PLAYER_MASS

        self.player_bodies = []
        self.player_shapes = []
        self.player_left_team = []
        self.player_colors = []

        n_left, n_right = players_number
        nb_players_done = 0

        left_positions = spacing(n_left, size, offset, offset, dim_x / 2, dim_y)
        for i in range(n_left):
            pos = left_positions[i].tolist()
            self._create_player(pos, angle=0, left_team=True, player_id=nb_players_done)
            nb_players_done += 1

        if n_right != 0:
            right_positions = spacing(n_right, size, dim_x / 2 + offset, offset, dim_x / 2, dim_y, revert_x=True)
            for i in range(n_right):
                pos = right_positions[-i].tolist()
                self._create_player(pos, angle=np.pi, left_team=False, player_id=nb_players_done)
                nb_players_done += 1

        if phantom_player is not None:
            pos = [phantom_player["position_x"], phantom_player["position_y"]]
            self._create_player(pos, angle=np.pi, left_team=False, player_id=nb_players_done)
            nb_players_done += 1

        # Add all to space
        for body, shape in zip(self.player_bodies, self.player_shapes):
            self.space.add(body, shape)

        selected_player = 0 if human else None
        return selected_player

    def _create_player(self, pos, angle, left_team, player_id):
        size = Settings.PLAYER_LEN
        mass = Settings.PLAYER_MASS
        color = Settings.PLAYER_LEFT_COLOR if left_team else Settings.PLAYER_RIGHT_COLOR

        moment = pymunk.moment_for_box(mass, (size, size))
        body = pymunk.Body(mass, moment)
        body.position = pos
        body.previous_position = tuple(pos)
        body.angle = angle
        body.canShoot = False
        body.hadBall = False
        body.player_id = player_id
        body.previous_angle = angle

        if Settings.RANDOM_PLAYER_INIT:
            body.angle = random() * 2 * np.pi

        shape = pymunk.Poly.create_box(body, (size, size))
        shape.elasticity = Settings.PLAYER_ELASTICITY
        shape.friction = Settings.PLAYER_FRICTION
        shape.color = color
        shape.is_player = True
        shape.left_team = left_team

        self.player_bodies.append(body)
        self.player_shapes.append(shape)
        self.player_left_team.append(left_team)
        self.player_colors.append(color)

    def step(self, dt: float, n_substeps: int) -> None:
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
            self.space.step(sub_dt / 1000)

    # --- State Access ---

    def get_ball_state(self) -> BallState:
        b = self.ball_body
        return BallState(
            entity_id=-1,
            position=tuple(b.position),
            velocity=tuple(b.velocity),
            angle=b.angle,
            previous_position=tuple(b.previous_position),
            last_player_shoot=b.last_player_shoot,
            radius=Settings.BALL_RADIUS,
        )

    def get_player_state(self, player_id: int) -> PlayerState:
        body = self.player_bodies[player_id]
        shape = self.player_shapes[player_id]
        return PlayerState(
            entity_id=player_id,
            position=tuple(body.position),
            velocity=tuple(body.velocity),
            angle=body.angle,
            angular_velocity=body.angular_velocity,
            left_team=shape.left_team,
            can_shoot=body.canShoot,
            had_ball=body.hadBall,
            previous_position=tuple(body.previous_position),
            previous_angle=body.previous_angle,
            color=shape.color,
        )

    def get_all_player_states(self) -> list[PlayerState]:
        return [self.get_player_state(i) for i in range(len(self.player_bodies))]

    def get_n_players(self) -> int:
        return len(self.player_bodies)

    # --- State Mutation ---

    def set_velocity(self, entity_id: int, vx: float, vy: float) -> None:
        body = self._get_body(entity_id)
        body.velocity = (vx, vy)

    def set_angular_velocity(self, entity_id: int, omega: float) -> None:
        body = self._get_body(entity_id)
        body.angular_velocity = omega

    def set_angle(self, entity_id: int, angle: float) -> None:
        body = self._get_body(entity_id)
        body.angle = angle

    def set_position(self, entity_id: int, x: float, y: float) -> None:
        body = self._get_body(entity_id)
        body.position = (x, y)

    def _get_body(self, entity_id: int) -> pymunk.Body:
        if entity_id == -1:
            return self.ball_body
        return self.player_bodies[entity_id]

    # --- Bookkeeping ---

    def snapshot_previous_positions(self) -> None:
        for body in self.player_bodies:
            body.previous_position = tuple(body.position)
            body.previous_angle = body.angle
        self.ball_body.previous_position = tuple(self.ball_body.position)

    def update_previous_position(self, entity_id: int) -> None:
        body = self._get_body(entity_id)
        body.previous_position = tuple(body.position)

    def update_can_shoot(self, player_id: int, value: bool) -> None:
        self.player_bodies[player_id].canShoot = value

    def update_had_ball(self, player_id: int, value: bool) -> None:
        self.player_bodies[player_id].hadBall = value

    def set_last_player_shoot(self, value: Optional[int]) -> None:
        self.ball_body.last_player_shoot = value

    # --- Direct field access (no dataclass allocation) ---

    def get_position(self, entity_id: int) -> tuple[float, float]:
        body = self._get_body(entity_id)
        return tuple(body.position)

    def get_velocity(self, entity_id: int) -> tuple[float, float]:
        body = self._get_body(entity_id)
        return tuple(body.velocity)

    def get_angle(self, entity_id: int) -> float:
        body = self._get_body(entity_id)
        return float(body.angle)

    def get_left_team(self, player_id: int) -> bool:
        return bool(self.player_shapes[player_id].left_team)

    def get_can_shoot(self, player_id: int) -> bool:
        return bool(self.player_bodies[player_id].canShoot)

    def get_had_ball(self, player_id: int) -> bool:
        return bool(self.player_bodies[player_id].hadBall)

    def get_previous_position(self, entity_id: int) -> tuple[float, float]:
        body = self._get_body(entity_id)
        return tuple(body.previous_position)

    def get_last_player_shoot(self) -> Optional[int]:
        return self.ball_body.last_player_shoot

    def get_ball_radius(self) -> float:
        return float(Settings.BALL_RADIUS)

    # --- Raycasting ---

    def raycast(self, origin: tuple, end: tuple, radius: float,
                exclude_entity_id: Optional[int] = None) -> Optional[dict]:
        # Temporarily mask excluded entity
        original_filter = None
        if exclude_entity_id is not None:
            if exclude_entity_id == -1:
                excluded_shape = self.ball_shape
            else:
                excluded_shape = self.player_shapes[exclude_entity_id]
            original_filter = excluded_shape.filter
            excluded_shape.filter = pymunk.ShapeFilter(mask=0)

        hit = self.space.segment_query_first(origin, end, radius, pymunk.ShapeFilter())

        # Restore filter
        if original_filter is not None:
            excluded_shape.filter = original_filter

        if hit is None:
            return None

        hit_shape = hit.shape
        point = tuple(hit.point)

        if hasattr(hit_shape, "is_ball") and hit_shape.is_ball:
            entity_type = 4
        elif hasattr(hit_shape, "is_player") and hit_shape.is_player:
            entity_type = 5 if hit_shape.left_team else 6
        elif hit_shape.collision_type == Settings.LEFT_GOAL_COLLISION_TYPE:
            entity_type = 2
        elif hit_shape.collision_type == Settings.RIGHT_GOAL_COLLISION_TYPE:
            entity_type = 3
        else:
            entity_type = 1  # wall

        return {"point": point, "entity_type": entity_type}

    # --- Rendering ---

    def get_render_data(self) -> dict:
        players = []
        for i in range(len(self.player_bodies)):
            body = self.player_bodies[i]
            shape = self.player_shapes[i]
            vertices = [tuple(p.rotated(body.angle) + body.position) for p in shape.get_vertices()]
            players.append({
                "position": tuple(body.position),
                "angle": body.angle,
                "color": shape.color,
                "vertices": vertices,
            })

        ball_data = None
        if self.ball_body is not None:
            ball_data = {
                "position": tuple(self.ball_body.position),
                "radius": Settings.BALL_RADIUS,
                "color": Settings.BALL_COLOR,
                "angle": self.ball_body.angle,
            }

        return {
            "players": players,
            "ball": ball_data,
            "walls": self.wall_segments,
        }
