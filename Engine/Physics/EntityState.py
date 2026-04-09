# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayerState:
    entity_id: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    angle: float
    angular_velocity: float
    left_team: bool
    can_shoot: bool
    had_ball: bool
    previous_position: tuple[float, float]
    previous_angle: float
    color: tuple


@dataclass
class BallState:
    entity_id: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    angle: float
    previous_position: tuple[float, float]
    last_player_shoot: Optional[int]
    radius: float
