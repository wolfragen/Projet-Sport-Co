# -*- coding: utf-8 -*-
"""
Simple heuristic policies to bootstrap learning.
"""

import math


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def chase_and_shoot(player, ball, rotate_thresh: float = math.radians(10)) -> int:
    """
    Heuristic: rotate towards the ball, move forward, shoot if possible.
    Actions:
      0: forward, 1: rotate left, 2: rotate right, 3: shoot
    """
    body, _ = player
    ball_body, _ = ball

    if getattr(body, "canShoot", False):
        return 3

    vec_x = ball_body.position[0] - body.position[0]
    vec_y = ball_body.position[1] - body.position[1]
    angle_to_ball = math.atan2(vec_y, vec_x)
    angle_diff = normalize_angle(angle_to_ball - body.angle)

    if abs(angle_diff) > rotate_thresh:
        return 1 if angle_diff < 0 else 2

    return 0
