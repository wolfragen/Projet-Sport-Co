# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:06:33 2025

@author: quent
"""

import pymunk

from Settings import Settings


def buildBoard(space):
    static_body = space.static_body

    # Field parameters
    offset = Settings.SCREEN_OFFSET
    width = Settings.DIM_X
    height = Settings.DIM_Y
    goal_len = Settings.GOAL_LEN
    wall_radius = Settings.WALL_RADIUS
    player_len = Settings.PLAYER_LEN

    # Top and bottom lines
    top_line = (offset+player_len, offset), (offset + width-player_len, offset)
    bottom_line = (offset+player_len, offset + height), (offset + width-player_len, offset + height)

    # Vertical sides with goal openings
    left_top_goal = offset + (height - goal_len)/2
    left_bottom_goal = offset + (height + goal_len)/2
    right_top_goal = offset + (height - goal_len)/2
    right_bottom_goal = offset + (height + goal_len)/2

    # Static walls
    static_lines = [
        # Top and bottom
        pymunk.Segment(static_body, top_line[0], top_line[1], wall_radius),
        pymunk.Segment(static_body, bottom_line[0], bottom_line[1], wall_radius),

        # Left side with goal opening
        pymunk.Segment(static_body, (offset, offset+player_len), (offset, left_top_goal), wall_radius),
        pymunk.Segment(static_body, (offset, left_bottom_goal), (offset, offset + height-player_len), wall_radius),

        # Right side with goal opening
        pymunk.Segment(static_body, (offset + width, offset+player_len), (offset + width, right_top_goal), wall_radius),
        pymunk.Segment(static_body, (offset + width, right_bottom_goal), (offset + width, offset + height-player_len), wall_radius),
        
        # Build small segments to facilitate ball-handling around the corners
        pymunk.Segment(static_body, (offset + player_len, offset), (offset, offset + player_len), wall_radius),
        pymunk.Segment(static_body, (offset + width - player_len, offset), (offset + width, offset + player_len), wall_radius),
        pymunk.Segment(static_body, (offset + player_len, offset + height), (offset, offset + height - player_len), wall_radius),
        pymunk.Segment(static_body, (offset + width - player_len, offset + height), (offset + width, offset + height - player_len), wall_radius),
    ]

    # Set elasticity and friction for walls
    for line in static_lines:
        line.elasticity = Settings.WALL_ELASTICITY
        line.friction = Settings.WALL_FRICTION

    space.add(*static_lines)

    # === Invisible goal sensors for AI vision ===
    left_goal_sensor = pymunk.Segment(
        static_body,
        (offset, left_top_goal),
        (offset, left_bottom_goal),
        1.0
    )
    right_goal_sensor = pymunk.Segment(
        static_body,
        (offset + width, right_top_goal),
        (offset + width, right_bottom_goal),
        1.0
    )

    # Sensors do not collide
    left_goal_sensor.sensor = True
    right_goal_sensor.sensor = True

    # Assign collision types for AI recognition
    left_goal_sensor.collision_type = Settings.LEFT_GOAL_COLLISION_TYPE
    right_goal_sensor.collision_type = Settings.RIGHT_GOAL_COLLISION_TYPE

    # Make invisible for debug draw
    left_goal_sensor.color = (0, 0, 0, 0)
    right_goal_sensor.color = (0, 0, 0, 0)

    space.add(left_goal_sensor, right_goal_sensor)

    # Save goal center positions for reference
    left_goal_position = (offset, (left_bottom_goal + left_top_goal)/2)
    right_goal_position = (offset + width, (right_bottom_goal + right_top_goal)/2)

    return left_goal_position, right_goal_position













