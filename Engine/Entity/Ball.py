# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:11:32 2025

@author: quent
"""

import pymunk
from random import randint

import Settings


def buildBall(space):
    # Ball parameters
    radius = Settings.BALL_RADIUS
    mass = Settings.BALL_MASS
    dim_x = Settings.DIM_X
    dim_y = Settings.DIM_Y
    offset = Settings.SCREEN_OFFSET

    # Calculate moment of inertia
    moment = pymunk.moment_for_circle(mass, 0, radius)

    # Dynamic body for the ball
    body = pymunk.Body(mass, moment)
    if(Settings.RANDOM_BALL_POSITION):
        body.position = (
            randint(round(offset +dim_x *1/10), round(offset +dim_x *9/10)),
            randint(round(offset +dim_y *1/10), round(offset +dim_y *9/10))
        )  # random position
    else:
        body.position = (
            offset +dim_x / 2,
            offset + dim_y / 2
        )  # center of the field
    body.previous_position = body.position

    # Circle shape for the ball
    shape = pymunk.Circle(body, radius)
    shape.elasticity = Settings.BALL_ELASTICITY
    shape.friction = Settings.BALL_FRICTION
    shape.color = Settings.BALL_COLOR

    # Tag for ray tracing or identification
    shape.is_ball = True

    # Add to space and update game dictionary
    space.add(body, shape)
    ball = (body, shape)
    
    return ball