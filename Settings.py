# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:30:06 2025

@author: quent
"""

import numpy as np

# Jeu :
DELTA_TIME = 100 # ms, should be a divider of 1000
MAX_FPS = 60 # Limits fps to a certain amount => less laggy
SCREEN_OFFSET = 75 # px
BACKGROUND_COLOR = (30, 30, 30)
SCORE_COLOR = (255, 215, 0) # text color for the score

# Terrain :
DIM_X = 750 # px
DIM_Y = 500 # px
GOAL_LEN = 250 # px
GROUND_FRICTION = 0.5
LEFT_GOAL_COLLISION_TYPE = 10 # Necessary to differenciate the left and right goals
RIGHT_GOAL_COLLISION_TYPE = 11 # Necessary to differenciate the left and right goals
MAX_DIST = np.sqrt(DIM_X*DIM_Y)

# Murs :
WALL_FRICTION = 0.9
WALL_ELASTICITY = 0.95

# Balle (cercle) : 
BALL_RADIUS = 15 # px
BALL_MASS = 0.2 # kg, useful for collisions with players
BALL_FRICTION = 0.2
BALL_ELASTICITY = 0.8
BALL_COLOR = (255, 255, 255, 255)

RANDOM_BALL_POSITION = True

# Player (square) : 
PLAYER_LEN = 30 # px
PLAYER_MASS = 1 # kg, useful for collisions with the ball
PLAYER_FRICTION = 0.0
PLAYER_ELASTICITY = 0.8
PLAYER_LEFT_COLOR = (30, 144, 255, 255) # Keep the fourth one, it's necessary for pymunk (opacity)
PLAYER_RIGHT_COLOR = (255, 69, 0, 255) # Keep the fourth one, it's necessary for pymunk (opacity)
PLAYER_ARROW_COLOR = (0,255,0,255) # Keep the fourth one, it's necessary for pymunk (opacity)
SHOOTING_SPEED = 500 # px/s
PLAYER_SHOOTING_RANGE = 10 # px
PLAYER_SPEED = 100 # px/s
PLAYER_ROT_SPEED = np.pi*0.75 # rad/s

# AI :
NUMBER_OF_RAYS = 0
RAY_ANGLE = 2*np.pi # rad, total angle to cover with rayTracing
VISION_RANGE = MAX_DIST # Limits the range of rays for rayTracing. "inf" by default here.
ENTRY_NEURONS = NUMBER_OF_RAYS*8 + 8


































