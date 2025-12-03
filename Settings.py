# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:30:06 2025

@author: quent
"""

import numpy as np

# Jeu :
DELTA_TIME: int = 100 # ms, should be a divider of 1000
MAX_FPS: int = 60 # Limits fps to a certain amount => less laggy
SCREEN_OFFSET: int = 75 # px
BACKGROUND_COLOR: tuple[int, int, int] = (30, 30, 30)
SCORE_COLOR: tuple[int, int, int] = (255, 215, 0) # text color for the score

# Terrain :
<<<<<<< HEAD
DIM_X: int = 750 # px
DIM_Y: int = 500 # px
GOAL_LEN: int = 498 # px
GROUND_FRICTION: float = 0.5
LEFT_GOAL_COLLISION_TYPE: int = 10 # Necessary to differenciate the left and right goals
RIGHT_GOAL_COLLISION_TYPE: int = 11 # Necessary to differenciate the left and right goals
MAX_DIST: float = np.sqrt(DIM_X*DIM_Y)
=======
DIM_X = 750 # px
DIM_Y = 500 # px
GOAL_LEN = 250 # px
GROUND_FRICTION = 0.5
LEFT_GOAL_COLLISION_TYPE = 10 # Necessary to differenciate the left and right goals
RIGHT_GOAL_COLLISION_TYPE = 11 # Necessary to differenciate the left and right goals
MAX_DIST = np.sqrt(DIM_X*DIM_Y)
>>>>>>> main

# Murs :
WALL_FRICTION: float = 0.9
WALL_ELASTICITY: float = 0.95

# Balle (cercle) : 
BALL_RADIUS: int = 15 # px
BALL_MASS: float = 0.2 # kg, useful for collisions with players
BALL_FRICTION: float = 0.2
BALL_ELASTICITY: float = 0.8
BALL_COLOR: tuple[int, int, int, int] = (255, 255, 255, 255)

RANDOM_BALL_POSITION: bool = True

# Player (square) : 
PLAYER_LEN: int = 30 # px
PLAYER_MASS: int = 1 # kg, useful for collisions with the ball
PLAYER_FRICTION: float = 0.0
PLAYER_ELASTICITY: float = 0.8
PLAYER_LEFT_COLOR: tuple[int, int, int, int] = (30, 144, 255, 255) # Keep the fourth one, it's necessary for pymunk (opacity)
PLAYER_RIGHT_COLOR: tuple[int, int, int, int] = (255, 69, 0, 255) # Keep the fourth one, it's necessary for pymunk (opacity)
PLAYER_ARROW_COLOR: tuple[int, int, int, int] = (0, 255, 0, 255) # Keep the fourth one, it's necessary for pymunk (opacity)
SHOOTING_SPEED: int = 500 # px/s
PLAYER_SHOOTING_RANGE: int = 10 # px
PLAYER_SPEED: int = 100 # px/s
PLAYER_ROT_SPEED: float = np.pi*0.75 # rad/s

# AI :
<<<<<<< HEAD
NUMBER_OF_RAYS: int = 0
RAY_ANGLE: float = 2*np.pi # rad, total angle to cover with rayTracing
VISION_RANGE: float = MAX_DIST # Limits the range of rays for rayTracing. "inf" by default here.
ENTRY_NEURONS: int = NUMBER_OF_RAYS*8 + 8
=======
NUMBER_OF_RAYS = 0
RAY_ANGLE = 2*np.pi # rad, total angle to cover with rayTracing
VISION_RANGE = MAX_DIST # Limits the range of rays for rayTracing. "inf" by default here.
ENTRY_NEURONS = NUMBER_OF_RAYS*8 + 9
>>>>>>> main


































