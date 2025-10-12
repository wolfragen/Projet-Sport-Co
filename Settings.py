# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:30:06 2025

@author: quent
"""

import numpy as np

# Jeu :
DELTA_TIME = 10 # ms, doit donner un fps entier...
MAX_FPS = 60 # limite l'actualisation graphique, n'impacte pas la simulation
SCREEN_OFFSET = 75 # px
BACKGROUND_COLOR = ()
SCORE_COLOR = (255,0,0)

# Terrain :
DIM_X = 750 # px
DIM_Y = 500 # px
GOAL_LEN = 200 # px
GROUND_FRICTION = 0.5
LEFT_GOAL_COLLISION_TYPE = 10
RIGHT_GOAL_COLLISION_TYPE = 11

# Murs :
WALL_FRICTION = 0.9
WALL_ELASTICITY = 0.95

# Balle (cercle) : 
BALL_RADIUS = 15 # m
BALL_MASS = 0.2
BALL_FRICTION = 0.2
BALL_ELASTICITY = 0.8

# Player (square) : 
PLAYER_LEN = 30 # m
PLAYER_MASS = 1
PLAYER_FRICTION = 0.0
PLAYER_ELASTICITY = 0.8
PLAYER_LEFT_COLOR = (0,0,255,255) # Garder le 4ème, c'est l'opacité pour pymunk
PLAYER_RIGHT_COLOR = (255,0,0,255)
PLAYER_ARROW_COLOR = (0,255,0,255)
SHOOTING_SPEED = 500 # px/s
PLAYER_SHOOTING_RANGE = 10 # px
PLAYER_SPEED = 100 # px/s
PLAYER_ROT_SPEED = np.pi*0.75 # rad/s

# AI :
NUMBER_OF_RAYS = 64
RAY_ANGLE = 2*np.pi #rad
VISION_RANGE = DIM_X+SCREEN_OFFSET
