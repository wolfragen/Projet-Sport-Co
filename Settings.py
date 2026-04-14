# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:30:06 2025

@author: quent
"""

import numpy as np
import os
from pathlib import Path

class _Settings:
    pass


Settings = _Settings()

# Jeu :
Settings.DELTA_TIME = 100 # ms, should be a divider of 1000
Settings.DELTA_SIMU = 7 # nombre de step par delta time
Settings.MAX_FPS = 60 # Limits fps to a certain amount:> less laggy
Settings.SCREEN_OFFSET = 75 # px
Settings.BACKGROUND_COLOR = (30, 30, 30)
Settings.SCORE_COLOR = (255, 215, 0) # text color for the score

# Terrain :
Settings.DIM_X = 750 # px
Settings.DIM_Y = 500 # px
Settings.GOAL_LEN = 250 # px
Settings.ROUND_CORNER = True
Settings.GROUND_FRICTION = 0.5
Settings.LEFT_GOAL_COLLISION_TYPE = 10 # Necessary to differenciate the left and right goals
Settings.RIGHT_GOAL_COLLISION_TYPE = 11 # Necessary to differenciate the left and right goals

# Murs :
Settings.WALL_FRICTION = 0.9
Settings.WALL_ELASTICITY = 0.95
Settings.WALL_RADIUS = 3

# Balle (cercle) : 
Settings.BALL_RADIUS = 15 # px
Settings.BALL_MASS = 0.2 # kg, useful for collisions with players
Settings.BALL_FRICTION = 0.2
Settings.BALL_ELASTICITY = 0.8
Settings.BALL_COLOR = (255, 255, 255, 255)

Settings.RANDOM_BALL_POSITION = True

# Player (square) : 
Settings.PLAYER_LEN = 30 # px
Settings.PLAYER_MASS = 1 # kg, useful for collisions with the ball
Settings.PLAYER_FRICTION = 0.0
Settings.PLAYER_ELASTICITY = 0.8
Settings.PLAYER_LEFT_COLOR = (30, 144, 255, 255) # Keep the fourth one, it's necessary for pymunk (opacity)
Settings.PLAYER_RIGHT_COLOR = (255, 69, 0, 255) # Keep the fourth one, it's necessary for pymunk (opacity)
Settings.PLAYER_ARROW_COLOR = (0,255,0,255) # Keep the fourth one, it's necessary for pymunk (opacity)
Settings.SHOOTING_SPEED = 500 # px/s
Settings.PLAYER_SHOOTING_RANGE = 10 # px
Settings.PLAYER_SPEED = 100 # px/s
Settings.PLAYER_ROT_SPEED = np.pi*0.75 # rad/s

Settings.RANDOM_PLAYER_INIT = True

# AI :
Settings.NUMBER_OF_RAYS = 0
Settings.RAY_ANGLE = 2*np.pi # rad, total angle to cover with rayTracing
Settings.ENTRY_NEURONS = 0
Settings.COMPETITIVE_VISION = True

Settings.IS_KAGGLE = os.path.exists("/kaggle")

Settings.MAX_DIST = np.sqrt(Settings.DIM_X*Settings.DIM_Y)
Settings.VISION_RANGE = Settings.MAX_DIST # Limits the range of rays for rayTracing. "inf" by default here.

Settings.WEIRD_CONTROL = False
Settings.PLAYERS_NUMBER = (1, 1) # default, overridden by training scripts
Settings.SIZE_MODIFIER = 1.0 # global scale for entity sizes (players, ball, shooting range)


def apply_size_modifier():
    """Apply SIZE_MODIFIER to entity sizes, shooting speed, and ground friction.

    Ground friction is an exponential damping: v(t) = v0 * damping^t, so the
    total distance travelled is v0 / -ln(damping). Scaling damping by ** (1/s)
    makes the distance travelled (at the scaled shooting speed) proportional
    to s² of the baseline — i.e. the ball now travels 70% of its post-scaling
    distance for s=0.7 (≈49% of the original s=1 distance).
    """
    s = Settings.SIZE_MODIFIER
    Settings.BALL_RADIUS = round(15 * s)
    Settings.PLAYER_LEN = round(30 * s)
    Settings.PLAYER_SHOOTING_RANGE = round(10 * s)
    Settings.SHOOTING_SPEED = round(500 * s)
    Settings.GROUND_FRICTION = 0.5 ** (1.0 / s) if s > 0 else 0.5
Settings.PHYSICS_ENGINE = "numba" # "pymunk" or "numba"
Settings.N_WORKERS = 512 # nombre d'environnements simulés en parallèle (batch physique)
Settings.N_THREADS = 24  # nombre de threads Numba (prange) — laisser quelques cœurs au système

if Settings.IS_KAGGLE:
    Settings.SAVE_FOLDER = Path("/kaggle/working/models")
    Settings.CONFIG_FILE = Path("/kaggle/working/neat-config.cfg")
else:
    Settings.SAVE_FOLDER = "C:/.ingé/Projet-Sport-Co-Networks/"
    Settings.CONFIG_FILE = "C:/.ingé/Projet-Sport-Co/AI/Algorithms/config_feed-forward_neat.cfg"


































