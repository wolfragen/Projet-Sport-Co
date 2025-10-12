# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:49:47 2025

@author: quent


Pour les types en rayTracing :
    - 0: rien trouvé
    - 1: mur
    - 2: goal gauche
    - 3: goal droit
    - 4: balle
    - 5: joueur gauche
    - 6: joueur droit

"""

import math
import numpy as np
import pymunk

import Settings


def rayTracing(game, player):
    body, shape = player
    space = game["space"]
    
    number_of_rays = Settings.NUMBER_OF_RAYS
    fov = Settings.RAY_ANGLE
    max_dist = Settings.VISION_RANGE
    
    start_angle = body.angle - fov / 2
    step = fov / (number_of_rays - 1)
    origin = body.position
    
    distances = np.full(number_of_rays, np.inf, dtype=np.float32)
    types = np.zeros(number_of_rays, dtype=np.int8)
    
    # Sauvegarde du filtre original
    original_filter = shape.filter
    
    # Ignorer le joueur en modifiant son mask
    shape.filter = pymunk.ShapeFilter(mask=0)
    
    for i in range(number_of_rays):
        angle = start_angle + i * step
        direction = pymunk.Vec2d(math.cos(angle), math.sin(angle))
        end_point = origin + direction * max_dist

        # Raytracing interne à pymunk
        ray_radius = int(max(1,round(Settings.PLAYER_LEN * 0.05)))
        hit = space.segment_query_first(origin, end_point, ray_radius, pymunk.ShapeFilter())

        if hit is not None:
            hit_shape = hit.shape
            distances[i] = hit.alpha * max_dist  # alpha = % de distance max
            
            # Identifier le type d'objet touché
            if hasattr(hit_shape, "is_ball") and hit_shape.is_ball:
                types[i] = np.uint8(4)
                
            elif hasattr(hit_shape, "is_player") and hit_shape.is_player:
                if hit_shape.left_team:
                    types[i] = np.uint8(5)# Joueur gauche
                else:
                    types[i] = np.uint8(6)  # Joueur droit
                    
            elif hit_shape.collision_type == Settings.LEFT_GOAL_COLLISION_TYPE:
                types[i] = np.uint8(2) # cage de gauche
            elif hit_shape.collision_type == Settings.RIGHT_GOAL_COLLISION_TYPE:
                types[i] = np.uint8(3) # cage de droite
                
            else:
                types[i] = np.uint8(1)  # mur par défaut
                
    shape.filter = original_filter

    return distances, types











































