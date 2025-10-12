# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:35:38 2025

@author: quent
"""

import pygame
import pymunk
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import Settings


def initScreen():
    """
    Initialise pygame et pymunk ainsi que l'affichage.

    Parameters
    ----------
    game : TYPE
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """
    
    screen = pygame.display.set_mode((Settings.DIM_X+Settings.SCREEN_OFFSET*2, Settings.DIM_Y+Settings.SCREEN_OFFSET*2))
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    return screen, draw_options


def clear_screen(screen):
    
    screen.fill(pygame.Color("white"))
    return


def draw_objects(game, draw_options, screen):
    space = game["space"]
    
    space.debug_draw(draw_options)
    
    color = Settings.PLAYER_ARROW_COLOR
    size = Settings.PLAYER_LEN / 4
    
    # Dessiner l'avant des joueurs
    for body, shape in game.get("players", []):
        # dessiner le carré
        points = [p.rotated(body.angle) + body.position for p in shape.get_vertices()]
        pygame.draw.polygon(screen, shape.color, points)
    
        # points locaux du triangle/flèche
        back = pymunk.Vec2d(-size, 0)        # arrière du joueur
        front = pymunk.Vec2d(size, 0)        # avant du joueur
        top = pymunk.Vec2d(0, -size)  # coin haut de la pointe
        bottom = pymunk.Vec2d(0, size) # coin bas de la pointe
    
        # appliquer rotation et translation selon body
        back_world = back.rotated(body.angle) + body.position
        front_world = front.rotated(body.angle) + body.position
        top_world = top.rotated(body.angle) + body.position
        bottom_world = bottom.rotated(body.angle) + body.position
    
        # ligne centrale
        pygame.draw.line(screen, color, back_world, front_world, 2)
        # lignes de la pointe
        pygame.draw.line(screen, color, front_world, top_world, 2)
        pygame.draw.line(screen, color, front_world, bottom_world, 2)
    return


def display(game, screen, draw_options):
    
    clear_screen(screen)
    draw_objects(game, draw_options, screen)
    draw_score(game, screen)
    pygame.display.flip()
    
    return


def buildBoard(game):
    space = game["space"]
    static_body = space.static_body
    
    # Paramètres
    offset = Settings.SCREEN_OFFSET
    width = Settings.DIM_X
    height = Settings.DIM_Y
    goal_len = Settings.GOAL_LEN
    
    # Haut et bas
    top_line = (offset, offset), (offset + width, offset)
    bottom_line = (offset, offset + height), (offset + width, offset + height)
    
    # Côtés verticaux avec goals ouverts
    left_top_goal = offset + (height - goal_len)/2
    left_bottom_goal = offset + (height + goal_len)/2
    right_top_goal = offset + (height - goal_len)/2
    right_bottom_goal = offset + (height + goal_len)/2
    
    # Murs
    static_lines = [
        # Haut et bas
        pymunk.Segment(static_body, top_line[0], top_line[1], 0.0),
        pymunk.Segment(static_body, bottom_line[0], bottom_line[1], 0.0),
    
        # Gauche
        pymunk.Segment(static_body, (offset, offset), (offset, left_top_goal), 0.0),
        pymunk.Segment(static_body, (offset, left_bottom_goal), (offset, offset + height), 0.0),
    
        # Droite
        pymunk.Segment(static_body, (offset + width, offset), (offset + width, right_top_goal), 0.0),
        pymunk.Segment(static_body, (offset + width, right_bottom_goal), (offset + width, offset + height), 0.0),
    ]
    
    for line in static_lines:
        line.elasticity = Settings.WALL_ELASTICITY
        line.friction = Settings.WALL_FRICTION
    space.add(*static_lines)
    
    # === Cages invisibles pour la vision IA :D ===
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

    # Capteurs (pas de collisions)
    left_goal_sensor.sensor = True
    right_goal_sensor.sensor = True

    # Pour que l'IA les reconnaisse
    left_goal_sensor.collision_type = Settings.LEFT_GOAL_COLLISION_TYPE
    right_goal_sensor.collision_type = Settings.RIGHT_GOAL_COLLISION_TYPE

    # Invisible pour debug_draw
    left_goal_sensor.color = (0, 0, 0, 0)
    right_goal_sensor.color = (0, 0, 0, 0)

    space.add(left_goal_sensor, right_goal_sensor)

    return

def buildBall(game):
    space = game["space"]

    # Paramètres
    radius = Settings.BALL_RADIUS
    mass = Settings.BALL_MASS

    # Calcul du moment d'inertie
    moment = pymunk.moment_for_circle(mass, 0, radius)

    # Body dynamique
    body = pymunk.Body(mass, moment)
    body.position = (Settings.SCREEN_OFFSET + Settings.DIM_X/2,
                     Settings.SCREEN_OFFSET + Settings.DIM_Y/2) # milieu de terrain

    # Shape de la balle
    shape = pymunk.Circle(body, radius)
    shape.elasticity = Settings.BALL_ELASTICITY
    shape.friction = Settings.BALL_FRICTION
    
    shape.is_ball = True

    space.add(body, shape)
    game["ball"] = (body, shape)
    
    return

def buildPlayers(game, dim_x=Settings.DIM_X, dim_y=Settings.DIM_Y):
    #TODO adapter pour autre chose que du 1v1 => tuple de nombre de joueur (1,1) ?
    space = game["space"]

    # Paramètres
    size = Settings.PLAYER_LEN
    mass = Settings.PLAYER_MASS

    def create_square(pos, angle, left_team):
        color = Settings.PLAYER_RIGHT_COLOR
        if left_team:
            color = Settings.PLAYER_LEFT_COLOR
            
        moment = pymunk.moment_for_box(mass, (size, size))
        body = pymunk.Body(mass, moment)
        body.position = pos
        body.angle = angle
        
        shape = pymunk.Poly.create_box(body, (size, size))
        shape.elasticity = Settings.PLAYER_ELASTICITY
        shape.friction = Settings.PLAYER_FRICTION
        
        shape.color = color
        shape.is_player = True
        shape.left_team = left_team
        
        return body, shape

    # Joueur gauche
    left_pos = (Settings.SCREEN_OFFSET + Settings.DIM_X/4,
                Settings.SCREEN_OFFSET + Settings.DIM_Y/2)
    left_player = create_square(left_pos, angle=0, left_team=True)

    # Joueur droit
    right_pos = (Settings.SCREEN_OFFSET + 3*Settings.DIM_X/4,
                 Settings.SCREEN_OFFSET + Settings.DIM_Y/2)
    right_player = create_square(right_pos, angle=np.pi, left_team=False)

    space.add(*left_player, *right_player)
    game["players"] = [left_player, right_player]
    game["players_left"] = [left_player]
    game["players_right"] = [right_player]
    game["selected_player"] = left_player

    return


def draw_score(game, screen):
    score_left, score_right = game["score"]
    text = f"{score_left}   -   {score_right}"
    text_color = Settings.SCORE_COLOR
    
    # Police PIL (Arial, taille 36)
    try:
        font = ImageFont.truetype("arial.ttf", 36)  # vraie police TTF
    except IOError:
        font = ImageFont.load_default()  # fallback si Arial non dispo

    # Taille du texte
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Image PIL transparente
    pad_x, pad_y = 5, 5
    img = Image.new("RGBA", (text_width + 2*pad_x, text_height + 2*pad_y), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=text_color + (255,))  # (255) pour alpha => opacité

    # Convertir en surface Pygame
    mode = img.mode
    size = img.size
    data = img.tobytes()
    surf = pygame.image.fromstring(data, size, mode)

    # Position centrée horizontalement
    screen_width = Settings.DIM_X + 2 * Settings.SCREEN_OFFSET
    pos = (screen_width // 2 - text_width // 2, 20)

    screen.blit(surf, pos)
    return











































