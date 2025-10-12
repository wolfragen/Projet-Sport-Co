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
    Initialize the Pygame window and Pymunk drawing options.

    This function sets up the game display using the defined settings 
    (screen dimensions and offset) and returns the Pygame surface and 
    corresponding Pymunk drawing options used to render the game.

    Returns
    -------
    tuple
        screen : pygame.Surface  
            The main Pygame display surface used for rendering the game.  
        draw_options : pymunk.pygame_util.DrawOptions  
            The drawing configuration object for rendering Pymunk objects.
    """
    
    # Create the Pygame window with defined dimensions and offset
    screen = pygame.display.set_mode((
        Settings.DIM_X + Settings.SCREEN_OFFSET * 2,
        Settings.DIM_Y + Settings.SCREEN_OFFSET * 2
    ))
    
    # Initialize Pymunk draw options (for visualizing physics objects)
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # Return both display surface and drawing configuration
    return screen, draw_options


def clear_screen(screen: pygame.Surface):
    """
    Clear the given Pygame screen by filling it with white color.

    Parameters
    ----------
    screen : pygame.Surface
        The Pygame surface to clear.

    Returns
    -------
    None
        The screen is cleared in-place.
    """
    
    # Fill the screen with white to reset previous drawings
    screen.fill(Settings.BACKGROUND_COLOR)
    return


def draw_objects(game: dict, draw_options: "pymunk.pygame_util.DrawOptions", screen: pygame.Surface) -> None:
    """
    Draws all game objects on the screen, including players with their directional arrows.

    Parameters
    ----------
    game : dict
        The game state dictionary containing physics space and all objects.
    draw_options : pymunk.pygame_util.DrawOptions
        Pymunk draw options for rendering shapes.
    screen : pygame.Surface
        The Pygame surface to draw on.

    Returns
    -------
    None
        The function draws directly on the screen; does not return anything.
    """

    space = game["space"]
    
    # Draw all shapes using pymunk debug draw
    space.debug_draw(draw_options)
    
    arrow_color = Settings.PLAYER_ARROW_COLOR
    arrow_size = Settings.PLAYER_LEN / 4
    
    # Draw players and their directional arrows
    for body, shape in game.get("players", []):
        # Draw the player's square body
        points = [p.rotated(body.angle) + body.position for p in shape.get_vertices()]
        pygame.draw.polygon(screen, shape.color, points)
        
        # Define local coordinates for the arrow (triangle)
        back = pymunk.Vec2d(-arrow_size, 0)     # back of the player
        front = pymunk.Vec2d(arrow_size, 0)     # front of the player
        top = pymunk.Vec2d(0, -arrow_size)      # top point of arrow
        bottom = pymunk.Vec2d(0, arrow_size)    # bottom point of arrow
        
        # Rotate and translate points according to the player's body
        back_world = back.rotated(body.angle) + body.position
        front_world = front.rotated(body.angle) + body.position
        top_world = top.rotated(body.angle) + body.position
        bottom_world = bottom.rotated(body.angle) + body.position
        
        # Draw the central line of the arrow
        pygame.draw.line(screen, arrow_color, back_world, front_world, 2)
        # Draw the two lines forming the arrowhead
        pygame.draw.line(screen, arrow_color, front_world, top_world, 2)
        pygame.draw.line(screen, arrow_color, front_world, bottom_world, 2)
        
    return


def display(game: dict, screen: pygame.Surface, draw_options: "pymunk.pygame_util.DrawOptions") -> None:
    """
    Refresh the game display: clears the screen, draws all objects, shows the score, and updates the display.

    Parameters
    ----------
    game : dict
        The current game state dictionary containing players, ball, and scores.
    screen : pygame.Surface
        The Pygame surface to draw on.
    draw_options : pymunk.pygame_util.DrawOptions
        Pymunk draw options for rendering shapes.

    Returns
    -------
    None
        Draws directly to the screen; does not return a value.
    """

    # Clear the screen to a blank state
    clear_screen(screen)
    
    # Draw all game objects, including players and arrows
    draw_objects(game, draw_options, screen)
    
    # Draw the current score
    draw_score(game, screen)
    
    # Update the full display surface to the screen
    pygame.display.flip()
    
    return


def buildBoard(game: dict) -> None:
    """
    Builds the game field including walls, goals, and invisible goal sensors for AI vision.

    Parameters
    ----------
    game : dict
        The game state dictionary containing at least a 'space' key with the Pymunk space.

    Returns
    -------
    None
        Modifies the 'space' inside the game dictionary by adding static walls and goal sensors.
    """

    space = game["space"]
    static_body = space.static_body

    # Field parameters
    offset = Settings.SCREEN_OFFSET
    width = Settings.DIM_X
    height = Settings.DIM_Y
    goal_len = Settings.GOAL_LEN

    # Top and bottom lines
    top_line = (offset, offset), (offset + width, offset)
    bottom_line = (offset, offset + height), (offset + width, offset + height)

    # Vertical sides with goal openings
    left_top_goal = offset + (height - goal_len)/2
    left_bottom_goal = offset + (height + goal_len)/2
    right_top_goal = offset + (height - goal_len)/2
    right_bottom_goal = offset + (height + goal_len)/2

    # Static walls
    static_lines = [
        # Top and bottom
        pymunk.Segment(static_body, top_line[0], top_line[1], 0.0),
        pymunk.Segment(static_body, bottom_line[0], bottom_line[1], 0.0),

        # Left side with goal opening
        pymunk.Segment(static_body, (offset, offset), (offset, left_top_goal), 0.0),
        pymunk.Segment(static_body, (offset, left_bottom_goal), (offset, offset + height), 0.0),

        # Right side with goal opening
        pymunk.Segment(static_body, (offset + width, offset), (offset + width, right_top_goal), 0.0),
        pymunk.Segment(static_body, (offset + width, right_bottom_goal), (offset + width, offset + height), 0.0),
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
    game["left_goal_position"] = (offset, (left_bottom_goal + left_top_goal)/2)
    game["right_goal_position"] = (offset + width, (right_bottom_goal + right_top_goal)/2)

    return


def buildBall(game: dict) -> None:
    """
    Creates the ball in the center of the field with physical properties and adds it to the game space.

    Parameters
    ----------
    game : dict
        The game state dictionary containing at least a 'space' key with the Pymunk space.

    Returns
    -------
    None
        Adds the ball body and shape to the Pymunk space and updates the 'ball' key in the game dictionary.
    """

    space = game["space"]

    # Ball parameters
    radius = Settings.BALL_RADIUS
    mass = Settings.BALL_MASS

    # Calculate moment of inertia
    moment = pymunk.moment_for_circle(mass, 0, radius)

    # Dynamic body for the ball
    body = pymunk.Body(mass, moment)
    body.position = (
        Settings.SCREEN_OFFSET + Settings.DIM_X / 2,
        Settings.SCREEN_OFFSET + Settings.DIM_Y / 2
    )  # center of the field

    # Circle shape for the ball
    shape = pymunk.Circle(body, radius)
    shape.elasticity = Settings.BALL_ELASTICITY
    shape.friction = Settings.BALL_FRICTION
    shape.color = Settings.BALL_COLOR

    # Tag for ray tracing or identification
    shape.is_ball = True

    # Add to space and update game dictionary
    space.add(body, shape)
    game["ball"] = (body, shape)
    
    return


def buildPlayers(game: dict, dim_x: float = Settings.DIM_X, dim_y: float = Settings.DIM_Y) -> None:
    """
    Creates the player bodies and shapes for a 1v1 game and adds them to the Pymunk space.

    Parameters
    ----------
    game : dict
        The game state dictionary containing at least a 'space' key with the Pymunk space.
    dim_x : float, optional
        Width of the field, default is Settings.DIM_X.
    dim_y : float, optional
        Height of the field, default is Settings.DIM_Y.

    Returns
    -------
    None
        Adds player bodies and shapes to the Pymunk space and updates the corresponding keys in the game dictionary:
        'players', 'players_left', 'players_right', and 'selected_player'.
    """

    space = game["space"]

    # Player parameters
    size = Settings.PLAYER_LEN
    mass = Settings.PLAYER_MASS

    def create_square(pos: tuple[float, float], angle: float, left_team: bool) -> tuple[pymunk.Body, pymunk.Poly]:
        """
        Helper function to create a square player with given position, angle, and team.
        """
        color = Settings.PLAYER_LEFT_COLOR if left_team else Settings.PLAYER_RIGHT_COLOR
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

    # Left player
    left_pos = (Settings.SCREEN_OFFSET + dim_x / 4, Settings.SCREEN_OFFSET + dim_y / 2)
    left_player = create_square(left_pos, angle=0, left_team=True)

    # Right player
    right_pos = (Settings.SCREEN_OFFSET + 3 * dim_x / 4, Settings.SCREEN_OFFSET + dim_y / 2)
    right_player = create_square(right_pos, angle=np.pi, left_team=False)

    # Add to space and update game dictionary
    space.add(*left_player, *right_player)
    game["players"] = [left_player, right_player]
    game["players_left"] = [left_player]
    game["players_right"] = [right_player]
    game["selected_player"] = left_player

    return


def draw_score(game: dict, screen: pygame.Surface) -> None:
    """
    Draws the current score on the screen using PIL for text rendering.

    Parameters
    ----------
    game : dict
        Game state dictionary containing the 'score' key as a 2-element array.
    screen : pygame.Surface
        The Pygame surface on which to draw the score.

    Returns
    -------
    None
        The function draws the score on the screen and does not return anything.
    """

    score_left, score_right = game["score"]
    text = f"{score_left}   -   {score_right}"
    text_color = Settings.SCORE_COLOR

    # Load font using PIL (Arial, size 36)
    try:
        font = ImageFont.truetype("arial.ttf", 36)  # TrueType font
    except IOError:
        font = ImageFont.load_default()  # Fallback if Arial unavailable

    # Calculate text size
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Create transparent PIL image
    pad_x, pad_y = 5, 5
    img = Image.new("RGBA", (text_width + 2*pad_x, text_height + 2*pad_y), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=text_color + (255,))  # 255 = alpha

    # Convert PIL image to Pygame surface
    mode = img.mode
    size = img.size
    data = img.tobytes()
    surf = pygame.image.fromstring(data, size, mode)

    # Center horizontally on screen
    screen_width = Settings.DIM_X + 2 * Settings.SCREEN_OFFSET
    pos = (screen_width // 2 - text_width // 2, 20)

    screen.blit(surf, pos)
    
    return











































