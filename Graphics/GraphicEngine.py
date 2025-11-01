# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:35:38 2025

@author: quent
"""

import pygame
import pymunk
import pymunk.pygame_util
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

def startDisplay():
    pygame.init()
    screen, draw_options = initScreen()
    return screen, draw_options


def display(space, players, score, screen: pygame.Surface, draw_options: "pymunk.pygame_util.DrawOptions") -> None:
    # Clear the screen to a blank state
    clear_screen(screen)
    
    # Draw all game objects, including players and arrows
    draw_objects(space, players, draw_options, screen)
    
    # Draw the current score
    draw_score(score, screen)
    
    # Update the full display surface to the screen
    pygame.display.flip()
    
    return


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


def draw_objects(space, players, draw_options: "pymunk.pygame_util.DrawOptions", screen: pygame.Surface) -> None:
    # Draw all shapes using pymunk debug draw
    space.debug_draw(draw_options)
    
    arrow_color = Settings.PLAYER_ARROW_COLOR
    arrow_size = Settings.PLAYER_LEN / 4
    
    # Draw players and their directional arrows
    for body, shape in players:
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


def draw_score(score, screen: pygame.Surface) -> None:
    score_left, score_right = score
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











































