# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:35:38 2025

@author: quent
"""

import math
import pygame
from PIL import Image, ImageDraw, ImageFont

from Settings import Settings


def initScreen():
    screen = pygame.display.set_mode((
        Settings.DIM_X + Settings.SCREEN_OFFSET * 2,
        Settings.DIM_Y + Settings.SCREEN_OFFSET * 2
    ))
    return screen


def startDisplay():
    pygame.init()
    screen = initScreen()
    return screen, None  # draw_options no longer needed


def display(engine, score, screen: pygame.Surface, draw_options=None) -> None:
    clear_screen(screen)
    render_data = engine.get_render_data()
    draw_objects(render_data, screen)
    draw_score(score, screen)
    pygame.display.flip()


def clear_screen(screen: pygame.Surface):
    screen.fill(Settings.BACKGROUND_COLOR)


def draw_objects(render_data: dict, screen: pygame.Surface) -> None:
    # Draw walls (grey, matching pymunk debug_draw style)
    wall_color = (104, 104, 104)
    for a, b, radius in render_data.get("walls", []):
        pygame.draw.line(screen, wall_color, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), max(1, int(radius * 2)))

    # Draw ball with angle indicator line (matching pymunk debug_draw)
    ball = render_data.get("ball")
    if ball is not None:
        bx, by = ball["position"]
        pos = (int(bx), int(by))
        color = ball["color"][:3]
        r = ball["radius"]
        pygame.draw.circle(screen, color, pos, r)
        # Angle indicator line inside the ball
        angle = ball.get("angle", 0)
        end_x = bx + r * math.cos(angle)
        end_y = by + r * math.sin(angle)
        pygame.draw.line(screen, (0, 0, 0), pos, (int(end_x), int(end_y)), 1)

    arrow_color = Settings.PLAYER_ARROW_COLOR[:3]
    arrow_size = Settings.PLAYER_LEN / 4

    # Draw players and their directional arrows
    for player in render_data["players"]:
        # Draw the player's square body
        vertices = [(int(x), int(y)) for x, y in player["vertices"]]
        color = player["color"][:3]
        pygame.draw.polygon(screen, color, vertices)

        pos = player["position"]
        angle = player["angle"]

        # Arrow points in local space, rotated and translated
        def rotate_point(lx, ly):
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            return (pos[0] + rx, pos[1] + ry)

        back = rotate_point(-arrow_size, 0)
        front = rotate_point(arrow_size, 0)
        top = rotate_point(0, -arrow_size)
        bottom = rotate_point(0, arrow_size)

        pygame.draw.line(screen, arrow_color, back, front, 2)
        pygame.draw.line(screen, arrow_color, front, top, 2)
        pygame.draw.line(screen, arrow_color, front, bottom, 2)


def draw_score(score, screen: pygame.Surface) -> None:
    score_left, score_right = score
    text = f"{int(score_left)}   -   {int(score_right)}"
    text_color = Settings.SCORE_COLOR

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    pad_x, pad_y = 5, 5
    img = Image.new("RGBA", (text_width + 2*pad_x, text_height + 2*pad_y), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=text_color + (255,))

    mode = img.mode
    size = img.size
    data = img.tobytes()
    surf = pygame.image.fromstring(data, size, mode)

    screen_width = Settings.DIM_X + 2 * Settings.SCREEN_OFFSET
    pos = (screen_width // 2 - text_width // 2, 20)

    screen.blit(surf, pos)
