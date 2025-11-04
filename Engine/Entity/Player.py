# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:20:06 2025

@author: quent
"""

import pymunk
import numpy as np

import Settings


def buildPlayers(space, players_number: list[int,int], human: bool = False):
    # Screen parameters
    offset = Settings.SCREEN_OFFSET
    dim_x = Settings.DIM_X
    dim_y = Settings.DIM_Y

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
        body.previous_position = pos
        body.angle = angle

        shape = pymunk.Poly.create_box(body, (size, size))
        shape.elasticity = Settings.PLAYER_ELASTICITY
        shape.friction = Settings.PLAYER_FRICTION
        shape.color = color
        shape.is_player = True
        shape.left_team = left_team

        return body, shape
    
    def spacing(n_players, size, offset_x, offset_y, dim_x, dim_y, revert_x=False):
        """
        Compute player positions with:
        - Equal spacing between players and walls
        - Symmetry on Y-axis
        - Balanced X-axis
        - Optionally revert columns (first column becomes last)
        - Per-column vertical centering
        """
        # --- Determine best cols/rows ---
        best_diff = float('inf')
        best_cols = None
    
        for cols in range(1, n_players + 1):
            rows = int(np.ceil(n_players / cols))
            spacing_x = dim_x / (cols + 1)
            spacing_y = dim_y / (rows + 1)
            diff = abs(spacing_x - spacing_y)
            if diff < best_diff:
                best_diff = diff
                best_cols = cols
    
        cols = best_cols
        spacing_x = dim_x / (cols + 1)
    
        # --- Determine players per column ---
        base_count = n_players // cols
        extra = n_players % cols
        col_counts = [base_count + 1 if i < extra else base_count for i in range(cols)]
    
        # --- X coordinates per column ---
        x_coords = np.linspace(spacing_x, spacing_x * cols, cols)
    
        # --- Reverse columns if needed ---
        if revert_x:
            x_coords = x_coords[::-1]
    
        # --- Assign positions per column ---
        positions = []
        for x, n_in_col in zip(x_coords, col_counts):
            y_coords = np.linspace(dim_y / (n_in_col + 1), dim_y * n_in_col / (n_in_col + 1), n_in_col)
            x_col = np.full(n_in_col, x)
            positions.append(np.column_stack([x_col, y_coords]))
    
        positions = np.vstack(positions)
    
        # --- Mirror Y-axis for symmetry ---
        center_y = dim_y / 2
        mirrored_y = 2 * center_y - positions[:, 1]
        positions_sym_y = np.vstack([positions, np.column_stack([positions[:, 0], mirrored_y])])
    
        # --- Take only first n_players ---
        positions_final = positions_sym_y[:n_players]
    
        # --- Apply offset ---
        positions_final += np.array([offset_x, offset_y])
    
        return positions_final

    n_left, n_right = players_number
    left_players = []
    right_players = []
    
    left_positions = spacing(n_left, size, offset, offset, dim_x/2, dim_y)

    for i in range(n_left):
        pos = left_positions[i]
        left_players.append(create_square(pos.tolist(), angle=0, left_team=True))
        
    if(n_right != 0):
        right_positions = spacing(n_right, size, dim_x/2+offset, offset, dim_x/2, dim_y, revert_x=True)
        for i in range(n_right):
            pos = right_positions[-i]
            right_players.append(create_square(pos.tolist(), angle=np.pi, left_team=False))

    # Flatten the lists
    left_flat = [item for pair in left_players for item in pair]
    right_flat = [item for pair in right_players for item in pair]

    # Add to space and update game dictionary
    space.add(*left_flat, *right_flat)
    players = left_players + right_players
    players_left = left_players
    players_right = right_players
    selected_player = None
    if(human):
        selected_player = 0

    return players, players_left, players_right, selected_player










