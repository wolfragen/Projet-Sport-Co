# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:20:06 2025

@author: quent
"""

import numpy as np

from Settings import Settings


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
