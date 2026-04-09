# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Optional

from Engine.Physics.EntityState import PlayerState, BallState


class PhysicsEngine(ABC):
    """Abstract interface for the physics backend."""

    # --- Lifecycle ---

    @abstractmethod
    def create_world(self) -> None:
        """Create the physics world/space with damping."""
        ...

    @abstractmethod
    def build_board(self) -> tuple[tuple, tuple]:
        """Build walls and goals. Returns (left_goal_position, right_goal_position)."""
        ...

    @abstractmethod
    def build_ball(self) -> None:
        """Create the ball entity."""
        ...

    @abstractmethod
    def build_players(self, players_number: tuple[int, int], human: bool = False,
                      phantom_player=None) -> Optional[int]:
        """Create player entities. Returns selected_player id or None."""
        ...

    @abstractmethod
    def step(self, dt: float, n_substeps: int) -> None:
        """Advance simulation by dt milliseconds, using n_substeps."""
        ...

    # --- State Access ---

    @abstractmethod
    def get_ball_state(self) -> BallState:
        ...

    @abstractmethod
    def get_player_state(self, player_id: int) -> PlayerState:
        ...

    @abstractmethod
    def get_all_player_states(self) -> list[PlayerState]:
        ...

    @abstractmethod
    def get_n_players(self) -> int:
        ...

    # --- State Mutation ---

    @abstractmethod
    def set_velocity(self, entity_id: int, vx: float, vy: float) -> None:
        ...

    @abstractmethod
    def set_angular_velocity(self, entity_id: int, omega: float) -> None:
        ...

    @abstractmethod
    def set_angle(self, entity_id: int, angle: float) -> None:
        ...

    @abstractmethod
    def set_position(self, entity_id: int, x: float, y: float) -> None:
        ...

    # --- Bookkeeping ---

    @abstractmethod
    def snapshot_previous_positions(self) -> None:
        """Store current positions/angles as 'previous' for delta computations."""
        ...

    @abstractmethod
    def update_previous_position(self, entity_id: int) -> None:
        """Update previous_position to current position for a single entity."""
        ...

    @abstractmethod
    def update_can_shoot(self, player_id: int, value: bool) -> None:
        ...

    @abstractmethod
    def update_had_ball(self, player_id: int, value: bool) -> None:
        ...

    @abstractmethod
    def set_last_player_shoot(self, value: Optional[int]) -> None:
        ...

    # --- Direct field access (no dataclass allocation) ---

    @abstractmethod
    def get_position(self, entity_id: int) -> tuple[float, float]:
        ...

    @abstractmethod
    def get_velocity(self, entity_id: int) -> tuple[float, float]:
        ...

    @abstractmethod
    def get_angle(self, entity_id: int) -> float:
        ...

    @abstractmethod
    def get_left_team(self, player_id: int) -> bool:
        ...

    @abstractmethod
    def get_can_shoot(self, player_id: int) -> bool:
        ...

    @abstractmethod
    def get_had_ball(self, player_id: int) -> bool:
        ...

    @abstractmethod
    def get_previous_position(self, entity_id: int) -> tuple[float, float]:
        ...

    @abstractmethod
    def get_last_player_shoot(self) -> Optional[int]:
        ...

    @abstractmethod
    def get_ball_radius(self) -> float:
        ...

    # --- Raycasting ---

    @abstractmethod
    def raycast(self, origin: tuple, end: tuple, radius: float,
                exclude_entity_id: Optional[int] = None) -> Optional[dict]:
        """
        Cast a ray segment. Returns None or dict with keys:
            'point': tuple (x, y) - hit point
            'entity_type': int - 0=nothing, 1=wall, 2=left_goal, 3=right_goal,
                                 4=ball, 5=left_player, 6=right_player
        """
        ...

    # --- Rendering ---

    @abstractmethod
    def get_render_data(self) -> dict:
        """Return all data needed for rendering (positions, shapes, colors, angles)."""
        ...
