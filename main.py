# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
from typing import Callable
import torch
import random

import Settings
import Graphics.GraphicEngine as GE
import Player.PlayerActions as PActions
import AI.AIActions as AIActions
import Engine.Actions as Actions
import Engine.Utils as Utils
import Engine.Vision as Vision
import AI.Network as nn


should_continue = True

def initGame(players_number: list[int,int] = [1,1], score : np.array = np.zeros(2, dtype=np.uint8), human=True) -> dict:
    """
    Initiate a new game from a given score, and return the dict "game".
    
    Parameters
   -------
    score : np.ndarray, shape (2,), dtype=np.uint8, optional
        Score array. Default is np.zeros(2, dtype=np.uint8).
    
    Returns
   ----
    dict
        A dictionary containing:
        
        space : pymunk.Space
            The physics simulation space, containing all environment data.
            
        score : np.ndarray, shape (2,), dtype=np.uint8
            The current score (left, right).
            
        ball : (pymunk.Body, pymunk.Shape)
            The ball body and its shape.
            
        players : list of (pymunk.Body, pymunk.Shape)
            All players in the game.
            
        players_left : list of (pymunk.Body, pymunk.Shape)
            Players on the left team.
            
        players_right : list of (pymunk.Body, pymunk.Shape)
            Players on the right team.
            
        selected_player : (pymunk.Body, pymunk.Shape)
            The currently selected player.
            
        left_goal_position : tuple of float
            Position (x, y) of the left goal center.
            
        right_goal_position : tuple of float
            Position (x, y) of the right goal center.
    """
    
    space = pymunk.Space() # Physics simulation space
    space.damping = Settings.GROUND_FRICTION # Simulate the friction with the ground
    game = {
        "space": space,
        "score": score
        }
    
    GE.buildBoard(game) # Creates static objects
    GE.buildBall(game) # Creates the ball
    GE.buildPlayers(game, players_number, human) # Creates the players
    
    return game

def start_display():
    global should_continue
    should_continue = True
    pygame.init()
    screen, draw_options = GE.initScreen()
    return screen, draw_options

def stopGame() -> None:
    """
    Stop the current game and close all active windows.

    Returns
   ----
    None
        This function does not return any value.
    """
    
    global should_continue
    should_continue = False
    
    pygame.display.quit()
    pygame.quit()
    return


def main(players_number, models: nn.DeepRLNetwork = None, human: bool = True, display: bool = True, debug: bool = False) -> None:
    """
    Main game loop with optional graphical display.

    Parameters
    ----------
    players_number : list[int,int]
        Number of players per team.
    models : list[nn.DeepRLNetwork] or None
        AI models for each player.
    human : bool
        Whether a human is playing.
    display : bool
        If True, use Pygame display; if False, run headless.
    """
    
    global should_continue
    should_continue = True

    n_players = players_number[0] + players_number[1]
    if models is None:
        models = [None for i in range(n_players)]
    assert len(models) == n_players

    game = initGame(players_number, human=human)

    if display:
        screen, draw_options = start_display()
        clock = pygame.time.Clock()
        delta_time = Settings.DELTA_TIME
        fps = int(1000/delta_time)
        min_delta_time = 1000/Settings.MAX_FPS
        time = 0
        GE.display(game, screen, draw_options)

    while should_continue:
        if display:
            PActions.process_events(game, stopGame)  # human actions

        human_player = game["selected_player"]

        for i in range(n_players):
            player = game["players"][i]
            model = models[i]
            if player != human_player:
                action_array = AIActions.play(game, player, model=model)
                if debug:
                    print(f"{action_array=}")

        # Physics steps
        for _ in range(Settings.DELTA_TIME):
            game["space"].step(0.001)

        Actions.reset_movements(game)

        if should_continue and display:
            time += delta_time
            if time >= min_delta_time:
                GE.display(game, screen, draw_options)
                time -= min_delta_time
            clock.tick(fps)

        scored = Utils.checkIfGoal(game)
        for i in range(n_players):
            player = game["players"][i]
            if player != human_player:
                compute_reward(game, player, scored, debug)

        Utils.checkPlayersOut(game["players"])

        if bool(scored[0]):
            new_game = initGame(players_number, game["score"], human=human)
            game.update(new_game)

    if display:
        pygame.quit()


def compute_reward(game: dict, player: tuple[pymunk.Body, pymunk.Shape], scored: tuple[bool,bool], debug: bool =False) -> float:
    """
    Compute the reward for a given player based on the current game state.
    
    Parameters
   -------
    game : dict
        Current game state.
    player : tuple(pymunk.Body, pymunk.Shape)
        The player to compute reward for.
    scored : tuple[bool,bool]
        (has_scored, left_team_scored)
    
    Returns
   ----
    float
        The reward value for this player at this timestep.
    """
    body, shape = player
    ball_body, ball_shape = game["ball"]
    has_scored, left_team_scored = scored
    
    left_goal = game["left_goal_position"]
    right_goal = game["right_goal_position"]
    
    offset = Settings.SCREEN_OFFSET
    max_dist = Settings.MAX_DIST
    
    delta_time = Settings.DELTA_TIME
    player_speed = Settings.PLAYER_SPEED
    player_rotating_speed = Settings.PLAYER_ROT_SPEED
    shooting_speed = Settings.SHOOTING_SPEED
    
    reward = 0.0
        
    alpha, beta = 1.0, 0.5  # relative weights
    
    static_reward = -0.5
    delta_ball_player_coeff = 0
    dist_ball_player_coeff = 0
    delta_ball_goal_coeff = 0
    dist_ball_goal_coeff = 0
    goal_coeff = 0
    out_of_bound_coeff = 0
    delta_angle_coeff = 1
    angle_diff_coeff = 1
    
    # Distance to ball
    prev_dx = (ball_body.previous_position[0] - body.previous_position[0])
    prev_dy = (ball_body.previous_position[1] - body.previous_position[1])
    prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
    
    curr_dx = (ball_body.position[0] - body.position[0])
    curr_dy = (ball_body.position[1] - body.position[1])
    curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    
    delta = prev_dist - curr_dist
    delta_ball_player_reward = 0
    if(abs(delta) > (player_speed * delta_time/1000)/1000):
        delta_ball_player_reward = delta_ball_player_coeff* delta / (player_speed * delta_time/1000)
        
    dist_ball_player_reward = -dist_ball_player_coeff* curr_dist/max_dist
    
    
    # Distance of ball to opponent goal
    if shape.left_team:
        prev_dx = (right_goal[0] - ball_body.previous_position[0])
        prev_dy = (right_goal[1] - ball_body.previous_position[1])
        prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
        
        curr_dx = (right_goal[0] - ball_body.previous_position[0])
        curr_dy = (right_goal[1] - ball_body.previous_position[1])
        curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    else:
        prev_dx = (left_goal[0] - ball_body.previous_position[0])
        prev_dy = (left_goal[1] - ball_body.previous_position[1])
        prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
        
        curr_dx = (left_goal[0] - ball_body.previous_position[0])
        curr_dy = (left_goal[1] - ball_body.previous_position[1])
        curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    
    delta = prev_dist - curr_dist
    delta_ball_goal_reward = 0
    if(abs(delta) > (shooting_speed * delta_time/1000)/1000):
        delta_ball_goal_reward = delta_ball_goal_coeff* delta / (shooting_speed * delta_time/1000)
        
    dist_ball_goal_reward = -dist_ball_goal_coeff* curr_dist/max_dist
        
        
    # Angle between player and ball
    vec_ball = np.array([ball_body.position[0] - body.position[0], ball_body.position[1] - body.position[1]])
    angle_to_ball = np.arctan2(vec_ball[1], vec_ball[0])
    angle_diff = (angle_to_ball - body.angle + np.pi) % (2*np.pi) - np.pi # back to -pi ; +pi
    max_angle = np.deg2rad(10)
    angle_diff_reward = 0
    if abs(angle_diff) <= max_angle:
        angle_diff_reward = angle_diff_coeff*5
    else:
        angle_diff_reward = angle_diff_coeff * (1 - abs(angle_diff/np.pi)*2)
        
    previous_vec_ball = np.array([ball_body.previous_position[0] - body.previous_position[0], ball_body.previous_position[1] - body.previous_position[1]])
    previous_angle_to_ball = np.arctan2(previous_vec_ball[1], previous_vec_ball[0])
    previous_angle_diff = (previous_angle_to_ball - body.previous_angle + np.pi) % (2*np.pi) - np.pi # back to -pi ; +pi
    
    delta = (abs(previous_angle_diff) - abs(angle_diff) + np.pi) % (2*np.pi) - np.pi
    delta_angle_reward = 0
    if(abs(delta) > (player_rotating_speed * delta_time/1000)/np.pi):
        delta_angle_reward = delta_angle_coeff * delta / (player_rotating_speed * delta_time/1000)
    
    # Penalize for being out of bounds
    x, y = body.position
    out_of_bound_reward = 0
    if (x < offset or x > Settings.DIM_X + offset):
        out_of_bound_reward = -out_of_bound_coeff
    
    # Scoring
    goal_reward = 0
    if(has_scored):
        # goal
        if shape.left_team and left_team_scored:
            goal_reward = goal_coeff
        elif (not shape.left_team) and (not left_team_scored):
            goal_reward = goal_coeff
        else:
            goal_reward = -goal_coeff
    
    reward = (static_reward + goal_reward + out_of_bound_reward + delta_angle_reward 
            + angle_diff_reward + dist_ball_player_reward + delta_ball_goal_reward 
            + delta_ball_player_reward + dist_ball_goal_reward)
    if(debug):
        print("------------")
        if(static_reward != 0): print(f"{static_reward=}")
        if(goal_reward != 0): print(f"{goal_reward=}")
        if(out_of_bound_reward != 0): print(f"{out_of_bound_reward=}")
        if(delta_angle_reward != 0): print(f"{delta_angle_reward=}")
        if(angle_diff_reward != 0): print(f"{angle_diff_reward=}")
        if(dist_ball_player_reward != 0): print(f"{dist_ball_player_reward=}")
        if(delta_ball_goal_reward != 0): print(f"{delta_ball_goal_reward=}")
        if(dist_ball_goal_reward != 0): print(f"{dist_ball_goal_reward=}")
        if(delta_ball_player_reward != 0): print(f"{delta_ball_player_reward=}")
        print(f"TOTAL REWARD = {reward}")
    
    
    return reward/10


def simulate_episode(
    players_number: list[int,int],
    models: list[nn.DeepRLNetwork],
    max_steps: int,
    scoring_function: Callable[[dict, tuple[pymunk.Body, pymunk.Shape], bool], float],
    epsilon: float = 0.0,
    display: bool = False,
    screen=None,
    draw_options=None,
    ) -> list[dict]:
    """
    Run one full simulation episode and collect experience tuples for DeepRL training.

    Parameters
    ----------
    players_number : list[int,int]
        Number of players per team.
    models : list[nn.DeepRLNetwork]
        AI models for each player.
    max_steps : int
        Maximum number of timesteps to simulate.
    scoring_function : Callable
        Function that computes the reward for a given player.
    epsilon : float
        Probability of taking random action.
    display : bool
        If True, shows the game via Pygame.

    Returns
    -------
    experiences : list[dict]
        Each element corresponds to one player's trajectory.
    """

    # Initialization
    game = initGame(players_number)
    players = game["players"]
    n_players = len(players)

    if should_continue and display:
        GE.display(game, screen, draw_options)
        clock = pygame.time.Clock()
        delta_time = Settings.DELTA_TIME
        fps = int(1000/delta_time)
        min_delta_time = 1000/Settings.MAX_FPS
        time = 0

    # Store experience per player
    experiences = [
        {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}
        for _ in range(n_players)
    ]

    visions = [Vision.getVision(game, p) for p in players]

    for step in range(max_steps):
        # Actions
        actions_t = []
        for i, player in enumerate(players):
            model = models[i]
            if random.random() < epsilon:
                action = AIActions.play(game, player, vision=visions[i])  # random
            else:
                action = AIActions.play(game, player, model=model, vision=visions[i])
            actions_t.append(action)

        # Physics step
        for _ in range(Settings.DELTA_TIME):
            game["space"].step(0.001)

        Actions.reset_movements(game)
        scored = Utils.checkIfGoal(game)
        done = bool(scored[0])

        # Compute rewards & next states
        for i, player in enumerate(players):
            reward = scoring_function(game, player, scored)
            reward = np.float32(reward)
            next_vision = Vision.getVision(game, player)

            exp = experiences[i]
            exp["states"].append(visions[i])
            exp["actions"].append(actions_t[i])
            exp["rewards"].append(reward)
            exp["next_states"].append(next_vision)
            exp["dones"].append(done)
            visions[i] = next_vision

        Utils.checkPlayersOut(players)


        if should_continue and display:
            PActions.process_events(game, stopGame)  # human actions
            if should_continue:
                time += delta_time
                if time >= min_delta_time:
                    GE.display(game, screen, draw_options)
                    time -= min_delta_time
                clock.tick(fps)

        if done:
            break

    # Convert lists to arrays
    for exp in experiences:
        exp["states"] = np.array(exp["states"], dtype=np.float32)
        exp["actions"] = np.array(exp["actions"], dtype=np.float32)
        exp["rewards"] = np.array(exp["rewards"], dtype=np.float32)
        exp["next_states"] = np.array(exp["next_states"], dtype=np.float32)
        exp["dones"] = np.array(exp["dones"], dtype=bool)

    return experiences





if(__name__ == "__main__"):
    
    """
    # Simulation graphique avec un humain
    model = nn.DeepRLNetwork(dimensions=[Settings.ENTRY_NEURONS, 512, 256, 128, 4])
    models = [nn.load_network(model, "C:/.ingé/Projet-Sport-Co-Networks") for i in range(2)]
    main(players_number=(1,1), models=models, human=True, debug=True)
    """
    
    models = [nn.DeepRLNetwork(dimensions=[Settings.ENTRY_NEURONS, 512, 256, 128, 4]) for i in range(1)]
    
    # optimizer : adamax => 
    nn.train_dqn_for_duration(
        players_number=(1,0),
        models=models,
        optimizer_cls=torch.optim.Adam,
        lr=1e-4,
        epsilon_decay=600,
        simulate_episode=simulate_episode,
        scoring_function = compute_reward,
        max_duration_s=300,
        device="cpu",
        batch_size=512,
        buffer_size=50_000,
        display=False,
        start_display=start_display,
        initial_calls = 25,
        max_calls = 1000,
        steps_growth_rate = 1.005,
        delay_params = 50,
    )
    
    nn.save_network(models[0], "C:/.ingé/Projet-Sport-Co-Networks")
    






































