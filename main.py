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

def initGame(players_number: list[int,int] = [1,1], score : np.array = np.zeros(2, dtype=np.uint8)) -> dict:
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
    GE.buildPlayers(game, players_number) # Creates the players
    
    return game

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


def main(players_number, models: nn.DeepRLNetwork = None) -> None:
    """
    Main game loop.  
    Handles initialization, player actions (human and AI), physics updates, display refresh, 
    and game state checks such as goals or out-of-bound players.

    Returns
   ----
    None
        This function does not return any value.
    """
    n_players = players_number[0] + players_number[1]
    
    if(models is None):
        models = [None for i in range(n_players)]
        
    assert len(models) == n_players
    
    pygame.init()
    game = initGame(players_number)
    screen, draw_options = GE.initScreen()
    
    clock = pygame.time.Clock() # Necessary to "force" loop time
    
    delta_time = Settings.DELTA_TIME
    fps = int(1000/delta_time)
    
    min_delta_time = 1000/Settings.MAX_FPS # Limits fps
    
    time = 0
    GE.display(game, screen, draw_options) # draw game
        
    while(should_continue):
        
        PActions.process_events(game, stopGame) # Human actions
        if(should_continue):
            
            human_player = game["selected_player"]
            
            for i in range(n_players):
                player = game["players"][i]
                model = models[i]
                
                if player != human_player:
                    AIActions.play(game, player, model=model) # AI Actions
            
            for step in range(delta_time): # Necessary to avoid tunneling
                game["space"].step(0.001) # 1 ms at a time
                
            time += delta_time
            Actions.reset_movements(game) # Resets velocity for all players
            
            if(time >= min_delta_time): # Fps
                GE.display(game, screen, draw_options)
                time -= min_delta_time
                
            clock.tick(fps) # Force the loop to trigger at a certain pace
            scored = Utils.checkIfGoal(game)
            Utils.checkPlayersOut(game["players"]) # Checks if players are out of bounds
            
            if(bool(scored[0])):
                new_game = initGame(players_number, game["score"])
                game.update(new_game)
    return


def compute_reward(game: dict, player: tuple[pymunk.Body, pymunk.Shape], scored: tuple[bool,bool]) -> float:
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
    dim_x = Settings.DIM_X
    dim_y = Settings.DIM_Y
    
    reward = 0.0
        
    alpha, beta = 1.0, 0.5  # relative weights
    
    # Distance to ball
    prev_dx = (ball_body.previous_position[0] - body.previous_position[0]) / dim_x
    prev_dy = (ball_body.previous_position[1] - body.previous_position[1]) / dim_y
    prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
    
    curr_dx = (ball_body.position[0] - body.position[0]) / dim_x
    curr_dy = (ball_body.position[1] - body.position[1]) / dim_y
    curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    
    delta = prev_dist - curr_dist
    reward += np.tanh(delta)/2 # récompense entre -1 et 1, normalement assez petite
    
    
    # Distance of ball to opponent goal
    if shape.left_team:
        prev_dx = (right_goal[0] - ball_body.previous_position[0]) / dim_x
        prev_dy = (right_goal[1] - ball_body.previous_position[1]) / dim_y
        prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
        
        curr_dx = (right_goal[0] - ball_body.previous_position[0]) / dim_x
        curr_dy = (right_goal[1] - ball_body.previous_position[1]) / dim_y
        curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    else:
        prev_dx = (left_goal[0] - ball_body.previous_position[0]) / dim_x
        prev_dy = (left_goal[1] - ball_body.previous_position[1]) / dim_y
        prev_dist = np.sqrt(alpha * prev_dx**2 + beta * prev_dy**2)
        
        curr_dx = (left_goal[0] - ball_body.previous_position[0]) / dim_x
        curr_dy = (left_goal[1] - ball_body.previous_position[1]) / dim_y
        curr_dist = np.sqrt(alpha * curr_dx**2 + beta * curr_dy**2)
    
    delta = prev_dist - curr_dist
    reward += np.tanh(delta) # récompense entre -1 et 1, normalement assez petite
    
    
    # Penalize for being out of bounds
    x, y = body.position
    offset = Settings.SCREEN_OFFSET
    if x < offset or x > Settings.DIM_X + offset:
        reward -= 25
    
    # Scoring
    if(has_scored):
        # goal
        if shape.left_team and left_team_scored:
            reward += 100
        elif (not shape.left_team) and (not left_team_scored):
            reward += 100
        else:
            reward -= 100
    
    return reward/100


def simulate_episode(
    players_number: list[int,int],
    models: list[nn.DeepRLNetwork],
    max_steps: int,
    scoring_function: Callable[[dict, tuple[pymunk.Body, pymunk.Shape], bool], float],
    epsilon: float = 0.0
    ) -> list[dict]:
    """
    Run one full simulation episode and collect experience tuples for DeepRL training.

    Parameters
   -------
    max_steps : int
        Maximum number of timesteps to simulate.
    scoring_function : Callable
        Function that computes the reward for a given player.

    Returns
   ----
    experiences : list of dict
        Each element corresponds to one player's trajectory, containing:
            'states'       : np.ndarray of shape (T, input_dim)
            'actions'      : np.ndarray of shape (T, output_dim)
            'rewards'      : np.ndarray of shape (T,)
            'next_states'  : np.ndarray of shape (T, input_dim)
            'dones'        : np.ndarray of shape (T,)
    """

    # Initialization
    game = initGame(players_number)
    players = game["players"]
    n_players = len(players)

    # Store experience per player
    experiences = [
        {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }
        for id_player in range(n_players)
    ]

    # Precompute initial vision (avoid recomputing unnecessarily)
    visions = [Vision.getVision(game, p) for p in players]

    for step in range(max_steps):
        ball_body, ball_shape = game["ball"]
        ball_body.previous_position = ball_body.position
        
        # Actions
        actions_t = []
        for i, player in enumerate(players):
            body, shape = player
            body.previous_position = body.position
            model = models[i]
            if random.random() < epsilon:
                action = AIActions.play(game, player, vision=visions[i]) # Random
            else:
                action = AIActions.play(game, player, model=model, vision=visions[i])
            actions_t.append(action)

        # Physics step
        for step in range(Settings.DELTA_TIME):
            game["space"].step(0.001)

        Actions.reset_movements(game)
        scored = Utils.checkIfGoal(game)
        done = bool(scored[0]) # episode ends when a goal is scored
        Utils.checkPlayersOut(players)

        # Compute rewards & next states
        for i, player in enumerate(players):
            reward = scoring_function(game, player, scored)
            reward = np.float32(reward)
            if(reward is None): print(reward)

            next_vision = Vision.getVision(game, player)

            # Store experience
            exp = experiences[i]
            exp["states"].append(visions[i])
            exp["actions"].append(actions_t[i])
            exp["rewards"].append(reward)
            exp["next_states"].append(next_vision)
            exp["dones"].append(done)

            # Update cached vision
            visions[i] = next_vision

        # End simulation early if goal scored
        if done:
            break

    # Convert lists to NumPy arrays for efficient training
    for exp in experiences:
        exp["states"] = np.array(exp["states"], dtype=np.float32)
        exp["actions"] = np.array(exp["actions"], dtype=np.float32)
        exp["rewards"] = np.array(exp["rewards"], dtype=np.float32)
        exp["next_states"] = np.array(exp["next_states"], dtype=np.float32)
        exp["dones"] = np.array(exp["dones"], dtype=bool)

    return experiences

"""
# Simulation graphique avec un humain
model = nn.DeepRLNetwork(dimensions=[456, 512, 256, 128, 8])
models = [nn.load_network(model, "C:/.ingé/Projet-Sport-Co-Networks") for i in range(16)]
main(players_number=(8,8), models = models)
"""

models = [nn.DeepRLNetwork(dimensions=[456, 512, 256, 128, 8]) for i in range(1)]

nn.train_dqn_for_duration(
    players_number=(1,0),
    models=models,
    optimizer_cls=torch.optim.Adam,
    lr=1e-4,
    simulate_episode=simulate_episode,
    scoring_function = compute_reward,
    max_duration_s=300,
)

nn.save_network(models[0], "C:/.ingé/Projet-Sport-Co-Networks")








































