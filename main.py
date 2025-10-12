# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import numpy as np
import pymunk
import pymunk.pygame_util
import pygame

import Settings
import Graphics.GraphicEngine as GE
import Player.PlayerActions as PActions
import AI.AIActions as AIActions
import Engine.Actions as Actions
import Engine.Utils as Utils


should_continue = True

def initGame(score : np.array = np.zeros(2, dtype=np.uint8)) -> dict:
    """
    Initiate a new game from a given score, and return the dict "game".
    
    Parameters
    ----------
    score : np.ndarray, shape (2,), dtype=np.uint8, optional
        Score array. Default is np.zeros(2, dtype=np.uint8).
    
    Returns
    -------
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
    GE.buildPlayers(game) # Creates the players
    
    return game

def stopGame() -> None:
    """
    Stop the current game and close all active windows.

    Returns
    -------
    None
        This function does not return any value.
    """
    
    global should_continue
    should_continue = False
    
    pygame.display.quit()
    pygame.quit()
    return


def main() -> None:
    """
    Main game loop.  
    Handles initialization, player actions (human and AI), physics updates, display refresh, 
    and game state checks such as goals or out-of-bound players.

    Returns
    -------
    None
        This function does not return any value.
    """
    
    pygame.init()
    game = initGame()
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
            
            for player in game["players"]:
                if player != human_player:
                    AIActions.play(game, player) # AI Actions
            
            for step in range(delta_time): # Necessary to limit "teleportation" issues
                game["space"].step(0.001) # 1 ms at a time
                
            time += delta_time
            Actions.reset_movements(game) # Resets velocity for all players
            
            if(time >= min_delta_time): # Fps
                GE.display(game, screen, draw_options)
                time -= min_delta_time
                
            clock.tick(fps) # Force the loop to trigger at a certain pace
            Utils.checkIfGoal(game, initGame)
            Utils.checkPlayersOut(game["players"]) # Checks if players are out of bounds
    return


main()









































