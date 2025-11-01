# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:15:11 2025

@author: quent
"""

from typing import Callable
import numpy as np
import pygame

import Settings
from Graphics.GraphicEngine import display, startDisplay
from Engine.Utils import checkIfGoal, createSpace, checkPlayersOut
from Engine.Entity.Board import buildBoard
from Engine.Entity.Ball import buildBall
from Engine.Entity.Player import buildPlayers
from Engine.Vision import getVision
from Engine.Actions import reset_movements
from AI.AIActions import play
from Player.PlayerActions import process_events

class LearningEnvironment():
    def __init__(self, players_number: list[int,int], scoring_function: Callable, training_progression=0.0,
        display: bool = False, simulation_speed: float = 1.0, screen=None, draw_options=None, human=False):
        
        self.done = False
        self.human = human
        
        self.players_number = players_number
        self.n_players = players_number[0] + players_number[1]
        
        self.scoring_function = scoring_function
        self.training_progression = training_progression
        
        self.display = display
        self.screen = screen
        self.draw_options = draw_options
        
        self._init_game()
        if(display):
            self._initDisplay(simulation_speed)
            
            
    
    def reset(self):
        self._initGame()
        
    def step(self):
        
        space = self.space
        for _ in range(Settings.DELTA_TIME):
            space.step(0.001)

        reset_movements(self.players)
        self._checkIfDone()
        
        if self.display:
            self._tickDisplay()
            self._processHumanEvents()
        
        rewards = [self.getReward(player_id) for player_id in range(self.n_players)]
        checkPlayersOut(self.players) # check for players out of bound
        return rewards
    
    def playerAct(self, player_id, action):
        player = self.players[player_id]
        return play(player, self.ball, action)
        
    def getState(self, player_id):
        player = self.players[player_id]
        return getVision(self.space, player, self.ball, self.left_goal_position, self.right_goal_position)
    
    def getReward(self, player_id):
        player = self.players[player_id]
        return self.scoring_function(player, self.ball, self.left_goal_position, self.right_goal_position, self.score, self.training_progression)
    
    def isDone(self):
        return self.done
    
    
    
    def _init_game(self, score : np.array = None):
        self.score = np.zeros(2)
        if(score != None):
            self.score = score
        space = createSpace()
        
        self.left_goal_position, self.right_goal_position = buildBoard(space) # Creates static objects
        self.ball = buildBall(space) # Creates the ball
        self.players, self.players_left, self.players_right, self.selected_player = buildPlayers(space, self.players_number, self.human) # Creates the players
        
        self.space = space
        
    def _initDisplay(self, simulation_speed):
        if(self.screen == None or self.draw_options == None):
            self.screen, self.draw_options = startDisplay()
        
        self.clock = pygame.time.Clock()
        self.delta_time = Settings.DELTA_TIME
        self.fps = int(1000/self.delta_time)* simulation_speed
        self.min_delta_time = 1000/Settings.MAX_FPS
        self.time = 0
        
        self._display()
        
    def _endDisplay(self):
        self.display = False
        pygame.display.quit()
        pygame.quit()
        
    def _tickDisplay(self):
        time = self.time
        
        time += self.delta_time
        if time >= self.min_delta_time:
            self._display()
            time -= self.min_delta_time
        self.clock.tick(self.fps)
        
    def _display(self):
        display(self.space, self.players, self.score, self.screen, self.draw_options)
    
    def _checkIfDone(self):
        self.done = checkIfGoal(self.ball, self.score)
        if(self.done and self.human):
            self.done = False
            self._init_game(self.score)
        
    def _processHumanEvents(self):
        should_stop, action = process_events()
        if(should_stop):
            return self._endDisplay()
        if(action != -1 and self.selected_player != None):
            return self.playerAct(self.selected_player, action)
    
    
    
    
    
    
    
    
    
    