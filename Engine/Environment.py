# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:15:11 2025

@author: quent
"""

import numpy as np
import pygame
from random import randint

from Settings import Settings
from Graphics.GraphicEngine import display, startDisplay
from Engine.Utils import checkIfGoal, checkPlayersOut, checkPlayersCanShoot
from Engine.Physics import NumbaEngine
from Engine.Vision import getVision, getGlobalVision
from Engine.Actions import reset_movements
from AI.AIActions import play
from Player.PlayerActions import process_events

class LearningEnvironment():
    def __init__(self, players_number: list[int,int], scoring_function, reward_coeff_dict, mean_steps=2000, training_progression=0.0,
        display: bool = False, simulation_speed: float = 1.0, screen=None, draw_options=None, human=False, phantom_player = False):

        self.done = False
        self.human = human

        self.players_number = players_number
        self.n_players = players_number[0] + players_number[1]
        self.previous_actions = [-1 for i in range(self.n_players)]

        self.training_progression = training_progression
        self.scoring_function = scoring_function
        self.reward_coeff_dict = reward_coeff_dict
        self.mean_steps = mean_steps

        self.display = display
        self.screen = screen
        self.draw_options = draw_options
        self.phantom_player = None
        if(self.n_players == 1 and Settings.COMPETITIVE_VISION): # TODO phantom_player is True...
            self.phantom_player = {"position_x": Settings.SCREEN_OFFSET + randint(Settings.PLAYER_LEN, Settings.DIM_X-Settings.PLAYER_LEN),
                               "position_y": Settings.SCREEN_OFFSET + randint(Settings.PLAYER_LEN, Settings.DIM_Y-Settings.PLAYER_LEN)}

        self._init_game()
        if(display):
            self._initDisplay(simulation_speed)


    def reset(self, soft=False):
        if(soft):
            self._init_game(score = self.score)
        else:
            self._init_game()
        self.done = False

    def reset_after_goal(self):
        """Reset positions after a goal — preserve scores for multi-goal episodes."""
        self._init_game(score=self.score)
        self.done = False

    def step(self, human_events = True, debug=False):

        self.previous_score = self.score.copy()
        self.engine.snapshot_previous_positions()

        dt = Settings.DELTA_TIME
        self.engine.step(dt, Settings.DELTA_SIMU)

        temp_id = self.engine.get_last_player_shoot()
        self._checkIfDone()
        self.engine.set_last_player_shoot(temp_id)

        reset_movements(self.engine)
        checkPlayersCanShoot(self.engine)
        rewards = [self.getReward(player_id, debug) for player_id in range(self.n_players)]

        checkPlayersOut(self.engine) # check for players out of bound

        if self.display:
            self._tickDisplay()

        return rewards

    def playerAct(self, player_id, action):
        self.previous_actions[player_id] = action
        return play(self.engine, player_id, action)

    def getState(self, player_id):
        return getVision(self.engine, player_id, self.left_goal_position, self.right_goal_position, phantom_player=self.phantom_player)

    def getGlobalState(self):
        return getGlobalVision(self.engine, self.left_goal_position, self.right_goal_position, phantom_player=self.phantom_player)

    def getReward(self, player_id, debug=False):
        player_state = self.engine.get_player_state(player_id)
        ball_state = self.engine.get_ball_state()
        action = self.previous_actions[player_id]
        return self.scoring_function(self.reward_coeff_dict, player_state, self.players_number, action, ball_state, self.left_goal_position,
                                       self.right_goal_position, self.score, self.previous_score, self.mean_steps,
                                       self.training_progression, debug)

    def isDone(self):
        return self.done

    def fast_step(self, actions):
        """
        Mega-kernel fast path: apply actions + physics + rewards + vision in one Numba call.
        actions: list of int actions per player.
        Returns: (visions, global_vision, rewards, done)
          - visions: np.array (n_players, ENTRY_NEURONS) — next states for all players
          - global_vision: np.array (global_vision_size,) — critic state
          - rewards: np.array (n_players,) — rewards
          - done: bool
        """
        visions, global_vision, rewards, done = self.engine.full_step(actions, mean_steps=self.mean_steps)
        self.done = done
        if done and self.human:
            self.reset(soft=True)
        if self.display:
            self._tickDisplay()
        return visions, global_vision, rewards, done



    def _init_game(self, score : np.array = None):
        if(score is not None):
            self.score = score
        else:
            self.score = np.zeros(2)
            self.previous_score = np.zeros(2)

        if Settings.PHYSICS_ENGINE == "numba":
            engine = NumbaEngine()
        else:
            from Engine.Physics.PymunkEngine import PymunkEngine
            engine = PymunkEngine()
        engine.create_world()
        self.left_goal_position, self.right_goal_position = engine.build_board()
        engine.build_ball()
        self.selected_player = engine.build_players(self.players_number, self.human, self.phantom_player)
        self.engine = engine

        engine.snapshot_previous_positions()
        checkPlayersCanShoot(engine)

        # Setup for full_step fast path (NumbaEngine only)
        if Settings.PHYSICS_ENGINE == "numba":
            engine.set_reward_coeffs(self.reward_coeff_dict)
            engine.set_score(self.score)

    def _initDisplay(self, simulation_speed):
        if(self.screen is None):
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
        self.time = time

    def _display(self):
        display(self.engine, self.score, self.screen, self.draw_options)

    def _checkIfDone(self):
        self.done = checkIfGoal(self.engine, self.score)
        if(self.done and self.human):
            self.reset(soft=True)

    def _processHumanEvents(self):
        should_stop, action = process_events()
        if(should_stop):
            self._endDisplay()
            return True, -1
        if(action != -1 and self.selected_player != None):
            self.playerAct(self.selected_player, action)
            return True, action
        return False, -1
