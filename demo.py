# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 10:39:02 2026

@author: konra
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", module="pygame.pkgdata")
warnings.filterwarnings("ignore", message="Your system is avx2 capable")
import pygame

import math
import pandas as pd

import Settings
from Play import humanGame, debugGame
from AI.Algorithms.DQN import getRandomDQNAgents, dqn_train, runTests
from AI.Algorithms.NEAT import neat_train
from AI.Rewards.Reward import computeReward


if(__name__ == "__main__"):
    
    players_number = (1,0)

    dimensions = (Settings.ENTRY_NEURONS, 2**7, 2**7, 2**6, 3)
    batch_size = 128
    lr = 1e-5
    gamma = 0.99
    buffer_size = 100_000

    epsilon = 0.8
    epsilon_min = 0.05

    num_episodes = 10_000
    wait_rate = 0
    exploration_rate = 0.5 - wait_rate # à x%, on atteint le min d'epsilon, en incluant le temps "stagnant"
    num_wait = round(num_episodes*wait_rate) # number of episodes to wait until epsilon decays

    starting_max_steps = 250
    ending_max_steps = 250
    
    soft_update = True
    sync_rate = 1000
    tau = 5e-3

    display = False
    simulation_speed = 10.0

    #epsilon_decay = math.exp(math.log(epsilon_min/epsilon)/((num_episodes-num_wait)*exploration_rate)) # computed decay to get 0.05 at the end of exploration phase.
    epsilon_decay = (epsilon - epsilon_min) / ((num_episodes-num_wait)*exploration_rate)
    linear_decay = True

    scoring_function = computeReward
    reward_coeff_dict = {
        "static_reward": -0.002,
        "delta_ball_player_coeff": 0.01,
        "delta_ball_goal_coeff": 0.02,
        "can_shoot_coeff": 0.1,
        "goal_coeff": 5,
        "wrong_goal_coeff": -1
        }

    cuda = False #torch.cuda.is_available()

    n_players = players_number[0] + players_number[1]
    
    """
    agents = getRandomDQNAgents(n=n_players, dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                           epsilon_decay=epsilon_decay, linear_decay=True, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma, 
                           soft_update=soft_update, tau=tau, cuda=cuda)

    Settings.GOAL_LEN = 499
    Settings.ENTRY_NEURONS = 8
    agents[0].load("C:/.ingé/Projet-Sport-Co-Networks/fail=0.008/0_best")
    debugGame(players_number, agents, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, human=False)"""
    
    
    """
    config_file = "C:/.ingé/Projet-Sport-Co/AI/Algorithms/config_feed-forward_neat.cfg"
    neat_train(config_file=config_file, players_number=players_number, generations=100, n_eval=20, max_steps=250,
                   scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict)"""
    

    Settings.GOAL_LEN = 250
    Settings.ENTRY_NEURONS = 9
    dimensions = (Settings.ENTRY_NEURONS, 2**8, 2**7, 2**6, 4)
    players_number = (1,1)
    n_players = players_number[0] + players_number[1]
    agents = getRandomDQNAgents(n=n_players, dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                           epsilon_decay=epsilon_decay, linear_decay=True, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma, 
                           soft_update=soft_update, tau=tau, cuda=cuda)
    
    agents[1].load("C:/.ingé/Projet-Sport-Co-Networks/shooting_AI/0_84_shoot")
    debugGame(players_number, agents, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, human=True, max_steps=10000)
    
                           



































