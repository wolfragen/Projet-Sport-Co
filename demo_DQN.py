# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", module="pygame.pkgdata")
warnings.filterwarnings("ignore", message="Your system is avx2 capable")
import torch

from Settings import Settings
from Play import humanGame
from AI.Algorithms.DQN import getRandomDQNAgents
from AI.Rewards.Reward import computeReward

if(__name__ == "__main__"):

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
    cuda = torch.cuda.is_available()
    save_folder = Settings.SAVE_FOLDER
    
    players_number = (1,1)
    Settings.ENTRY_NEURONS=8
    Settings.COMPETITIVE_VISION = False
    Settings.GOAL_LEN = 499
    Settings.ROUND_CORNER = False
                           
    dimensions = (Settings.ENTRY_NEURONS, 2**7, 2**7, 2**6, 3)

    scoring_function = computeReward
    fail_reward = -10
    reward_coeff_dict = {
        "static_reward": -0.0005,
        "delta_ball_player_coeff": 0,
        "delta_ball_goal_coeff": 0.05,
        "can_shoot_coeff": 0.5, # Si le but est passé après un tir allié, sans touche d'un autre joueur
        "goal_coeff": 10,
        "wrong_goal_coeff": fail_reward,
        "has_ball_coeff":0
        }
    
    agent = getRandomDQNAgents(n=1, dimensions=dimensions, batch_size=128, lr=3e-4, sync_rate=1000, buffer_size=50_000, 
                           epsilon_decay=0.99995, linear_decay=True, epsilon=1.0, epsilon_min=0.05, gamma=0.99, soft_update=True, tau=5e-3, 
                           betas=(0.9, 0.999), eps=1e-8, cuda=False)[0]
    
    
    
    
    ##################################################################################################################################################################
    agent.load(save_folder + "fail=0.008/0_best")

    agents = [None, agent]
    players_number = (1,1)
    humanGame(players_number, agents, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict)
    
    ##################################################################################################################################################################
    






























