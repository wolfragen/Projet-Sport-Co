# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import math

import Settings
from Play import humanGame, debugGame, trainingGame, train
from AI.Algorithms.DQN import getRandomDQNAgents
from AI.Rewards.Reward import computeReward

if(__name__ == "__main__"):
    
    players_number = (1,0)
    
    dimensions = (Settings.ENTRY_NEURONS, 128, 128, 4)
    sync_rate = 5
    batch_size = 128
    lr = 3e-4
    gamma = 0.995
    buffer_size = 50_000
    
    epsilon = 1.0
    epsilon_min = 0.05
    
    num_episodes = 3000
    wait_rate = 0.05 * 0
    exploration_rate = 0.5 - wait_rate # à 80%, on atteint le min d'epsilon, en incluant le temps "stagnant"
    num_wait = round(num_episodes*wait_rate) # number of episodes to wait until epsilon decay
    
    starting_max_steps = 20
    ending_max_steps = 20
    
    display = False
    simulation_speed = 10.0
    
    #epsilon_decay = math.exp(math.log(epsilon_min/epsilon)/((num_episodes-num_wait)*exploration_rate)) # computed decay to get 0.05 at the end of exploration phase.
    epsilon_decay = (epsilon - epsilon_min) / ((num_episodes-num_wait)*exploration_rate)
    linear_decay = True
    
    n_players = players_number[0] + players_number[1]
    agents = getRandomDQNAgents(n=n_players, dimensions=dimensions, batch_size=batch_size, lr=lr, buffer_size=buffer_size, 
                           epsilon_decay=epsilon_decay, linear_decay=True, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma)
    
    
    
    """
    agents[0].random = False
    

    save_folder = "C:/.ingé/Projet-Sport-Co-Networks/"
    scoring_function = computeReward
    
    train(players_number, agents, num_episodes, scoring_function, save_folder, sync_rate=sync_rate, 
          wait_rate=wait_rate, exploration_rate=exploration_rate, 
          starting_max_steps=starting_max_steps, ending_max_steps=ending_max_steps, 
          display=display, simulation_speed=simulation_speed, moyenne_ratio=0.05)
    """
    
    agents[0].load("C:/.Ingé/Projet-Sport-Co-Networks/0")
    debugGame(players_number, agents)
    




































