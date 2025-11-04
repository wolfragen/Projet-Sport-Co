# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import math

import Settings
from Play import humanGame, debugGame, trainingGame, train, runTests
from AI.Algorithms.DQN import getRandomDQNAgents
from AI.Rewards.Reward import computeReward

if(__name__ == "__main__"):
    
    players_number = (1,0)
    
    dimensions = (Settings.ENTRY_NEURONS, 256, 128, 128, 5)
    sync_rate = 1000
    batch_size = 128
    lr = 1e-4
    gamma = 0.995
    buffer_size = 100_000
    
    epsilon = 1.0
    epsilon_min = 0.1
    
    num_episodes = 20000
    wait_rate = 0.05
    exploration_rate = 0.8 - wait_rate # à 80%, on atteint le min d'epsilon, en incluant le temps "stagnant"
    num_wait = round(num_episodes*wait_rate) # number of episodes to wait until epsilon decay
    
    starting_max_steps = 40
    ending_max_steps = 40
    
    display = False
    simulation_speed = 10.0
    
    #epsilon_decay = math.exp(math.log(epsilon_min/epsilon)/((num_episodes-num_wait)*exploration_rate)) # computed decay to get 0.05 at the end of exploration phase.
    epsilon_decay = (epsilon - epsilon_min) / ((num_episodes-num_wait)*exploration_rate)
    linear_decay = True
    
    scoring_function = computeReward
    
    n_players = players_number[0] + players_number[1]
    agents = getRandomDQNAgents(n=n_players, dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                           epsilon_decay=epsilon_decay, linear_decay=True, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma)
    
    
    
    agents[0].random = False
    #agents[0].load("C:/.Ingé/Projet-Sport-Co-Networks/Suivi_test_step=27.3_fail=0.28")
    

    save_folder = "C:/.ingé/Projet-Sport-Co-Networks/"
    
    train(players_number, agents, num_episodes, scoring_function, save_folder, 
          wait_rate=wait_rate, exploration_rate=exploration_rate, 
          starting_max_steps=starting_max_steps, ending_max_steps=ending_max_steps, 
          display=display, simulation_speed=simulation_speed, moyenne_ratio=0.05)

    
    #agents[0].load("C:/.Ingé/Projet-Sport-Co-Networks/Suivi_test_step=27.3_fail=0.28")
    #debugGame(players_number, agents)
    
    
    #runTests(players_number=players_number, agents=agents, scoring_function=scoring_function, max_steps=ending_max_steps, nb_tests=10_000)





































