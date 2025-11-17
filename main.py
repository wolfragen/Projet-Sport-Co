# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import math
import pandas as pd
import os

import Settings
from Play import humanGame, debugGame, trainingGame, train, runTests
from AI.Algorithms.DQN import getRandomDQNAgents
from AI.Rewards.Reward import computeReward

        
def save_training_parameters(csv_path, **params):
    """
    Saves training parameters to a CSV using pandas.
    Automatically handles:
    - tuples/lists → stored as strings
    - functions → stored by name
    """

    processed = {}
    for key, value in params.items():

        # Save functions by their name
        if callable(value):
            processed[key] = value.__name__

        # Convert tuples/lists to string
        elif isinstance(value, (tuple, list)):
            processed[key] = str(value)

        else:
            processed[key] = value

    # Convert to DataFrame with two columns
    df = pd.DataFrame(
        {"parameter": list(processed.keys()),
         "value": list(processed.values())}
    )

    # Save
    df.to_csv(csv_path, index=False)
    print(f"Parameters saved to: {csv_path}")

if(__name__ == "__main__"):
    
    players_number = (1,0)

    dimensions = (Settings.ENTRY_NEURONS, 2**6, 2**5, 2**4, 3)
    sync_rate = 1000
    batch_size = 512
    lr = 1e-4
    gamma = 0.99
    buffer_size = 100_000

    epsilon = 0.8
    epsilon_min = 0.1

    num_episodes = 5_000
    wait_rate = 0
    exploration_rate = 0.5 - wait_rate # à x%, on atteint le min d'epsilon, en incluant le temps "stagnant"
    num_wait = round(num_episodes*wait_rate) # number of episodes to wait until epsilon decay

    starting_max_steps = 250
    ending_max_steps = 250

    display = False
    simulation_speed = 10.0

    #epsilon_decay = math.exp(math.log(epsilon_min/epsilon)/((num_episodes-num_wait)*exploration_rate)) # computed decay to get 0.05 at the end of exploration phase.
    epsilon_decay = (epsilon - epsilon_min) / ((num_episodes-num_wait)*exploration_rate)
    linear_decay = True

    scoring_function = computeReward

    cuda = False #torch.cuda.is_available()

    n_players = players_number[0] + players_number[1]
    agents = getRandomDQNAgents(n=n_players, dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                           epsilon_decay=epsilon_decay, linear_decay=True, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma, cuda=cuda)



    agents[0].random = False
    # agents[0].load("C:/.Ingé/Projet-Sport-Co-Networks/score=0.23")
    # debugGame(players_number, agents, human=False)"""
    
    
    save_folder = "C:/.ingé/Projet-Sport-Co-Networks/"
    
    kwargs = dict(
        players_number=players_number,
        sync_rate=sync_rate,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        num_episodes=num_episodes,
        wait_rate=wait_rate,
        exploration_rate=exploration_rate,
        num_wait=num_wait,
        starting_max_steps=starting_max_steps,
        ending_max_steps=ending_max_steps,
        epsilon_decay=epsilon_decay,
        linear_decay=linear_decay,
        scoring_function=scoring_function
    )
    save_training_parameters(save_folder + "training_parameters.csv", **kwargs)

    train(players_number, agents, num_episodes, scoring_function, save_folder, 
          wait_rate=wait_rate, exploration_rate=exploration_rate, 
          starting_max_steps=starting_max_steps, ending_max_steps=ending_max_steps, 
          display=display, simulation_speed=simulation_speed, moyenne_ratio=0.05)
    
    """
    agents[0].load("C:/.Ingé/Projet-Sport-Co-Networks/Suivi_test_step=27.3_fail=0.28")
    #debugGame(players_number, agents)
    
    
    runTests(players_number=players_number, agents=agents, scoring_function=scoring_function, max_steps=ending_max_steps, nb_tests=10_000)
    """




































