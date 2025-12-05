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

    dimensions = (Settings.ENTRY_NEURONS, 2**8, 2**7, 2**6, 4)
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
    agents = getRandomDQNAgents(n=n_players, dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                           epsilon_decay=epsilon_decay, linear_decay=True, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma, 
                           soft_update=soft_update, tau=tau, cuda=cuda)



    agents[0].random = False
    # agents[0].load("C:/.Ingé/Projet-Sport-Co-Networks/score=0.23")
    debugGame(players_number, agents, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, human=True)
    
    
    save_folder = "C:/.ingé/Projet-Sport-Co-Networks/"
    
    kwargs = dict(
        players_number=players_number,
        soft_update=soft_update,
        tau=tau,
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
        scoring_function=scoring_function,
        reward_coeff_dict=reward_coeff_dict
    )
    
    
    save_training_parameters(save_folder + "training_parameters.csv", **kwargs)
    
    """
    dqn_train(players_number, agents, scoring_function, reward_coeff_dict, num_episodes, save_folder, 
          wait_rate=wait_rate, exploration_rate=exploration_rate, 
          starting_max_steps=starting_max_steps, ending_max_steps=ending_max_steps, 
          display=display, simulation_speed=simulation_speed, moyenne_ratio=0.05)"""
    
    """
    config_file = "C:/.ingé/EI2/Projet-Sport-Co/AI/Algorithms/config_feed-forward_neat.cfg"
    neat_train(config_file=config_file, players_number=players_number, generations=100, n_eval=20, max_steps=250,
                   scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict)"""
    
    """
    import cProfile
    import pstats
    
    prof = cProfile.Profile()
    prof.run("train(players_number, agents, num_episodes, save_folder, wait_rate=wait_rate, exploration_rate=exploration_rate, starting_max_steps=starting_max_steps, ending_max_steps=ending_max_steps, display=display, simulation_speed=simulation_speed, moyenne_ratio=0.05, end_test=False)")
    prof.dump_stats('output.prof')
    
    with open(save_folder + 'profiled.txt', 'w') as stream:
        stats = pstats.Stats('output.prof', stream=stream)
        stats.sort_stats('cumtime')
        stats.print_stats()
    """
    
    """
    agents[0].load("C:/.Ingé/Projet-Sport-Co-Networks/fail=0.017/0_best")
    debugGame(players_number, agents)
    
    #runTests(players_number=players_number, agents=agents, max_steps=ending_max_steps, nb_tests=10_000)
    """
                           



































