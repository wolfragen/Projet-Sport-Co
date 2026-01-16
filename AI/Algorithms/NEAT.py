# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 09:18:08 2025

@author: quent
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
from functools import partial
import neat
from neat.parallel import ParallelEvaluator
import numpy as np
import pickle  # pour sauvegarder les meilleurs génomes

from Engine.Environment import LearningEnvironment

class NeatAgent:
    def __init__(self, genome, config,
                 allow_backward=False):
        """
        NEAT agent for a 2D football player.
        
        allow_backward: if True -> forward ∈ [-1, 1]
                        if False -> forward ∈ [0, 1]
        """
        self.genome = genome
        self.config = config
        self.allow_backward = allow_backward

        # Create neural network from genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def act(self, observation):
        # NEAT expects a Python list
        inp = list(map(float, observation))

        raw_output = self.net.activate(inp)
        # raw_output is a list of 3 numbers in [-∞, +∞]
        
        # Unpack raw outputs
        forward_raw = raw_output[0]
        turn_raw    = raw_output[1]
        kick_raw    = raw_output[2]

        # ---- Normalize outputs ----

        turn = np.tanh(turn_raw)  # maps to [-1, 1]

        if self.allow_backward:
            forward = np.tanh(forward_raw)  # [-1, 1]
        else:
            forward = (np.tanh(forward_raw) + 1) / 2  # [0, 1]

        kick_power = 1 / (1 + np.exp(-kick_raw))  # sigmoid => [0, 1]

        return np.array([forward, turn, kick_power])
    
    
def neat_eval_solo(genome, config, players_number, n_eval,
                    scoring_function, reward_coeff_dict, max_steps):
    agent = NeatAgent(genome, config, allow_backward=False)
    total_reward = 0
    
    for _ in range(n_eval):
        env = LearningEnvironment(players_number=(1,0),
                                  scoring_function=scoring_function,
                                  reward_coeff_dict=reward_coeff_dict,
                                  training_progression=1)
        temp_reward = 0
        obs = env.getState(0)
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.act(obs)
            env.playerAct(0, action)
            rewards = env.step()
            temp_reward += rewards[0]
            obs = env.getState(0)
            done = env.isDone()
            step += 1
        
        total_reward += temp_reward
    return total_reward/n_eval
        
        
def neat_train(config_file, players_number, generations, n_eval, max_steps,
               scoring_function, reward_coeff_dict):

    # Charger la config NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Créer la population
    p = neat.Population(config)

    # Ajouter reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Variables dynamiques (mutables dans la fonction)
    dynamic = {
        "n_eval": n_eval,
        "max_steps": max_steps
    }

    # -- Fonction d'évaluation utilisée par NEAT --
    def eval_genomes(genomes, config):

        # Ajuster dynamiquement les paramètres selon la génération
        current_generation = p.generation
        """
        if current_generation < 20:
            dynamic["n_eval"] = 3
            dynamic["max_steps"] = 150
        elif current_generation < 50:
            dynamic["n_eval"] = 5
            dynamic["max_steps"] = 200
        else:
            dynamic["n_eval"] = 10
            dynamic["max_steps"] = 250"""

        # Construire l'évaluateur parallèle pour cette génération
        eval_func = partial(
            neat_eval_solo,
            players_number=(1, 0),
            n_eval=dynamic["n_eval"],
            scoring_function=scoring_function,
            reward_coeff_dict=reward_coeff_dict,
            max_steps=dynamic["max_steps"]
        )

        pe = ParallelEvaluator(num_workers=16, eval_function=eval_func)

        # Évaluer tous les génomes (PAS une génération complète)
        return pe.evaluate(genomes, config)

    # Exécuter TOUT l'entraînement d'un coup
    winner = p.run(eval_genomes, n=generations)

    # Sauvegarder le meilleur génome
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("Entraînement terminé ! Meilleur génome sauvegardé dans 'best_genome.pkl'.")
    return winner, stats
    
    
    
    
    
    
    
    
    
    