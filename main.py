# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", module="pygame.pkgdata")
warnings.filterwarnings("ignore", message="Your system is avx2 capable")
import torch
import numba
from Settings import Settings, apply_size_modifier
numba.set_num_threads(Settings.N_THREADS)
from Play import humanGame, debugGame
from AI.Algorithms.DQN import runTests
from AI.Algorithms.PPO import PPOAgent, solo_training, one_v_one_training, team_training
from AI.Algorithms.RANDOM import RandomAgent
from AI.Algorithms.elo_ranking import rank_actors
from AI.Rewards.Reward import computeReward


if(__name__ == "__main__"):

    # ==========================================================================
    # MODE SELECTION: 0 = Solo (1v0), 1 = 1v1, 2 = Team (XvX)
    # ==========================================================================
    MODE = 2

    cuda = torch.cuda.is_available()
    save_folder = Settings.SAVE_FOLDER
    scoring_function = computeReward

    # ======================================================================
    # SOLO TRAINING (1v0)
    # ======================================================================
    if MODE == 0:
        players_number = (1, 0)
        Settings.PLAYERS_NUMBER = players_number
        Settings.ENTRY_NEURONS = 12

        dimensions_actor = (Settings.ENTRY_NEURONS, 64, 64, 32, 4)
        dimensions_critic = (Settings.ENTRY_NEURONS, 64, 64, 32, 1)  # critic = actor state (solo)

        reward_coeff_dict = {
            "static_reward": -0.001,
            "delta_ball_player_coeff": 0.02,
            "delta_ball_goal_coeff": 0.05,
            "can_shoot_coeff": 1,
            "has_ball_coeff": 0.01,
            "goal_coeff": 20,
            "wrong_goal_coeff": -5,
        }

        agent = PPOAgent(
            dimensions=(dimensions_actor, dimensions_critic),
            scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict,
            rollout_size=Settings.N_WORKERS * 64, lr_actor=1e-4, lr_critic=3e-4, n_epoch=4, lr_decay=False,
            clip_eps=0.2, gamma=0.99, lmbda=0.95,
            critic_loss_coeff=0.5, entropy_loss_coeff=0.01, normalize_advantage=False,
            max_grad_norm=1, cuda=cuda,
        )

        # --- Training ---
        solo_training(
            agent, max_duration=3600*2, num_episodes=50000,
            save_path=save_folder+"solo/",
            resume_from=save_folder+"solo/",
            interval_notify=512, max_steps_per_game=2048,
            draw_penalty=-5, eval_interval=1024,
            rolling_size=2048,
        )

        # --- Testing ---
        # agent.load(save_folder + "solo/actor_solo.pt")
        # random_agent = RandomAgent(action_dim=4)
        # runTests(players_number=(1,1), agents=[agent, random_agent], max_steps=2048, nb_tests=1000,
        #          scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict)

        # --- Debug ---
        # debugGame((1,1), [None, agent], scoring_function=scoring_function,
        #           reward_coeff_dict=reward_coeff_dict, human=True, max_steps=10000)

        # --- Human play ---
        # humanGame((1,1), [None, agent], scoring_function=scoring_function,
        #           reward_coeff_dict=reward_coeff_dict)


    # ======================================================================
    # 1v1 TRAINING
    # ======================================================================
    elif MODE == 1:
        players_number = (1, 1)
        Settings.PLAYERS_NUMBER = players_number
        Settings.ENTRY_NEURONS = 12 + (sum(players_number) - 1) * 2

        dimensions_actor = (Settings.ENTRY_NEURONS, 2**7, 2**6, 2**5, 4)
        dimensions_critic = (Settings.ENTRY_NEURONS, 2**7, 2**6, 2**5, 1)

        reward_coeff_dict = {
            "static_lead_reward": 0.003,
            "static_draw_reward": -0.001,
            "delta_ball_player_coeff": 0.005,
            "delta_ball_goal_coeff": 0.02,
            "can_shoot_coeff": 0.2,
            "goal_coeff": 1,
            "wrong_goal_coeff": -1,
            "has_ball_coeff": 0.01,
        }

        # rollout_size = N_WORKERS * steps_per_env_per_rollout
        # With N_WORKERS=512, use 64 steps/env to maintain meaningful GAE horizon
        # (original design was ~2048/24≈85 steps/env with 24 workers)
        rollout_size_1v1 = Settings.N_WORKERS * 64  # = 32768

        agent = PPOAgent(
            dimensions=(dimensions_actor, dimensions_critic),
            scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict,
            rollout_size=rollout_size_1v1, lr_actor=1e-4, lr_critic=3e-4, n_epoch=4, lr_decay=False,
            clip_eps=0.2, gamma=0.99, lmbda=0.95,
            critic_loss_coeff=0.5, entropy_loss_coeff=0.01, normalize_advantage=True,
            max_grad_norm=1, cuda=cuda,
        )

        random_agent = RandomAgent(action_dim=4)
        dimensions_actor = (Settings.ENTRY_NEURONS, 2**7, 2**6, 2**5, 4)
        best_1v1 = PPOAgent(
            dimensions=(dimensions_actor, dimensions_critic),
            scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict,
            rollout_size=rollout_size_1v1, lr_actor=1e-4, lr_critic=3e-4, n_epoch=4, lr_decay=False,
            clip_eps=0.2, gamma=0.99, lmbda=0.95,
            critic_loss_coeff=0.5, entropy_loss_coeff=0.01, normalize_advantage=True,
            max_grad_norm=1, cuda=cuda,
        )
        best_1v1.load(save_folder + "1v1_perso_best.pt")

        # --- Training ---
        one_v_one_training(
            agent, max_duration=3600*4, num_episodes=100000,
            save_path=save_folder+"1v1/",
            resume_from=save_folder+"1v1/",
            interval_notify=512, opponent_save_interval=1024,
            max_pool_size=20, draw_penalty=0,
            max_steps_per_game=2048, eval_interval=1024,
            rolling_size=4096,
            eval_agent=best_1v1,
        )

        agent.load(save_folder + "1v1/actor_362.pt")

        # --- Testing ---
        """ runTests(players_number=players_number, agents=[agent, best_1v1], max_steps=2048, nb_tests=10000,
                 scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict) """

        # --- Debug ---
        """ debugGame(players_number, [agent, best_1v1], scoring_function=scoring_function,
                  reward_coeff_dict=reward_coeff_dict, human=False, max_steps=10000) """

        # --- Human play ---
        """ humanGame(players_number, [None, agent], scoring_function=scoring_function,
                  reward_coeff_dict=reward_coeff_dict) """

        # --- Elo ranking ---
        """ rank_actors(save_folder + "1v1/", players_number=players_number,
                    dimensions_actor=dimensions_actor, reward_coeff_dict=reward_coeff_dict) """


    # ======================================================================
    # TEAM TRAINING (XvX)
    # ======================================================================
    elif MODE == 2:
        players_number = (2, 2)
        n_left, n_right = players_number
        Settings.PLAYERS_NUMBER = players_number
        Settings.SIZE_MODIFIER = 0.7
        apply_size_modifier()
        n_players = sum(players_number)
        Settings.ENTRY_NEURONS = 12 + (n_players - 1) * 3

        dimensions_actor = (Settings.ENTRY_NEURONS, 128, 64, 64, 32, 4)
        dimensions_critic = (6 + n_players * 5, 128, 128, 64, 64, 1)

        reward_coeff_dict = {
            "static_lead_reward": 0.003,
            "static_draw_reward": -0.001,
            "delta_ball_player_coeff": 0.002,
            "delta_ball_goal_coeff": 0.02,
            "can_shoot_coeff": 0.2,
            "goal_coeff": 1,
            "wrong_goal_coeff": -1,
            "has_ball_coeff": 0.01,
        }

        agent = PPOAgent(
            dimensions=(dimensions_actor, dimensions_critic),
            scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict,
            rollout_size=Settings.N_WORKERS * n_left * 64, lr_actor=1e-4, lr_critic=3e-4, n_epoch=4, lr_decay=False,
            clip_eps=0.2, gamma=0.995, lmbda=0.95,
            critic_loss_coeff=0.5, entropy_loss_coeff=0.01, normalize_advantage=True,
            max_grad_norm=1, cuda=cuda,
        )
        best_2v2 = PPOAgent(
            dimensions=(dimensions_actor, dimensions_critic),
            scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict,
            rollout_size=Settings.N_WORKERS * n_left * 64, lr_actor=1e-4, lr_critic=3e-4, n_epoch=4, lr_decay=False,
            clip_eps=0.2, gamma=0.995, lmbda=0.95,
            critic_loss_coeff=0.5, entropy_loss_coeff=0.01, normalize_advantage=True,
            max_grad_norm=1, cuda=cuda,
        )
        best_2v2.load(save_folder + "team/actor_105.pt")

        # --- Training ---
        """ team_training(
            agent, players_number=players_number,
            max_duration=3600*6, num_episodes=200000,
            save_path=save_folder+"team/",
            resume_from=save_folder+"team/",
            interval_notify=256, opponent_save_interval=1024,
            max_pool_size=20, draw_penalty=0,
            max_steps_per_game=2048, eval_interval=1024,
            rolling_size=4096,
            team_reward_ratio=0.9,
        ) """
        
        random_agent = RandomAgent(action_dim=4)
        agent.load(save_folder + "team/actor_105.pt")

        # --- Testing ---
        """ agents = [agent, agent, best_2v2, best_2v2]
        runTests(players_number=players_number, agents=agents, max_steps=2048, nb_tests=10000,
                 scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict) """

        # --- Debug ---
        agents = [agent, agent, best_2v2, best_2v2]
        debugGame(players_number, agents, scoring_function=scoring_function,
                  reward_coeff_dict=reward_coeff_dict, human=False, max_steps=10000)

        # --- Human play ---
        """ agents = [None, agent, agent, agent]
        humanGame(players_number, agents, scoring_function=scoring_function,
                  reward_coeff_dict=reward_coeff_dict) """
