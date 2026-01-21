# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 20:17:10 2025

@author: quent
"""

import pygame

from Graphics.GraphicEngine import startDisplay
from Engine.Environment import LearningEnvironment


def humanGame(players_number, agents, scoring_function, reward_coeff_dict):
    n_players = players_number[0] + players_number[1]
    assert len(agents) == n_players
    
    screen, draw_options = startDisplay()
    env = LearningEnvironment(players_number=players_number, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                              display=True, screen=screen, draw_options=draw_options, human=True)
    
    while(not env.isDone() and env.display):
        human_player = env.selected_player
        
        for player_id in range(n_players):
            agent = agents[player_id]
            if player_id != human_player:
                state = env.getState(player_id)
                action = agent.act(state, train=False)
                env.playerAct(player_id, action)
                    
        env.step()
    
    env._endDisplay()
    return


def debugGame(players_number, agents, scoring_function, reward_coeff_dict, max_steps=1000, human = False):
    n_players = players_number[0] + players_number[1]
    assert len(agents) == n_players
    
    screen, draw_options = startDisplay()
    env = LearningEnvironment(players_number=players_number, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                              display=True, screen=screen, draw_options=draw_options, human=human)
    step = 1
    
    states = [None for _ in range(n_players)]
    actions = [None for _ in range(n_players)]
    rewards = [None for _ in range(n_players)]
    
    while(not env.isDone() and env.display and step<max_steps+1):
        
        human_player = env.selected_player
        for player_id in range(n_players):
            agent = agents[player_id]
            state = env.getState(player_id)
            action = None
            
            if(player_id != human_player or not human):
                action = agent.act(state, train=False)
                env.playerAct(player_id, action)
            else:
                temp_continue = False
                while(not temp_continue):
                    temp_continue, action = env._processHumanEvents()
                
                
            states[player_id] = state
            actions[player_id] = action
            
        if(not human):
            temp_continue = False
            while not temp_continue:
                event = pygame.event.wait()
                keys = pygame.key.get_pressed()
                
                if event.type == pygame.QUIT:
                    env._endDisplay()
                    return
                
                elif keys[pygame.K_SPACE]:
                    temp_continue = True
        
        rewards = env.step(debug=True)
        
        print("="*100)
        print()
        print(" "*47 + f"{step=}")
        print()
        print("="*100)
        print()
        
        for player_id in range(n_players):
            state = states[player_id]
            action = actions[player_id]
            reward = rewards[player_id]
            reward_dict = env.last_reward_components[player_id]

            print(f"{player_id=} | {action=} | reward_total={reward:.4f}")
            
            if reward_dict:
                for k, v in reward_dict.items():
                    print(f"    {k:<30}: {v:.4f}")
            print()

            
        step += 1
    
    env._endDisplay()
    return


























