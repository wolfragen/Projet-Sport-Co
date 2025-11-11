# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 20:17:10 2025

@author: quent
"""

import pygame
from dataclasses import dataclass
import math
import time
from collections import deque
import numpy as np

import Settings
from Graphics.GraphicEngine import startDisplay
from Engine.Environment import LearningEnvironment
from AI.Rewards.Reward import computeReward
from Player.PlayerActions import process_events


def humanGame(players_number, agents):
    n_players = players_number[0] + players_number[1]
    assert len(agents) == n_players
    
    screen, draw_options = startDisplay()
    env = LearningEnvironment(players_number=players_number, scoring_function=computeReward,
                              display=True, screen=screen, draw_options=draw_options, 
                              human=True)
    
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


def debugGame(players_number, agents, max_steps=1000, human = False):
    n_players = players_number[0] + players_number[1]
    assert len(agents) == n_players
    
    screen, draw_options = startDisplay()
    env = LearningEnvironment(players_number=players_number, scoring_function=computeReward,
                              display=True, screen=screen, draw_options=draw_options, 
                              human=human)
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
        
        rewards = env.step()
        
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
            print(f"{player_id=} | {state=} | {action=} | {reward=:.2f}")
            print()
            
        step += 1
    
    env._endDisplay()
    return


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    actions: list[list[int]]
    steps: int
    score: tuple[int,int]
    success: bool
    display: bool

def trainingGame(players_number, agents, scoring_function, max_steps, training_progression, train=True, 
                 display=False, simulation_speed=1.0, screen=None, draw_options=None, gather_data=False):
    n_players = players_number[0] + players_number[1]
    
    env = LearningEnvironment(players_number=players_number, scoring_function=scoring_function, training_progression=training_progression,
                              display=display, simulation_speed=simulation_speed, screen=screen, draw_options=draw_options,
                              human=False)
    step = 1
    
    states = [env.getState(player_id) for player_id in range(n_players)]
    done = False
    
    total_reward = 0
    actions = [None for _ in range(n_players)]
    
    while(not done):
        
        for player_id in range(n_players):
            agent = agents[player_id]
            state = states[player_id]
            action = agent.act(state, train=train)
            env.playerAct(player_id, action)
                
            states[player_id] = state
            actions[player_id] = action
        
        rewards = env.step()
        done = (env.isDone() or step>=max_steps)
        
        for player_id in range(n_players):
            state = states[player_id]
            next_state = env.getState(player_id)
            action = actions[player_id]
            reward = rewards[player_id]
            
            if(train):
                agent.remember(state, action, reward, next_state, done)
                if(not gather_data): # on entraine pas sur les données de départ.
                    agent.replay()
            
            states[player_id] = next_state
            total_reward += reward
            
        step += 1
    return EpisodeResult(total_reward=total_reward, actions=None, steps=step-1, score=env.score, success=env.isDone(), display=env.display)

def train(players_number, agents, num_episodes, scoring_function, save_folder, wait_rate=0.1, exploration_rate=0.8, 
          starting_max_steps=100, ending_max_steps=1000, display=False, simulation_speed=1.0, moyenne_ratio=0.1):
    assert len(agents) == players_number[0] + players_number[1]
    if(save_folder != None and save_folder[-1] != "/"):
        save_folder += "/"
    
    num_wait = round(num_episodes*wait_rate)
    max_steps_decay = math.exp(math.log(ending_max_steps/starting_max_steps)/((num_episodes-num_wait)*exploration_rate))
    
    max_steps = starting_max_steps
    
    screen, draw_options = None,None
    if(display):
        screen, draw_options = startDisplay()
    
    nb_moyenne = round(num_episodes*moyenne_ratio)
    moyenne_reward = deque(maxlen=nb_moyenne)
    moyenne_step = deque(maxlen=nb_moyenne)
    moyenne_score_left = deque(maxlen=nb_moyenne)
    moyenne_score_right = deque(maxlen=nb_moyenne)
    moyenne_done = deque(maxlen=nb_moyenne)
    
    print("Starting to gather data")
    
    while(len(agents[0].memory) < agents[0].batch_size*50):
        result = trainingGame(players_number=players_number, agents=agents, max_steps=max_steps, training_progression=0, scoring_function=scoring_function, 
                              display=display, simulation_speed=simulation_speed, screen=screen, draw_options=draw_options, gather_data=True)
        score = result.score
        display = result.display
        
        moyenne_reward.append(result.total_reward)
        moyenne_step.append(result.steps)
        moyenne_score_left.append(score[0])
        moyenne_score_right.append(score[1])
        moyenne_done.append(result.success)
    
    print("Starting training")
    start = time.time()
    
    for episode in range(num_episodes):
        
        result = trainingGame(players_number=players_number, agents=agents, max_steps=max_steps, training_progression=min(1,(episode+1)/(num_episodes*exploration_rate)), scoring_function=scoring_function, 
                              display=display, simulation_speed=simulation_speed, screen=screen, draw_options=draw_options, gather_data=False)
        
        score = result.score
        display = result.display
        
        moyenne_reward.append(result.total_reward)
        moyenne_step.append(result.steps)
        moyenne_score_left.append(score[0])
        moyenne_score_right.append(score[1])
        moyenne_done.append(result.success)
        
        if((episode+1) > num_wait):
            max_steps = min(ending_max_steps, max_steps*max_steps_decay)
            for agent in agents:
                agent.decayEpsilon()
        
        if((episode+1) % 100 == 0):
            # Progress Bar
            bar_length = 40
            progress = (episode + 1) / num_episodes
            filled = int(progress * bar_length)
            bar = "█" * filled + " " * (bar_length - filled)
            
            elapsed = time.time() - start
            speed = (episode+1)/elapsed
            
            print(f"Episode {episode+1} | Reward: {np.mean(moyenne_reward):.2f} | Steps: {np.mean(moyenne_step):.1f} | epsilon={agents[0].epsilon:.2f} | Score: {np.mean(moyenne_score_left):.2f} - {np.mean(moyenne_score_right):.2f} | Win: {np.mean(moyenne_done):.2f} | {speed:.1f} eps/s | {bar} | {progress*100:6.2f}%")
            if(agents[0].epsilon == agents[0].epsilon_min):
                runTests(players_number, agents, scoring_function, max_steps, nb_tests=100, should_print=False)
    
    if(save_folder != None):
        for agent_id in range(len(agents)):
            agent.onlineNetwork.save(save_folder + f"{agent_id}")
            
    print()
    print("="*100)
    print()
    print(" "*45 + "Testing..." + " "*45)
    print()
    print("="*100)
    print()
    
    runTests(players_number=players_number, agents=agents, scoring_function=scoring_function, max_steps=max_steps)
    return




def testingGame(players_number, agents, scoring_function, max_steps, training_progression):
    n_players = players_number[0] + players_number[1]
    
    env = LearningEnvironment(players_number=players_number, scoring_function=scoring_function, training_progression=training_progression,
                              display=False, human=False)
    step = 1
    
    states = [env.getState(player_id) for player_id in range(n_players)]
    done = False
    
    total_reward = 0
    actions = [[] for _ in range(n_players)]
    
    while(not done):
        
        for player_id in range(n_players):
            agent = agents[player_id]
            state = states[player_id]
            action = agent.act(state, train=False)
            env.playerAct(player_id, action)
                
            states[player_id] = state
            actions[player_id].append(action)
        
        rewards = env.step()
        done = (env.isDone() or step>=max_steps)
        
        for player_id in range(n_players):
            state = states[player_id]
            next_state = env.getState(player_id)
            action = actions[player_id][step-1]
            reward = rewards[player_id]
            
            states[player_id] = next_state
            total_reward += reward
            
        step += 1
    return EpisodeResult(total_reward=total_reward, actions=actions, steps=step-1, score=env.score, success=env.isDone(), display=env.display)

def runTests(players_number, agents, scoring_function, max_steps, training_progression=1.0, nb_tests=10_000, should_print=True):
    
    rewards = 0
    steps = 0
    nb_fail = 0
    score_left = 0
    score_right = 0
    
    for episode in range(nb_tests):
        
        result = testingGame(players_number=players_number, agents=agents, scoring_function=scoring_function, max_steps=max_steps, training_progression=training_progression)
        rewards += result.total_reward
        steps += result.steps
        score = result.score
        score_left += score[0]
        score_right += score[1]
        
        if(not result.success):
            nb_fail += 1
        
        if((episode+1)%(nb_tests/10) == 0 and should_print):
            print(f"Tests en cours: {(episode+1)/nb_tests*100}%")
    
    print(f"{nb_tests} tests | Reward: {rewards/nb_tests:.2f} | Steps: {steps/nb_tests:.1f} | Score: {score_left/nb_tests:.2f} / {score_right/nb_tests:.2f} | failed: {nb_fail/nb_tests:.2f}")
    return























