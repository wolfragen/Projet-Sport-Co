# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:55:36 2025

@author: quent
"""

import torch
from torchrl.data import (
    ReplayBuffer,
    LazyTensorStorage,
    PrioritizedSampler,
)
from tensordict import TensorDict
import numpy as np
from dataclasses import dataclass
import math
import time
from collections import deque
import pandas as pd

from AI.Network import DeepRLNetwork
from Graphics.GraphicEngine import startDisplay
from Engine.Environment import LearningEnvironment

class DQNAgent:
    def __init__(self, dimensions, batch_size, lr, sync_rate, buffer_size, epsilon_decay, linear_decay=True, 
                 epsilon=1.0, epsilon_min=0.05, gamma=0.99, betas=(0.9, 0.999), eps=1e-8, soft_update=True, tau=5e-3,
                 random=False, cuda=False):
        self.batch_size = batch_size
        self.action_dim = dimensions[-1]
        self.lr = lr
        self.gamma = gamma
        self.epsilon = float(epsilon)
        self.epsilon_decay = epsilon_decay
        self.linear_decay = linear_decay
        self.epsilon_min = epsilon_min
        self.random = random

        self.device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

        # Buffer
        sampler = PrioritizedSampler(
            max_capacity=buffer_size,
            alpha=0.6,
            beta=0.4,
            eps=eps,
            dtype=torch.float32,
            reduction="mean",
            max_priority_within_buffer=1.0,
        )
        self.memory = ReplayBuffer(
            storage=LazyTensorStorage(buffer_size),
            sampler=sampler,
            pin_memory=False,
            prefetch=0,
        )

        # Networks
        self.onlineNetwork = DeepRLNetwork(dimensions).to(self.device)
        self.targetNetwork = DeepRLNetwork(dimensions).to(self.device)
        self.targetNetwork.eval()

        self.sync_rate = sync_rate
        self.soft_update = soft_update
        self.tau = tau
        
        self.sync_value = 0

        # Optimizer & Loss
        self.optimizer = torch.optim.Adam(self.onlineNetwork.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=1e-5)
        self.loss_function = torch.nn.MSELoss()

    @torch.no_grad()
    def act(self, state, train=True):
        if self.random or (train and np.random.rand() <= self.epsilon):
            if(state[-1]):
                return np.random.randint(self.action_dim)
            else:
                return np.random.randint(self.action_dim-1) #TODO: sans shoot
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.onlineNetwork(state_tensor)
        return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done, td_priority=1.0):
        transition = TensorDict(
            {
                "state": torch.tensor(state, dtype=torch.float32, device=self.device),
                "action": torch.tensor(action, dtype=torch.long, device=self.device),
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
                "next_state": torch.tensor(next_state, dtype=torch.float32, device=self.device),
                "done": torch.tensor(done, dtype=torch.float32, device=self.device),
            },
            batch_size=[]
        )
        self.memory.add(transition)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        states = batch["state"]
        actions = batch["action"].long().unsqueeze(1)
        rewards = batch["reward"].unsqueeze(1)
        next_states = batch["next_state"]
        dones = batch["done"].unsqueeze(1)
        weights = batch.get("weights", torch.ones(self.batch_size, 1, device=self.device))
        indices = batch.get("idx", None)

        # Q-learning target
        predicted_q = self.onlineNetwork(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.targetNetwork(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * (1 - dones) * next_q_values

        td_errors = predicted_q - target_q
        loss = (weights * td_errors.pow(2)).mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineNetwork.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        if indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().abs())

        if self.soft_update:
            self.softUpdateNetwork()
        else:
            # Target network sync
            self.sync_value += 1
            if self.sync_value >= self.sync_rate:
                self.sync_value = 0
                self.syncTargetNetwork()

    def decayEpsilon(self):
        if self.linear_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def syncTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
        self.targetNetwork.eval()
    
    def softUpdateNetwork(self):
        with torch.no_grad():
            for target_param, online_param in zip(self.targetNetwork.parameters(),
                                                  self.onlineNetwork.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * online_param.data)

    def save(self, path):
        torch.save(self.onlineNetwork.state_dict(), path)

    def load(self, path):
        self.onlineNetwork.load_state_dict(torch.load(path, map_location=self.device))
        self.syncTargetNetwork()
        self.random = False


def getRandomDQNAgents(n, dimensions, batch_size=128, lr=3e-4, sync_rate=1000, buffer_size=50_000, 
                       epsilon_decay=0.99995, linear_decay=True, epsilon=1.0, epsilon_min=0.05, gamma=0.99, soft_update=True, tau=5e-3, 
                       betas=(0.9, 0.999), eps=1e-8, cuda=False):
    agents = []
    for i in range(n):
        agents.append(DQNAgent(dimensions=dimensions, batch_size=batch_size, lr=lr, sync_rate=sync_rate, buffer_size=buffer_size, 
                               epsilon_decay=epsilon_decay, linear_decay=linear_decay, epsilon=epsilon, epsilon_min=epsilon_min, gamma=gamma, soft_update=soft_update, tau=tau, 
                               betas=betas, eps=eps, random=True, cuda=cuda))
    return agents


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    actions: list[list[int]]
    steps: int
    score: tuple[int,int]
    success: bool
    display: bool

def trainingGame(players_number, agents, scoring_function, reward_coeff_dict, max_steps, training_progression, train=True, 
                 display=False, simulation_speed=1.0, screen=None, draw_options=None, gather_data=False):
    n_players = players_number[0] + players_number[1]
    
    env = LearningEnvironment(players_number=players_number, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                              training_progression=training_progression, display=display, simulation_speed=simulation_speed, screen=screen, 
                              draw_options=draw_options, human=False)
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

def dqn_train(players_number, agents, scoring_function, reward_coeff_dict, num_episodes, save_folder, wait_rate=0.1, exploration_rate=0.8, 
          starting_max_steps=100, ending_max_steps=1000, display=False, simulation_speed=1.0, moyenne_ratio=0.1, end_test=True):
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
    
    reward_history = []
    step_history = []
    score_left_history = []
    score_right_history = []
    done_history = []
    epsilon_history = []
    
    min_fail_percent = 1.0
    fail_percent_history = []
    reward_history_test = []
    step_history_test = []
    
    print("Starting to gather data")
    
    while(len(agents[0].memory) < agents[0].batch_size*50):
        result = trainingGame(players_number=players_number, agents=agents, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                              max_steps=max_steps, training_progression=0, 
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
        
        result = trainingGame(players_number=players_number, agents=agents, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                              max_steps=max_steps, training_progression=min(1,(episode+1)/(num_episodes*exploration_rate)), 
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
            
            reward_mean = np.mean(moyenne_reward)
            step_mean = np.mean(moyenne_step)
            epsilon = agents[0].epsilon
            score_left_mean = np.mean(moyenne_score_left)
            score_right_mean = np.mean(moyenne_score_right)
            done_mean = np.mean(moyenne_done)
            
            reward_history.append(reward_mean)
            step_history.append(step_mean)
            epsilon_history.append(epsilon)
            score_left_history.append(score_left_mean)
            score_right_history.append(score_right_mean)
            done_history.append(done_mean)
            
            print(f"Episode {episode+1} | Reward: {reward_mean:.2f} | Steps: {step_mean:.1f} | epsilon={epsilon:.2f} | Score: {score_left_mean:.2f} - {score_right_mean:.2f} | Win: {done_mean:.2f} | {speed:.1f} eps/s | {bar} | {progress*100:6.2f}%")
            if(agents[0].epsilon == agents[0].epsilon_min or agents[0].epsilon <= 0.2):
                r, s, s_left, s_right, fail_percent = runTests(players_number=players_number, agents=agents, 
                                                               scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                                                               max_steps=max_steps, nb_tests=100, should_print=False)
                fail_percent_history.append(fail_percent)
                reward_history_test.append(r)
                step_history_test.append(s)
                
                if(fail_percent < min_fail_percent and (min_fail_percent-fail_percent >= 0.01 or fail_percent < min_fail_percent*4/5)):
                    # Enlever la partie aléatoire : on regarde sur 1000 tests si on a eu un bon résultat.
                    
                    r, s, s_left, s_right, fail_percent = runTests(players_number=players_number, agents=agents, 
                                                                   scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                                                                   max_steps=max_steps, nb_tests=1000, should_print=False)
                    if(fail_percent < min_fail_percent):
                        min_fail_percent = fail_percent
                        for agent_id in range(len(agents)):
                            agent.onlineNetwork.save(save_folder + f"{agent_id}_best")
                                
                        if(fail_percent < 0.005):
                            print("Early stopping, agent fully trained !")
                            break
                        
            else:
                fail_percent_history.append(None)
                reward_history_test.append(None)
                step_history_test.append(None)
    
    if(save_folder != None):
        for agent_id in range(len(agents)):
            agent.onlineNetwork.save(save_folder + f"{agent_id}")
            agent.load(save_folder + f"{agent_id}_best")
            
        df = pd.DataFrame({
            "reward": reward_history,
            "steps": step_history,
            "epsilon": epsilon_history,
            "score_left": score_left_history,
            "score_right": score_right_history,
            "done": done_history,
            "fail_percent": fail_percent_history,
            "reward_test": reward_history_test,
            "step_test": step_history_test,
        })
        df.to_csv(save_folder + "training_data.csv", index=False)
    
    if(end_test):
        print()
        print("="*100)
        print()
        print(" "*45 + "Testing..." + " "*45)
        print()
        print("="*100)
        print()
        
        runTests(players_number=players_number, agents=agents, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                 max_steps=max_steps)
    
    return reward_history, step_history, epsilon_history, score_left_history, score_right_history, done_history, fail_percent_history, reward_history_test, step_history_test




def testingGame(players_number, agents, scoring_function, reward_coeff_dict, max_steps, training_progression):
    n_players = players_number[0] + players_number[1]
    
    env = LearningEnvironment(players_number=players_number, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                              training_progression=training_progression, display=False, human=False)
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

def runTests(players_number, agents, scoring_function, reward_coeff_dict, max_steps, training_progression=1.0, nb_tests=10_000, should_print=True):
    
    rewards = 0
    steps = 0
    nb_fail = 0
    score_left = 0
    score_right = 0
    
    for episode in range(nb_tests):
        
        result = testingGame(players_number=players_number, agents=agents, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                             max_steps=max_steps, training_progression=training_progression)
        rewards += result.total_reward
        steps += result.steps
        score = result.score
        score_left += score[0]
        score_right += score[1]
        
        if(score[0] != 1):
            nb_fail += 1
        
        if((episode+1)%(nb_tests/10) == 0 and should_print):
            print(f"Tests en cours: {(episode+1)/nb_tests*100}%")
    
    print(f"{nb_tests} tests | Reward: {rewards/nb_tests:.2f} | Steps: {steps/nb_tests:.1f} | Score: {score_left/nb_tests:.2f} / {score_right/nb_tests:.2f} | failed: {nb_fail/nb_tests:.3f}")
    return rewards/nb_tests, steps/nb_tests, score_left/nb_tests, score_right/nb_tests, nb_fail/nb_tests







































