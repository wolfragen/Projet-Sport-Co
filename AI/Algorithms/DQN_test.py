# -*- coding: utf-8 -*-

import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import math
from dataclasses import dataclass

class DeepRLNetwork(nn.Module):
    """
    Feedforward network for Deep RL.

    Parameters
    ----------
    dimensions : List[int]
        List of integers representing the number of neurons in each layer.
    """
    def __init__(self, dimensions: list[int], activation=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

        # optional: initialize weights explicitly
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

class DQNAgent:
    def __init__(self, dimensions, batch_size, lr, gamma, 
                 epsilon, epsilon_min, epsilon_decay, buffer_size):
        self.batch_size = batch_size
        self.action_dim = dimensions[-1]
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.memory = deque(maxlen=buffer_size)
        
        self.onlineNetwork = DeepRLNetwork(dimensions)
        self.targetNetwork = DeepRLNetwork(dimensions)
        
        self.optimizer = optim.Adam(self.onlineNetwork.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        self.loss_function = torch.nn.MSELoss()

    def act(self, state, train):
        if train and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        
        q_values = self.onlineNetwork(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    
    def replay(self, batch_size=None):
        if( batch_size is None): batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        
        ## convert to torch tensors ##
        observations = torch.tensor(np.array(observations), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_observations = torch.tensor(next_observations, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        ##############################
        
        predicted_q = self.onlineNetwork(observations).gather(1, actions)
        with torch.no_grad():
            next_q_values, _ = self.targetNetwork(next_observations).max(dim=1, keepdim=True)
            target_q = rewards + self.gamma * (1 - dones) * next_q_values

        self.loss = self.loss_function(predicted_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def decayEpsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        
    def syncTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
        self.targetNetwork.eval() # no gradient for target network, it's not "learning"
        
class AngleEnvironment():
    def __init__(self):
        self.ballAngle = 0
        self.playerAngle = 0
        self.THRESHOLD = 3
        
        self.reset()
        
    def reset(self):
        self.ballAngle = random.randint(0,359)-180 # random between -180 and 180
        self.playerAngle = 0
        
        while(self.isDone()):
            self.ballAngle = random.randint(0,359)-180 # random between -180 and 180
            self.playerAngle = 0
        
    def step(self, action):
        delta = 0
        match action:
            case 0:
                delta = -5
            case 2:
                delta = 5
            case _:
                delta = 0
        self.playerAngle += delta
        
        while(self.playerAngle > 180): self.playerAngle -= 360
        while(self.playerAngle < -180): self.playerAngle += 360
        
    def getState(self):
        relative = self.getRelative()
        sign = 1
        if(relative<0):
            sign = -1
        return [abs(relative)/180, sign] # normalized [-1,1]
    
    def getRelative(self):
        relative = self.ballAngle - self.playerAngle
        while(relative > 180): relative -=360
        while(relative < -180): relative +=360
        return relative # [-180,180]
    
    def getReward(self, state, action):
        angle = abs(self.ballAngle - self.playerAngle)
        if(angle > 180): angle = 360 - angle
        
        if(angle <= self.THRESHOLD):
            return 0
        
        return -0.1
    
    def isDone(self):
        angle = abs(self.ballAngle - self.playerAngle)
        if(angle > 180): angle = 360 - angle
        
        return angle <= self.THRESHOLD
    
    def getBallAngle(self):
        return self.ballAngle
    
    def getplayerAngle(self):
        return self.playerAngle
    
    def toString(self):
        string = "Environment(" + "ballAngle=" + self.ballAngle + ", playerAngle=" + self.playerAngle + ", relative=" + self.getRelative() + ")"
        return string
    
def custom_train():
    batch_size = 64
    num_episodes = 3000
    num_wait = round(num_episodes*10/100) # number of episodes to wait until epsilon decay
    max_step = 100
    
    lr = 3e-4
    gamma = 0.99
    exploration_rate = 0.5
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = math.exp(math.log(epsilon_min/epsilon)/((num_episodes-num_wait)*exploration_rate)) # computed decay to get 0.05 at the end of exploration phase.
    
    sync_rate = 20
    
    nb_moyenne = round(num_episodes/10)
    
    moyenne_reward = deque(maxlen=nb_moyenne)
    moyenne_step = deque(maxlen=nb_moyenne)
    probability_actions = deque(maxlen=nb_moyenne)
    
    agent = DQNAgent((2,3), batch_size=batch_size, lr=lr, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, 
                     epsilon_decay=epsilon_decay, buffer_size=8_000)
    
    start = time.time()
    
    @dataclass(frozen=True)
    class EpisodeResult:
        total_reward: float
        steps: int
        sum_actions: int
        probability_actions: list[float]
        
    
    def runEpisode(agent, train=True):
        env = AngleEnvironment()
        total_reward = 0
        step = 0
        sum_actions = 0
        nb_actions = np.zeros(3)
        
        while(not env.isDone() and step < max_step):
            state = env.getState()
            action = agent.act(state, train)
            
            nb_actions[action] += 1 # keep count of each action taken
            sum_actions += action -1 # not used anymore, i'll delete it later
            env.step(action)
            
            reward = env.getReward(state, action)
            next_state = env.getState()
            
            agent.remember(state, action, reward, next_state, env.isDone()) # Store data
            agent.replay() # Fit
            
            total_reward += reward
            step += 1
            
        probability_actions = np.zeros(3)
        for i in range(3):
            probability_actions[i] = nb_actions[i] / step
                
        return EpisodeResult(total_reward, step, sum_actions, probability_actions)
    
    def calculMoyenneAction(actions):
        # actions : list([nb_action_0, nb_action_1, nb_action_2])
        nb_actions = np.zeros(3)
        for arr in actions:
            for i in range(3):
                nb_actions[i] += arr[i]
        
        steps = len(actions)
        for i in range(3):
            nb_actions[i] /= steps
        
        return nb_actions # Avg of each action taken
    
    def runTests(agent, nb_tests=1000):
        
        rewards = 0
        steps = 0
        nb_actions = np.zeros(3)
        nb_fail = 0
        
        for episode in range(nb_tests):
            
            result = runEpisode(agent, train=False)
            rewards += result.total_reward
            steps += result.steps
            nb_action = result.probability_actions
            for i in range(3):
                nb_actions[i] += nb_action[i]
            
            if(result.steps == max_step):
                nb_fail += 1
            
            if((episode+1)%(nb_tests/10) == 0):
                print(f"Tests en cours: {(episode+1)/nb_tests*100}%")
        
        print(f"{nb_tests} tests | Reward: {rewards/nb_tests:.2f} | Steps: {steps/nb_tests:.1f} | PActions: {nb_actions[0]/nb_tests:.2f} / {nb_actions[1]/nb_tests:.2f} / {nb_actions[2]/nb_tests:.2f} | failed: {nb_fail}")
        return
    
    for episode in range(num_episodes):
        
        result = runEpisode(agent)
        
        total_reward = result.total_reward
        steps = result.steps
        
        probability_actions.append(result.probability_actions)
        moyenne_reward.append(total_reward)
        moyenne_step.append(steps)
        
        if((episode+1) % sync_rate == 0):
            agent.syncTargetNetwork()
        
        if((episode+1) > num_wait):
            agent.decayEpsilon()
        
        if((episode+1) % 10 == 0):
            # Progress Bar
            bar_length = 40
            progress = (episode + 1) / num_episodes
            filled = int(progress * bar_length)
            bar = "█" * filled + " " * (bar_length - filled)
            
            elapsed = time.time() - start
            speed = (episode+1)/elapsed
            
            moyenne_proba_action = calculMoyenneAction(probability_actions)
            
            print(f"""Episode {episode+1} | Reward: {np.mean(moyenne_reward):.2f} | Steps: {np.mean(moyenne_step):.1f} | PActions: {moyenne_proba_action[0]:.2f} / {moyenne_proba_action[1]:.2f} / {moyenne_proba_action[2]:.2f} | epsilon={agent.epsilon:.2f} | {speed:.1f} eps/s | {bar} | {progress*100:6.2f}%""")
    
    runTests(agent)
    agent.onlineNetwork.save("C:/.ingé/Projet-Sport-Co-Network_testAngle")
    return
    
    
if __name__ == "__main__":
    
    custom_train()