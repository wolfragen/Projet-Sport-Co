# --- import --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from Graphics.GraphicEngine import startDisplay
from Engine.Environment import LearningEnvironment

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

#class GaussianPolicy(DeepRLNetwork)


class Actor(nn.Module):

    def __init__(self, dimensions, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        for i in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 3:
                layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)

        self.mean = nn.Linear(dimensions[-2], dimensions[-1])
        self.log_std = nn.Linear(dimensions[-2], dimensions[-1])

    def forward(self, state):
        x = self.net(state)

        mean = self.mean(x)

        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization
        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean = torch.tanh(mean)
        return action, log_prob, mean


    






class SACAgent:


    def __init__(self, dimensions, batch_size, lr, sync_rate, buffer_size, epsilon_decay, linear_decay=True, 
                 epsilon=1.0, epsilon_min=0.05, gamma=0.99, betas=(0.9, 0.999), eps=1e-8, soft_update=True, tau=5e-3, alpha=0.2,
                 random=False, cuda=False):
        
        self.batch_size = batch_size
        self.action_dim = dimensions[-1]
        self.lr = lr
        self.tau = tau
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

        self.critic1 = DeepRLNetwork(dimensions).to(self.device)
        self.critic2 = DeepRLNetwork(dimensions).to(self.device)

        self.critic1_target = DeepRLNetwork(dimensions).to(self.device)
        self.critic2_target = DeepRLNetwork(dimensions).to(self.device)

        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.lr, betas=betas, eps=eps)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr, betas=betas, eps=eps)

        self.actor = Actor(dimensions).to(device)
        self.actor_optimizer() = torch.optim.Adam(self.actor.parameters(), lr=self.lr, betas=betas, eps=eps)

        self.target_entropy = -self.action_dim

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr, betas=betas, eps=eps)
        self.alpha = alpha

        self.action_scale = self.action_dim/2
        self.action_bias = self.action_dim/2

    @torch.no_grad()
    def act(self, state: np.ndarray, eval: bool=False) -> int:
        if self.random:
            return np.random.randint(self.action_dim)
        elif eval:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        
        action = action * self.action_scale + self.action_bias
        return action.detach().cpu().numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        transition = TensorDict(
            {
                "state": torch.tensor(state, dtype=torch.float32, device=self.device),
                "action": torch.tensor(action, dtype=torch.float32, device=self.device),
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
                "next_state": torch.tensor(next_state, dtype=torch.float32, device=self.device),
                "done": torch.tensor(done, dtype=torch.float32, device=self.device),
            },
            batch_size = []
        )
        self.memory.add(transition)
    
    def replay(self):

        batch = self.memory.sample(self.batch_size)
        
        states = batch["state"]
        actions = batch["action"].long().unsqueeze(1)
        rewards = batch["reward"].unsqueeze(1)
        next_states = batch["next_state"]
        dones = batch["done"].unsqueeze(1)
        weights = batch.get("weights", torch.ones(self.batch_size, 1, device=self.device))
        indices = batch.get("idx", None)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_states)
            next_action = next_action * self.action_scale + self.action_bias

            q1_next = self.critic1_target(next_states, next_action)
            q2_next = self.critic2_target(next_states, next_action)

            q_next = torch.min(q1_next, q2_next) - self.alpha* next_log_pi
            target_q = reward + (1-done)*self.gamma * q_next

            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)

            loss_q1 = F.mse_loss(q1, target_q)
            loss_q2 = F.mse_loss(q2, target_q)

            self.critic1_optimizer.zero_grad()
            loss_q1.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            loss_q2.backward()
            self.critic2_optimizer.step()

            pi, log_pi, _ = self.actor.sample(states)
            pi = pi * self.action_scale + self.action_bias

            q1_pi = self.critic1(states, pi)
            q2_pi = self.critic2(states, pi)

            min_q_pi = torch.min(q1_pi, q2_pi)

            policy_loss = (self.alpha * log_pi - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimize.step()

            alpha_loss = -(self.log_alpha* (log_pi+ self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            for target, source in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target.data.copy_(self.tau * source.data+ (1 - self.tau) * target.data)

            for target, source in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target.data.copy_(self.tau * source.data+ (1 - self.tau) * target.data)


def train_SAC(model, scoring_function, reward_coeff_dict,  max_duration, num_episodes, save_path, interval_info):
    env = LearningEnvironment(players_number=(1,0), scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, human=False)
    print(f"Starting training with Soft Actor Critic for {max_duration} seconds and {num_episodes} episodes")

    start = time.time()
    current_reward = 0
    num_game = 0
    score_history1, score_history2 = 0,0

    total_step = 0
    start_step = 10_000

    for i in range(1, num_episodes+1):
        if time.time() - start > max_duration:
            print("Reached max time for training")
            break
        state = env.getState(0)

        for t in range(200):

            with torch.no_grad():
                if total_step < start_step:
                    action = np.random.randint(model.action_dim)
                else:
                    action = model.act(state)

            env.playerAct(0, action)
            reward = env.step()[0]
            current_reward += reward
            done = env.isDone()
            next_state = env.getState(0)

            model.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                score_history1 += env.score[0]
                score_history2 += env.score[1]
                num_game += 1
                env.reset()

            loss = model.replay()


            


            

        
