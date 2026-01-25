# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:04:18 2025

@author: quent
"""

import copy
import os
import random
import time

import numpy as np
import torch
from torch import nn

from AI.Network import DeepRLNetwork
from AI.Algorithms.RANDOM import RandomAgent
from AI.Heuristics import chase_and_shoot
from Engine.Environment import LearningEnvironment
from Play import runTests


class ActorNetwork(DeepRLNetwork):
    """
    Actor network for PPO.
    The last layer uses Softmax() since the network outputs a probability distribution that must sum to 1.
    """

    def __init__(self, dimensions: list[int], device: torch.device, lr: float, lr_decay: bool, lr_decay_t_max: int):
        super().__init__(dimensions=dimensions)
        # Append Softmax for action distribution
        self.net = nn.Sequential(self.net, nn.Softmax(dim=1))
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0)
        if lr_decay:
            self.lr_decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, T_max=max(1, int(lr_decay_t_max))
            )
        else:
            self.lr_decay_scheduler = None

    def act(self, state, log_prob_only=False, train=True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.net(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        if not train:
            return action.item()
        logprob = dist.log_prob(action)
        if log_prob_only:
            return logprob.item()
        return action.item(), logprob.item()


class CriticNetwork(DeepRLNetwork):
    """
    Critic network for PPO.
    """

    def __init__(self, dimensions, device, lr, lr_decay, lr_decay_t_max: int):
        super().__init__(dimensions)
        self.device = device
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0)
        if lr_decay:
            self.lr_decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, T_max=max(1, int(lr_decay_t_max))
            )
        else:
            self.lr_decay_scheduler = None


class PPOAgent:
    """
    PPO agent.
    """

    def __init__(
        self,
        dimensions: tuple[list[int]],
        scoring_function: callable,
        reward_coeff_dict: dict[float],
        rollout_size: int,
        lr_actor: float,
        lr_critic: float,
        n_epoch: int,
        lr_decay: bool = True,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        critic_loss_coeff: float = 0.5,
        entropy_loss_coeff: float = 0.01,
        normalize_advantage: bool = True,
        max_grad_norm: float = 1.0,
        cuda: bool = False,
    ):

        assert dimensions[1][-1] == 1, "Output of the critic network must be 1 dimensionnal"
        assert dimensions[0][0] == dimensions[1][0], "Actor and critic networks should have the same input size"

        self.n_epoch = n_epoch
        self.rollout_size = rollout_size

        self.gamma = gamma
        self.lmbda = lmbda

        self.clip_eps = clip_eps
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.max_grad_norm = max_grad_norm

        self.scoring_function = scoring_function
        self.reward_coeff_dict = reward_coeff_dict

        self.normalize_advantage = normalize_advantage
        self.lr_decay = lr_decay
        self.device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")
        self.init_memory()

        # Networks
        self.actor = ActorNetwork(
            dimensions=dimensions[0],
            device=self.device,
            lr=lr_actor,
            lr_decay=lr_decay,
            lr_decay_t_max=n_epoch,
        )
        self.critic = CriticNetwork(
            dimensions=dimensions[1],
            device=self.device,
            lr=lr_critic,
            lr_decay=lr_decay,
            lr_decay_t_max=n_epoch,
        )

    def init_memory(self):
        self.memory = {
            "states": [],
            "log_probs": [],
            "dones": [],
            "vals": [],
            "actions": [],
            "rewards": [],
        }

    def remember(self, states, log_probs, dones, vals, actions, rewards):
        self.memory["states"].append(states)
        self.memory["log_probs"].append(log_probs)
        self.memory["dones"].append(dones)
        self.memory["vals"].append(vals)
        self.memory["actions"].append(actions)
        self.memory["rewards"].append(rewards)

    def evaluate(self, states, action):
        action_probs = self.actor.net(states)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic.net(states)
        return action_logprobs, state_values, dist_entropy

    def compute_gae(self, last_value: float, last_done: bool):
        rewards = self.memory["rewards"]
        values = self.memory["vals"]
        dones = self.memory["dones"]

        advantages = []
        returns = []

        gae = 0.0
        next_value = 0.0 if last_done else float(last_value)

        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[i])
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            gae = delta + self.gamma * self.lmbda * mask * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

            next_value = values[i]

        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def replay(self, last_value: float, last_done: bool):
        """Update the policy using data in memory."""
        old_states = torch.as_tensor(np.array(self.memory["states"]), dtype=torch.float32, device=self.device)
        old_actions = torch.as_tensor(self.memory["actions"], dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(self.memory["log_probs"], dtype=torch.float32, device=self.device)

        advantages, returns = self.compute_gae(last_value=last_value, last_done=last_done)

        loss_hist = {"clip": 0.0, "val": 0.0, "entropy": 0.0}

        for _ in range(self.n_epoch):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            state_values = state_values.squeeze(-1)

            ratios = torch.exp(logprobs - old_logprobs)

            loss1 = ratios * advantages
            loss2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            loss_clip = -torch.min(loss1, loss2)

            loss_val = self.critic_loss_coeff * nn.functional.mse_loss(state_values, returns)
            loss_entropy = -self.entropy_loss_coeff * dist_entropy

            loss = (loss_clip + loss_val + loss_entropy).mean()

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            self.critic.optimizer.step()
            self.actor.optimizer.step()

            if self.lr_decay:
                self.actor.lr_decay_scheduler.step()
                self.critic.lr_decay_scheduler.step()

            loss_hist["clip"] += loss_clip.detach().mean().item()
            loss_hist["val"] += loss_val.detach().mean().item()
            loss_hist["entropy"] += loss_entropy.detach().mean().item()

        return {k: v / self.n_epoch for k, v in loss_hist.items()}

    def act(self, state, train=False):
        return self.actor.act(state, train=False)

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))


def pretrain_actor_imitation(
    model: PPOAgent,
    max_duration: int,
    num_episodes: int,
    max_steps_per_game: int = 256,
    interval_notify: int = 10,
):
    """
    Imitation pretraining: the actor mimics a simple heuristic (chase + shoot).
    """
    env = LearningEnvironment(
        players_number=(1, 0),
        scoring_function=model.scoring_function,
        human=False,
    )

    model.actor.train()
    start = time.time()
    total_loss = 0.0
    total_steps = 0

    for episode in range(1, num_episodes + 1):
        if time.time() - start > max_duration:
            print("Reached max time for imitation pretraining, interrupting")
            break

        env.reset()
        for _ in range(max_steps_per_game):
            state = env.getState(0)
            action = chase_and_shoot(env.players[0], env.ball)

            state_t = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)
            probs = model.actor.net(state_t)
            logp = torch.log(probs + 1e-8)
            loss = nn.functional.nll_loss(logp, torch.as_tensor([action], device=model.device))

            model.actor.optimizer.zero_grad()
            loss.backward()
            model.actor.optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            env.playerAct(0, action)
            env.step()
            if env.isDone():
                break

        if episode % interval_notify == 0 and total_steps > 0:
            avg_loss = total_loss / total_steps
            print(f"[{int(time.time()-start)}s] Imitation Ep {episode} | Avg loss {avg_loss:.4f}")
            total_loss = 0.0
            total_steps = 0


def train_PPO_model(
    model: PPOAgent,
    max_duration: int,
    num_episodes: int,
    save_path: str,
    interval_notify: int = 10,
    max_steps_per_game: int = 2048,
    draw_penalty=-0.5,
):
    """Train a PPO model."""

    env = LearningEnvironment(
        players_number=(1, 0),
        scoring_function=model.scoring_function,
        human=False,
    )
    print(f"Starting training for {max_duration} seconds ({num_episodes} episodes)")
    start = time.time()
    current_reward = 0
    num_game = 0
    score_history_1, score_history_2 = 0, 0

    done = False
    state = env.getState(0)
    step_in_game = 0
    total_steps = 0

    for i_episode in range(1, num_episodes + 1):
        if time.time() - start > max_duration:
            print("Reached max time for training, interrupting")
            break
        loss_hist = {"clip": 0, "val": 0, "entropy": 0}

        for _ in range(model.rollout_size):
            step_in_game += 1
            state_t = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)

            with torch.no_grad():
                action, logprob = model.actor.act(state)
                value = model.critic.net(state_t).squeeze(-1).item()

            env.playerAct(0, action)

            reward = env.step()[0]
            current_reward += reward
            done = env.isDone()

            timeout = step_in_game >= max_steps_per_game
            done_ppo = done or timeout

            if timeout:
                reward += draw_penalty
                current_reward += draw_penalty

            model.remember(state, logprob, done_ppo, value, action, reward)

            if done_ppo:
                score_history_1 += env.score[0]
                score_history_2 += env.score[1]
                num_game += 1

                total_steps += step_in_game
                step_in_game = 0
                env.reset()

            state = env.getState(0)

        with torch.no_grad():
            if done_ppo:
                last_value = 0.0
            else:
                state_t = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)
                last_value = model.critic.net(state_t).squeeze(-1).item()

        loss = model.replay(last_value=last_value, last_done=done)

        loss_hist["clip"] += loss["clip"]
        loss_hist["val"] += loss["val"]
        loss_hist["entropy"] += loss["entropy"]

        model.init_memory()
        if i_episode % interval_notify == 0:
            print(
                f"[{int(time.time()-start)}s] Ep {i_episode} | "
                f"Games {num_game} | "
                f"W/D/L {score_history_1}/{num_game - score_history_1 - score_history_2}/{score_history_2} "
                f"({score_history_1/num_game:.2f}) | "
                f"Avg steps {total_steps/num_game:.1f} | "
                f"Reward {current_reward/num_game:.4f} | "
            )
            num_game = 0
            current_reward = 0
            total_steps = 0
            score_history_1, score_history_2 = 0, 0
            loss_hist = {"clip": 0, "val": 0, "entropy": 0}

    print("Saving network...")
    model.save(save_path + "model.pt")
    print("Training finished")


def train_PPO_competitive(
    model: PPOAgent,
    max_duration: int,
    num_episodes: int,
    max_steps_per_game: int,
    save_path: str,
    interval_notify: int = 10,
    opponent_save_interval: int = 50,
    max_pool_size: int = 10,
    draw_penalty: float = -5.0,
    eval_interval: int = 500,
    warmup_random_episodes: int = 500,
    random_opponent_prob: float = 0.2,
):
    """
    Train a PPO agent using competitive self-play with an opponent pool.
    Includes full training diagnostics and evaluation vs a random agent.
    """

    env = LearningEnvironment(
        players_number=(1, 1),
        scoring_function=model.scoring_function,
        human=False,
    )

    # -------------------------
    # Opponent pool utilities
    # -------------------------
    opponent_pool = []

    def clone_opponent(source: PPOAgent) -> PPOAgent:
        opp = copy.deepcopy(source)
        opp.actor.eval()
        for p in opp.actor.parameters():
            p.requires_grad = False
        return opp

    # Initial opponent (episode 0 snapshot)
    opponent_pool.append(clone_opponent(model))

    # Random agent for evaluation / warmup
    random_agent = RandomAgent(action_dim=4)

    print(f"Starting PPO self-play with opponent pool ({num_episodes} episodes)")
    start_time = time.time()

    # -------------------------
    # Training statistics
    # -------------------------
    current_reward = 0.0
    score_0 = score_1 = 0
    games_played = 0
    wins = losses = draws = 0
    total_steps = 0

    state = env.getState(0)
    step_in_game = 0
    done = False

    # =========================
    # Training loop
    # =========================
    for episode in range(1, num_episodes + 1):

        if time.time() - start_time > max_duration:
            print("Max training time reached.")
            break

        # ---- sample opponent
        if episode <= warmup_random_episodes:
            opponent = random_agent
        else:
            r = random.random()
            if r < random_opponent_prob:
                opponent = random_agent
            elif r < (random_opponent_prob + 0.1):
                opponent = model
            elif r < (random_opponent_prob + 0.8):
                opponent = opponent_pool[-1]
            else:
                opponent = random.choice(opponent_pool)

        # ---- rollout
        for _ in range(model.rollout_size):

            state_t = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)

            with torch.no_grad():
                action_0, logprob = model.actor.act(state)
                value = model.critic.net(state_t).squeeze(-1).item()
                action_1 = opponent.act(env.getState(1))

            env.playerAct(0, action_0)
            env.playerAct(1, action_1)

            rewards = env.step()
            reward = rewards[0]

            current_reward += reward
            step_in_game += 1

            done_env = env.isDone()
            timeout = step_in_game >= max_steps_per_game
            done_ppo = done_env or timeout

            if timeout:
                reward += draw_penalty
                current_reward += draw_penalty

            model.remember(state, logprob, done_ppo, value, action_0, reward)

            if done_ppo:
                total_steps += step_in_game
                games_played += 1

                if env.score[0] > env.score[1]:
                    wins += 1
                elif env.score[0] < env.score[1]:
                    losses += 1
                else:
                    draws += 1

                score_0 += env.score[0]
                score_1 += env.score[1]

                env.reset()
                step_in_game = 0

            state = env.getState(0)

        # ---- PPO update
        with torch.no_grad():
            if done_ppo:
                last_value = 0.0
            else:
                state_t = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)
                last_value = model.critic.net(state_t).squeeze(-1).item()

        model.replay(last_value=last_value, last_done=done_ppo)
        model.init_memory()

        # ---- save opponent snapshot
        if episode % opponent_save_interval == 0:
            opponent_pool.append(clone_opponent(model))
            if len(opponent_pool) > max_pool_size:
                opponent_pool.pop(0)

        # -------------------------
        # Training diagnostics
        # -------------------------
        if episode % interval_notify == 0:
            avg_reward = current_reward / (model.rollout_size * interval_notify)
            avg_steps = total_steps / games_played if games_played > 0 else 0
            win_rate = wins / games_played if games_played > 0 else 0

            print(
                f"[{int(time.time()-start_time)}s] Ep {episode} | "
                f"Games {games_played} | "
                f"W/D/L {wins}/{draws}/{losses} "
                f"({win_rate:.2f}) | "
                f"Avg steps {avg_steps:.1f} | "
                f"Score {score_0/games_played:.2f}-{score_1/games_played:.2f} | "
                f"Reward {avg_reward:.4f} | "
                f"Pool {len(opponent_pool)}"
            )

            current_reward = 0.0
            score_0 = score_1 = 0
            games_played = 0
            wins = losses = draws = 0
            total_steps = 0

        # -------------------------
        # Evaluation vs random agent
        # -------------------------
        if episode % eval_interval == 0:
            print(">>> Evaluating vs random agent...")
            runTests(
                players_number=(1, 1),
                agents=[model, random_agent],
                scoring_function=model.scoring_function,
                max_steps=max_steps_per_game,
                training_progression=1.0,
                nb_tests=100,
                should_print=False,
            )

    print("Saving final model...")
    model.save(os.path.join(save_path, "model.pt"))
    print("Self-play training finished.")
