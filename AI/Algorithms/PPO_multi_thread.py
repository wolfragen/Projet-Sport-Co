# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 09:50:19 2026

@author: konra
"""

############ PIRE QUE LE MONO-THREAD ############

from multiprocessing import Process, Queue, cpu_count
import torch
import copy
import random
import time
from collections import deque
import os

from Engine.Environment import LearningEnvironment
from AI.Algorithms.DQN import runTests
from AI.Algorithms.PPO import PPOAgent
from AI.Algorithms.RANDOM import RandomAgent

def _ppo_competitive_worker(
    model,
    opponent,
    config,
    rollout_size,
    max_steps_per_game,
    draw_penalty,
    queue_out,
):
    import torch # recommandé visiblement... paramètres locaux pour torch.
    torch.set_num_threads(1) # obligatoire, un seul thread utilisé ici.

    env = LearningEnvironment(
        players_number=(1, 1) if opponent else (1, 0),
        scoring_function=config["scoring_function"],
        reward_coeff_dict=config["reward_coeff_dict"],
        mean_steps=config["mean_steps"],
        human=False,
    )

    state = env.getState(0)
    step_in_game = 0

    transitions = []
    stats = {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "score_0": 0,
        "score_1": 0,
        "total_steps": 0,
    }

    for rollout in range(rollout_size):

        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_0, logprob = model.actor.act(state)
            value = model.critic.net(state_t).squeeze(-1).item()
            if opponent:
                action_1 = opponent.act(env.getState(1))

        env.playerAct(0, action_0)
        if opponent:
            env.playerAct(1, action_1)

        rewards = env.step()
        reward = rewards[0]

        step_in_game += 1
        done_env = env.isDone()
        timeout = step_in_game >= max_steps_per_game
        rollout_timeout = (rollout == rollout_size - 1) and not timeout

        done_ppo = done_env or timeout or rollout_timeout
    
        if timeout:
            reward += draw_penalty
    
        transitions.append(
            (state, logprob, done_ppo, value, action_0, reward)
        )
    
        if done_ppo:
            if not rollout_timeout:
                stats["games"] += 1
                stats["total_steps"] += step_in_game
                stats["score_0"] += env.score[0]
                stats["score_1"] += env.score[1]
    
                if env.score[0] > env.score[1]:
                    stats["wins"] += 1
                elif env.score[0] < env.score[1]:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1
    
            env.reset()
            step_in_game = 0
    
        state = env.getState(0)

    queue_out.put((transitions, stats))
    
    
def train_PPO_competitive_parallel(
    model: PPOAgent,
    max_duration: int,
    num_episodes: int,
    save_path: str,
    interval_notify: int = 10,
    opponent_save_interval: int = 50,
    max_pool_size: int = 10,
    draw_penalty: float = -0.5,
    max_steps_per_game: int = 2048,
    eval_interval: int = 500,
    n_workers: int = None,
):

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    opponent_pool = []

    def clone_opponent(source):
        opp = copy.deepcopy(source)
        opp.actor.eval()
        for p in opp.actor.parameters():
            p.requires_grad = False
        return opp

    opponent_pool.append(clone_opponent(model))
    random_agent = RandomAgent(action_dim=4)

    print(f"Starting PPO self-play ({num_episodes} episodes, {n_workers} workers)")
    start_time = time.time()

    current_reward = 0.0
    score_0 = score_1 = 0
    games_played = 0
    wins = losses = draws = 0
    total_steps = 0

    mean_steps = deque(maxlen=100)
    mean_steps.append(max_steps_per_game)

    for episode in range(1, num_episodes + 1):

        if time.time() - start_time > max_duration:
            print("Max training time reached.")
            break

        r = random.random()
        if r < 0.1:
            opponent = model
        elif r < 0.7:
            opponent = opponent_pool[-1]
        elif r < 0.9:
            opponent = random.choice(opponent_pool)
        else:
            opponent = None

        q = Queue()
        workers = []

        per_worker_rollout = model.rollout_size

        for _ in range(n_workers):
            p = Process(
                target=_ppo_competitive_worker,
                args=(
                    copy.deepcopy(model),
                    clone_opponent(opponent) if opponent else None,
                    {
                        "scoring_function": model.scoring_function,
                        "reward_coeff_dict": model.reward_coeff_dict,
                        "mean_steps": sum(mean_steps)/len(mean_steps),
                    },
                    per_worker_rollout,
                    max_steps_per_game,
                    draw_penalty,
                    q,
                ),
            )
            p.start()
            workers.append(p)

        for _ in range(n_workers):
            transitions, stats = q.get()
            for t in transitions:
                model.remember(*t)

            games_played += stats["games"]
            wins += stats["wins"]
            losses += stats["losses"]
            draws += stats["draws"]
            score_0 += stats["score_0"]
            score_1 += stats["score_1"]
            total_steps += stats["total_steps"]

        for p in workers:
            p.join()

        model.replay(last_value=0.0, last_done=True)
        model.init_memory()

        mean_steps.append(
            total_steps / games_played if games_played > 0 else max_steps_per_game
        )

        if episode % opponent_save_interval == 0:
            opponent_pool.append(clone_opponent(model))
            if len(opponent_pool) > max_pool_size:
                opponent_pool.pop(0)

        if episode % interval_notify == 0:
            win_rate = wins / games_played if games_played else 0
            avg_steps = total_steps / games_played if games_played else 0
            avg_reward = current_reward / games_played if games_played else 0

            print(
                f"[{int(time.time()-start_time)}s] Ep {episode} | "
                f"Games {games_played} | "
                f"W/D/L {wins}/{draws}/{losses} "
                f"({win_rate:.2f}) | "
                f"Avg steps {avg_steps:.1f} ({sum(mean_steps)/len(mean_steps):.1f})| "
                f"Score {score_0/games_played:.2f}-{score_1/games_played:.2f} | "
                f"Reward {avg_reward:.4f} | "
                f"Pool {len(opponent_pool)}"
            )

            current_reward = 0
            score_0 = score_1 = 0
            games_played = wins = losses = draws = total_steps = 0

        if episode % eval_interval == 0:
            print(">>> Evaluating vs random agent...")
            runTests(
                players_number=(1, 1),
                agents=[model, random_agent],
                scoring_function=model.scoring_function,
                reward_coeff_dict=model.reward_coeff_dict,
                max_steps=max_steps_per_game,
                training_progression=1.0,
                nb_tests=100,
                should_print=False,
            )

    print("Saving final model...")
    model.save(os.path.join(save_path, "model.pt"))
    print("Self-play training finished.")