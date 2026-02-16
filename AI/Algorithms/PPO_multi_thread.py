# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 09:50:19 2026

@author: konra
"""

from multiprocessing import Process, Queue, cpu_count
import torch
import copy
import random
import time
from collections import deque
import os
import re

from Engine.Environment import LearningEnvironment
from AI.Algorithms.DQN import runTests
from AI.Algorithms.PPO import PPOAgent
from AI.Algorithms.RANDOM import RandomAgent
from logger import Logger


def load_previous_models(model: PPOAgent,
               load_path: str, 
               actors: list[PPOAgent],
               max_pool_size: int,
               logger: Logger | None = None):
    
    assert load_path is not None
    # Get all files in directory
    files = os.listdir(load_path)
    
    # Extract actor checkpoints with their index
    actor_pattern = re.compile(r"actor_(\d+)\.pt")

    for f in files:
        match = actor_pattern.match(f)
        if match:
            idx = int(match.group(1))
            actors.append((idx, f))
    
    assert len(actors) > 0, "No actor checkpoints found."
    
    # Sort actors by index
    actors.sort(key=lambda x: x[0])
    
    # -----------------------------
    # Load latest actor
    # -----------------------------
    last_actor_idx, last_actor_file = actors[-1]
    last_actor_path = os.path.join(load_path, last_actor_file)
    
    # -----------------------------
    # Load critic
    # -----------------------------
    critic_path = os.path.join(load_path, "critic.pt")
    assert os.path.exists(critic_path), "critic.pt not found."
    model.load(actor_path=last_actor_path, critic=True, critic_path=critic_path)
    
    # -----------------------------
    # Load past max_pool_size actors (excluding latest)
    # -----------------------------
    previous_actors = actors[:-1]  # remove latest
    pool_candidates = previous_actors[-max_pool_size:]  # take last max_pool_size
    
    opponent_pool = []
    for idx, filename in pool_candidates:
        path = os.path.join(load_path, filename)
        opponent_pool.append(torch.load(path))

    txt_1 = f"Loaded latest actor: actor_{last_actor_idx}.pt"
    txt_2 = f"Loaded {len(opponent_pool)} previous actors for pool"
    
    if logger is not None:
        logger.log(txt_1)
        logger.log(txt_2)

    print(txt_1)
    print(txt_2)

def _ppo_competitive_worker(
    model: PPOAgent,
    opponent: PPOAgent,
    mean_steps: int,
    per_worker_rollout: int,
    max_steps_per_game: int,
    draw_penalty: float,
    queue_out : deque,
):
    torch.set_num_threads(1) # obligatoire, un seul thread utilisé ici.

    if opponent is None : 
        env = LearningEnvironment(
            players_number=(1, 0),
            scoring_function=model.scoring_function,
            reward_coeff_dict=model.reward_coeff_dict,
            mean_steps = mean_steps,
            human=False,
            phantom_player=True,
        )
    else:
        env = LearningEnvironment(
            players_number=(1, 1),
            scoring_function=model.scoring_function,
            reward_coeff_dict=model.reward_coeff_dict,
            mean_steps = mean_steps,
            human=False
        )

    state = env.getState(0)
    step_in_game = 0

    transitions = []
    stats = {
        "games_played": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "score_0": 0,
        "score_1": 0,
        "current_reward": 0.0,
        "total_steps": 0,
    }

    for rollout in range(per_worker_rollout):

        state_t = torch.as_tensor(
                state, dtype=torch.float32, device=model.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_0, logprob = model.actor.act(state)
            value = model.critic.net(state_t).squeeze(-1).item()
            if opponent is not None : action_1 = opponent.act(env.getState(1))

        env.playerAct(0, action_0)
        if opponent is not None : env.playerAct(1, action_1)

        rewards = env.step()
        reward = rewards[0]

        step_in_game += 1

        done_env = env.isDone()
        timeout = step_in_game >= max_steps_per_game
        rollout_timeout = (rollout == model.rollout_size-1) and not timeout
        done_ppo = done_env or timeout or rollout_timeout

        if timeout:
            reward += draw_penalty

        stats["current_reward"] += reward

        transitions.append((state, logprob, done_ppo, value, action_0, reward))

        if done_ppo:

            if not rollout_timeout:
                # ---- bookkeeping
                stats["total_steps"] += step_in_game
                stats["games_played"] += 1
    
                if env.score[0] > env.score[1]:
                    stats["wins"] += 1
                elif env.score[0] < env.score[1]:
                    stats["losses"] += 1
                else:
                    stats["losses"] += 1
    
                stats["score_0"] += env.score[0]
                stats["score_1"] += env.score[1]

            env.reset()
            step_in_game = 0

        state = env.getState(0)

    queue_out.put((transitions, stats))


def clone_opponent(source):
    opp = copy.deepcopy(source)
    opp.actor.eval()
    for p in opp.actor.parameters():
        p.requires_grad = False
    return opp
    
    
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
    save_all = False,
    log = False,
    load_existing=False,
    load_path=None,
    n_workers: int = -1,
):

    if n_workers == -1:
        n_workers = max(1, cpu_count()-1)

        # -------------------------
    # Opponent pool utilities
    # -------------------------
    opponent_pool = []

    # Initial opponent (episode 0 snapshot)
    opponent_pool.append(clone_opponent(model))

    # Random agent for evaluation
    random_agent = RandomAgent(action_dim=4)

    # Logger
    if log:
        logger = Logger(os.path.join(save_path, f"log.txt"))
        logger.log("Parameters:")
        model.log_params(logger)
        logger.log(text = f"opponent_save_interval: {opponent_save_interval}")
        logger.log(text = f"max_pool_size: {max_pool_size}")
        logger.log(text = f"draw_penalty: {draw_penalty}")
        logger.log(text = f"max_steps_per_game: {max_steps_per_game}")
        logger.log(text = "********************")

    if(load_existing):
        actors = []
        load_previous_models(model, load_path, actors, max_pool_size, logger)

    txt = f"Starting PPO self-play with opponent pool ({num_episodes} episodes on {n_workers} threads)"
    print(txt)
    if log: logger.log(text = txt)
    start_time = time.time()

    mean_steps = deque(maxlen=100)
    games_played_for_mean_steps = deque(maxlen=100)
    total_models = 0

    current_reward = 0.0
    score_0 = score_1 = 0
    games_played = 0
    wins = losses = draws = 0
    total_steps = 0

    for episode in range(1, num_episodes + 1):

        if time.time() - start_time > max_duration:
            txt = "Max training time reached."
            print(txt)
            if log: logger.log(text = txt)
            break

        # ---- sample opponent
        r = random.random()
        if r < 0.1:
            opponent = model                        # mirror, 10%
        elif r < 0.7:
            opponent = opponent_pool[-1]       # most recent, 60%
        elif r < 1.1:
            opponent = random.choice(opponent_pool) # random, 20% -> TODO 30% pour l'instant
        else:
            opponent = None                   

        q = Queue()
        workers = []

        per_worker_rollout = model.rollout_size

        step_avg = sum(mean_steps)/len(mean_steps) if len(mean_steps) != 0 else max_steps_per_game

        for _ in range(n_workers):
            p = Process(
                target=_ppo_competitive_worker,
                args=(
                    copy.deepcopy(model),
                    clone_opponent(opponent) if opponent else None,
                    step_avg,
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

            games_played += stats["games_played"]
            games_played_for_mean_steps.append(stats["games_played"])
            wins += stats["wins"]
            losses += stats["losses"]
            draws += stats["draws"]
            score_0 += stats["score_0"]
            score_1 += stats["score_1"]
            current_reward += stats["current_reward"]
            total_steps += stats["total_steps"]
            mean_steps.append(stats["total_steps"])

        for p in workers:
            p.join()

        model.replay(last_value=0.0, last_done=True)
        model.init_memory()

        # ---- save opponent snapshot
        if episode % opponent_save_interval == 0:
            opponent_pool.append(clone_opponent(model))
            if len(opponent_pool) > max_pool_size:
                opponent_pool.pop(0)
            
            # Save model for safety issue
            model.save(actor_path = save_path+f"actor_{total_models}.pt", critic=True, critic_path= save_path+"critic.pt")
            total_models += 1

        # -------------------------
        # Training diagnostics
        # -------------------------
        if episode % interval_notify == 0:
            avg_reward = current_reward / games_played if games_played > 0 else 0
            avg_steps = total_steps / games_played if games_played > 0 else 0
            win_rate = wins / games_played if games_played > 0 else 0

            mean_steps_over_100_games = sum([steps*num_games for steps, num_games in zip(mean_steps, games_played_for_mean_steps)])/sum(games_played_for_mean_steps)

            txt = f"""[{int(time.time()-start_time)}s] Ep {episode} | Games {games_played} | W/D/L {wins}/{draws}/{losses} ({win_rate:.2f}) | Avg steps {avg_steps:.1f} ({mean_steps_over_100_games:.1f})| Score {score_0/games_played:.2f}-{score_1/games_played:.2f} | Reward {avg_reward:.4f} | Pool {len(opponent_pool)}"""
            print(txt)
            if log: logger.log(text = txt)

            # reset stats
            current_reward = 0.0
            score_0 = score_1 = 0
            games_played = 0
            wins = losses = draws = 0
            total_steps = 0

        # -------------------------
        # Evaluation vs random agent
        # -------------------------
        if episode % eval_interval == 0:

            txt = ">>> Evaluating vs random agent..."
            print(txt)
            if log: logger.log(text = txt)

            runTests(
                players_number=(1, 1),
                agents=[model, random_agent],
                scoring_function=model.scoring_function,
                reward_coeff_dict=model.reward_coeff_dict,
                max_steps=max_steps_per_game,
                training_progression=1.0,
                nb_tests=100,
                should_print=False,
                logger= logger if log else None
            )

    txt = "Saving final model..."
    print(txt)
    if log: logger.log(text = txt)

    model.save(os.path.join(save_path, f"actor_{total_models}.pt"), 
                    critic=save_all, 
                    critic_path=os.path.join(save_path, f"critic.pt") if save_all else None)

    txt = "Self-play training finished."
    print(txt)
    if log: logger.log(text = txt)
    logger.close()