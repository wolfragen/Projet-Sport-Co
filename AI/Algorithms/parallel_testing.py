# -*- coding: utf-8 -*-
"""
Unified parallel testing infrastructure using persistent ProcessPoolExecutor.
Workers are spawned once and reuse their LearningEnvironment across test batches.
"""

import time
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# --- Worker-side globals ---
_worker_env = None


def _init_test_worker(players_number, scoring_function, reward_coeff_dict, settings_overrides):
    """Called once per worker at pool startup. Creates a reusable environment."""
    global _worker_env
    torch.set_num_threads(1)

    # Apply Settings overrides (workers get fresh defaults on Windows spawn)
    from Settings import Settings
    for key, value in settings_overrides.items():
        setattr(Settings, key, value)

    from Engine.Environment import LearningEnvironment
    _worker_env = LearningEnvironment(
        players_number=players_number,
        scoring_function=scoring_function,
        reward_coeff_dict=reward_coeff_dict,
        display=False, human=False
    )


def _run_test_batch(args):
    """
    Worker function: run n_games test games using fast_step.
    Returns aggregated stats dict.
    """
    agent_proxies, n_games, max_steps, training_progression = args
    env = _worker_env
    n_players = len(agent_proxies)

    total_reward = 0.0
    total_steps = 0
    nb_fail = 0
    nb_not_draw = 0
    score_left = 0.0
    score_right = 0.0

    use_fast = hasattr(env.engine, 'full_step')

    for _ in range(n_games):
        env.reset()

        if use_fast:
            # Fast path: get initial states from one fast_step with no-op actions
            # Actually, get initial states via fast_step's vision output
            # First call getState normally for initial states, then switch to fast_step
            states = [env.getState(pid) for pid in range(n_players)]
        else:
            states = [env.getState(pid) for pid in range(n_players)]

        done = False
        step = 1
        game_reward = 0.0

        while not done:
            actions = [agent_proxies[pid].act(states[pid], train=False) for pid in range(n_players)]

            if use_fast:
                visions, gv, rewards, done_flag = env.fast_step(actions)
                done = done_flag or step >= max_steps
                game_reward += float(rewards.sum())
                # Next states from vision output
                for pid in range(n_players):
                    states[pid] = visions[pid]
            else:
                for pid in range(n_players):
                    env.playerAct(pid, actions[pid])
                rewards = env.step()
                done = env.isDone() or step >= max_steps
                for pid in range(n_players):
                    states[pid] = env.getState(pid)
                    game_reward += rewards[pid]

            step += 1

        score = env.engine._score if use_fast else env.score
        if score[0] + score[1] != 0:
            nb_not_draw += 1
            total_reward += game_reward
            total_steps += step - 1
            score_left += score[0]
            score_right += score[1]

        if score[0] != 1:
            nb_fail += 1

    return {
        "total_reward": total_reward,
        "total_steps": total_steps,
        "nb_fail": nb_fail,
        "nb_not_draw": nb_not_draw,
        "score_left": score_left,
        "score_right": score_right,
        "n_games": n_games,
    }


class TestingPool:
    """Persistent worker pool for parallel testing."""

    def __init__(self, n_workers, players_number, scoring_function, reward_coeff_dict):
        self.n_workers = n_workers

        # Capture current Settings state to propagate to workers
        from Settings import Settings
        settings_overrides = {
            "ENTRY_NEURONS": Settings.ENTRY_NEURONS,
            "PLAYERS_NUMBER": Settings.PLAYERS_NUMBER,
            "PHYSICS_ENGINE": Settings.PHYSICS_ENGINE,
            "COMPETITIVE_VISION": Settings.COMPETITIVE_VISION,
            "NUMBER_OF_RAYS": Settings.NUMBER_OF_RAYS,
            "RANDOM_BALL_POSITION": Settings.RANDOM_BALL_POSITION,
            "RANDOM_PLAYER_INIT": Settings.RANDOM_PLAYER_INIT,
        }

        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_test_worker,
            initargs=(players_number, scoring_function, reward_coeff_dict, settings_overrides)
        )

    def run_tests(self, agents, nb_tests, max_steps, training_progression=1.0, should_print=True):
        """Run nb_tests games across workers. Returns same format as DQN.runTests()."""
        proxies = []
        for a in agents:
            if hasattr(a, 'to_proxy'):
                proxies.append(a.to_proxy())
            else:
                proxies.append(a)

        # Distribute games across workers
        games_per_worker = nb_tests // self.n_workers
        remainder = nb_tests % self.n_workers
        tasks = []
        for i in range(self.n_workers):
            n = games_per_worker + (1 if i < remainder else 0)
            if n > 0:
                tasks.append((proxies, n, max_steps, training_progression))

        start_time = time.time()
        results = list(self._executor.map(_run_test_batch, tasks))

        # Aggregate results
        total_reward = 0.0
        total_steps = 0
        nb_fail = 0
        nb_not_draw = 0
        score_left = 0.0
        score_right = 0.0

        for r in results:
            total_reward += r["total_reward"]
            total_steps += r["total_steps"]
            nb_fail += r["nb_fail"]
            nb_not_draw += r["nb_not_draw"]
            score_left += r["score_left"]
            score_right += r["score_right"]

        if nb_not_draw > 0:
            avg_reward = total_reward / nb_not_draw
            avg_steps = total_steps / nb_not_draw
        else:
            avg_reward = 0
            avg_steps = 0

        if should_print:
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] {nb_tests} tests | Reward: {avg_reward:.2f} | W/D/L: {int(score_left)}/{nb_tests - nb_not_draw}/{int(score_right)} | Steps: {avg_steps:.1f} | Score: {score_left / nb_tests:.2f} / {score_right / nb_tests:.2f} | failed: {nb_fail / nb_tests:.3f}")

        return avg_reward, avg_steps, score_left / nb_tests, score_right / nb_tests, nb_fail / nb_tests, {}

    def shutdown(self):
        self._executor.shutdown(wait=True)


def batched_run_tests(agents, nb_tests, max_steps, players_number, reward_coeff_dict,
                      training_progression=1.0, should_print=True):
    """
    Run nb_tests games using VectorizedEnv — all games in one process, no multiprocessing.
    agents: list of agents with batch_act(states_batch) method.
    """
    from Engine.Physics.VectorizedEnv import VectorizedEnv

    n_players = len(agents)
    venv = VectorizedEnv(nb_tests, players_number, reward_coeff_dict)
    venv.reset_all()

    actions = np.zeros((nb_tests, n_players), dtype=np.int32)
    step_counts = np.zeros(nb_tests, dtype=np.int32)
    total_rewards = np.zeros(nb_tests, dtype=np.float64)
    active_mask = np.ones(nb_tests, dtype=np.bool_)

    start_time = time.time()

    # Initial step to get first visions (actions=0 → all move forward)
    visions, gvisions, rewards, dones = venv.step(actions, active_mask)

    for step in range(max_steps):
        # Batch inference per agent
        for pid in range(n_players):
            states = visions[:, pid]  # (nb_tests, ENTRY_NEURONS)
            agent_actions = agents[pid].batch_act(states)
            actions[:, pid] = agent_actions

        visions, gvisions, rewards, dones = venv.step(actions, active_mask)

        # Accumulate only for active envs
        for pid in range(n_players):
            total_rewards += rewards[:, pid] * active_mask
        step_counts += active_mask.astype(np.int32)

        # Mark newly done envs as inactive
        active_mask &= ~dones

        if not active_mask.any():
            break

    # Aggregate statistics
    scores = venv.scores
    score_left = scores[:, 0].sum()
    score_right = scores[:, 1].sum()
    scored = (scores[:, 0] + scores[:, 1]) > 0
    nb_not_draw = int(scored.sum())
    nb_fail = int((scores[:, 0] != 1).sum())

    avg_reward = total_rewards[scored].sum() / max(nb_not_draw, 1)
    avg_steps = step_counts[scored].sum() / max(nb_not_draw, 1)

    if should_print:
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] {nb_tests} tests | Reward: {avg_reward:.2f} | W/D/L: {int(score_left)}/{nb_tests - nb_not_draw}/{int(score_right)} | Steps: {avg_steps:.1f} | Score: {score_left / nb_tests:.2f} / {score_right / nb_tests:.2f} | failed: {nb_fail / nb_tests:.3f}")

    return avg_reward, avg_steps, score_left / nb_tests, score_right / nb_tests, nb_fail / nb_tests, {}
