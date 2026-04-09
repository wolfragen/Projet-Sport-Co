import torch
import numpy as np
from torch import nn
import time
import copy
import random
import os
from collections import deque

from AI.Network import DeepRLNetwork
from Engine.Environment import LearningEnvironment
from AI.Algorithms.DQN import runTests
from AI.Algorithms.RANDOM import RandomAgent
from Settings import Settings


# =========================
# Actor Network
# =========================
class ActorNetwork(DeepRLNetwork):
    def __init__(self, dimensions: list[int], device: torch.device, lr: float, lr_decay: bool):
        super().__init__(dimensions=dimensions, last_layer=nn.Softmax(dim=1))
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0)
        if lr_decay:
            self.lr_decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer)
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

    @torch.no_grad()
    def batch_act_train(self, states_batch):
        """Batch inference returning (actions, logprobs) both shape (batch,)."""
        states_t = torch.as_tensor(np.ascontiguousarray(states_batch), dtype=torch.float32, device=self.device)
        probs = self.net(states_t)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        return actions.cpu().numpy(), logprobs.cpu().numpy()


# =========================
# Critic Network
# =========================
class CriticNetwork(DeepRLNetwork):
    def __init__(self, dimensions, device, lr, lr_decay):
        super().__init__(dimensions)
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0)
        if lr_decay:
            self.lr_decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer)
        else:
            self.lr_decay_scheduler = None

    @torch.no_grad()
    def batch_value(self, states_batch):
        """Batch value estimation: states (batch, dim) → values (batch,)."""
        states_t = torch.as_tensor(np.ascontiguousarray(states_batch), dtype=torch.float32, device=self.device)
        return self.net(states_t).squeeze(-1).cpu().numpy()


# =========================
# PPO Agent
# =========================
class PPOAgent:
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
        assert dimensions[1][-1] == 1

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
        self.actor_dimensions = list(dimensions[0])
        self.init_memory()

        self.actor = ActorNetwork(dimensions[0], self.device, lr_actor, lr_decay)
        self.critic = CriticNetwork(dimensions[1], self.device, lr_critic, lr_decay)

    def init_memory(self):
        self.memory = {
            "actor_states": [],
            "critic_states": [],
            "log_probs": [],
            "dones": [],
            "vals": [],
            "actions": [],
            "rewards": [],
        }

    def remember(self, actor_states, critic_states, log_probs, dones, vals, actions, rewards):
        self.memory["actor_states"].append(np.asarray(actor_states, dtype=np.float32))
        self.memory["critic_states"].append(np.asarray(critic_states, dtype=np.float32))
        self.memory["log_probs"].append(float(log_probs))
        self.memory["dones"].append(bool(dones))
        self.memory["vals"].append(float(vals))
        self.memory["actions"].append(int(actions))
        self.memory["rewards"].append(float(rewards))

    def evaluate(self, actor_states, critic_states, action):
        actor_states = actor_states.to(self.device)
        critic_states = critic_states.to(self.device)
        action = action.to(self.device)

        action_probs = self.actor.net(actor_states)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic.net(critic_states)

        return action_logprobs, state_values, dist_entropy

    def reorder_memory_for_gae(self, n_envs):
        """
        Reorder interleaved transitions from [e0s1,e1s1,...,e7s1,e0s2,e1s2,...]
        to [e0s1,e0s2,...,e0sN,e1s1,e1s2,...,e1sN] so GAE computes correct
        temporal differences per trajectory.
        """
        if n_envs <= 1:
            return
        n_total = len(self.memory["rewards"])
        n_steps = n_total // n_envs  # steps per env

        for key in self.memory:
            old = self.memory[key]
            if len(old) != n_total:
                continue
            # Reshape to (n_steps, n_envs) then transpose to (n_envs, n_steps) then flatten
            reordered = []
            for env_idx in range(n_envs):
                for step_idx in range(n_steps):
                    reordered.append(old[step_idx * n_envs + env_idx])
            self.memory[key] = reordered

    def compute_gae(self, last_value: float, last_done: bool):
        rewards = self.memory["rewards"]
        values = self.memory["vals"]
        dones = self.memory["dones"]

        advantages, returns = [], []
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
        old_actor_states = torch.as_tensor(
            np.array(self.memory["actor_states"]), dtype=torch.float32, device=self.device)
        old_critic_states = torch.as_tensor(
            np.array(self.memory["critic_states"]), dtype=torch.float32, device=self.device)
        old_actions = torch.as_tensor(
            self.memory["actions"], dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(
            self.memory["log_probs"], dtype=torch.float32, device=self.device)

        advantages, returns = self.compute_gae(last_value, last_done)

        for _ in range(self.n_epoch):
            logprobs, state_values, dist_entropy = self.evaluate(old_actor_states, old_critic_states, old_actions)
            state_values = self.critic.net(old_critic_states).squeeze(-1)

            ratios = torch.exp(logprobs - old_logprobs)
            loss_clip = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            )

            loss_val = self.critic_loss_coeff * nn.functional.mse_loss(state_values, returns)
            loss_entropy = -self.entropy_loss_coeff * dist_entropy
            loss = (loss_clip + loss_val + loss_entropy).mean()

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            self.actor.optimizer.step()
            self.critic.optimizer.step()

            if self.lr_decay:
                self.actor.lr_decay_scheduler.step()
                self.critic.lr_decay_scheduler.step()

    def act(self, state, train=False):
        return self.actor.act(state, train=False)

    @torch.no_grad()
    def batch_act(self, states_batch):
        states_t = torch.as_tensor(np.ascontiguousarray(states_batch), dtype=torch.float32, device=self.device)
        probs = self.actor.net(states_t)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().cpu().numpy()

    def to_proxy(self):
        from AI.AgentProxy import AgentProxy
        return AgentProxy(
            state_dict=self.actor.state_dict(),
            dimensions=self.actor_dimensions,
            agent_type="ppo",
            device=str(self.device)
        )

    def save(self, actor_path, critic=False, critic_path=None):
        torch.save(self.actor.state_dict(), actor_path)
        if critic:
            assert critic_path is not None
            torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path, critic=False, critic_path=None):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic:
            assert critic_path is not None
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))


# =========================
# Helpers
# =========================

def clone_opponent(source: PPOAgent) -> PPOAgent:
    opp = copy.deepcopy(source)
    opp.actor.eval()
    for p in opp.actor.parameters():
        p.requires_grad = False
    return opp


def _select_opponent(opponent_pool, n_left, n_right, model):
    if n_left == 1 and n_right == 1:
        r = random.random()
        if r < 0.1:
            return model
        elif r < 0.8:
            return opponent_pool[-1]
        else:
            return random.choice(opponent_pool)
    else:
        if random.random() < 0.6:
            return opponent_pool[-1]
        return random.choice(opponent_pool)


def _evaluate_vs_random(model, players_number, max_steps, nb_tests=500):
    n_left, n_right = players_number
    agents = [model] * n_left + [RandomAgent(action_dim=4)] * n_right
    runTests(
        players_number=players_number, agents=agents,
        scoring_function=model.scoring_function, reward_coeff_dict=model.reward_coeff_dict,
        max_steps=max_steps, training_progression=1.0, nb_tests=nb_tests, should_print=True,
    )


def _load_opponent_pool(opponent_pool, load_path, model, max_pool_size):
    import re
    pattern = re.compile(r"actor_(\d+)\.pt")
    files = []
    for f in os.listdir(load_path):
        m = pattern.match(f)
        if m:
            files.append((int(m.group(1)), f))
    files.sort()
    for _, fname in files[-max_pool_size:]:
        opp = clone_opponent(model)
        opp.actor.load_state_dict(torch.load(os.path.join(load_path, fname), map_location=model.device))
        opponent_pool.append(opp)
    print(f"Loaded {len(files)} opponents from {load_path}")


# =========================
# Training: Solo (1v0)
# =========================

def solo_training(
    model: PPOAgent,
    max_duration: int,
    num_episodes: int,
    save_path: str,
    interval_notify: int = 20,
    max_steps_per_game: int = 2048,
    draw_penalty: float = -0.5,
    eval_interval: int = 500,
    rolling_size: int = 250,
):
    """
    Train a single agent against the environment (1v0).
    1 episode = 1 rollout of rollout_size transitions → replay().
    """
    players_number = (1, 0)
    n_envs = max(1, Settings.N_WORKERS) if Settings.PHYSICS_ENGINE == "numba" else 1
    use_vec = n_envs > 1

    os.makedirs(save_path, exist_ok=True)

    if use_vec:
        from Engine.Physics.VectorizedEnv import VectorizedEnv
        venv = VectorizedEnv(n_envs, players_number, model.reward_coeff_dict)
        venv.reset_all()
        dummy_actions = np.zeros((n_envs, 1), dtype=np.int32)
        visions, gvisions, _, _ = venv.step(dummy_actions)
    else:
        env = LearningEnvironment(players_number=players_number,
            scoring_function=model.scoring_function, reward_coeff_dict=model.reward_coeff_dict, human=False)

    print(f"Starting solo training for {max_duration}s ({num_episodes} replays x {model.rollout_size} transitions) | {n_envs} envs")
    start = time.time()

    period_reward = 0.0
    period_games = 0
    period_goals = 0
    total_replays = 0
    total_games = 0
    next_notify = interval_notify
    next_eval = eval_interval

    rolling_goals = deque(maxlen=rolling_size)

    step_in_game = np.zeros(n_envs, dtype=np.int32) if use_vec else 0
    if not use_vec:
        state = env.getState(0)

    while total_replays < num_episodes:
        if time.time() - start > max_duration:
            print(f"Time limit reached at {total_replays} replays")
            break

        transition_count = 0

        while transition_count < model.rollout_size:

            if use_vec:
                actor_states = np.ascontiguousarray(visions[:, 0])
                critic_states = actor_states
                actions, logprobs = model.actor.batch_act_train(actor_states)
                values = model.critic.batch_value(critic_states)

                all_actions = actions.reshape(n_envs, 1)
                visions, gvisions, rewards, dones = venv.step(all_actions)
                step_in_game += 1

                for i in range(n_envs):
                    done_goal = bool(dones[i])
                    done_timeout = step_in_game[i] >= max_steps_per_game
                    done_ppo = done_goal or done_timeout
                    r = float(rewards[i, 0])

                    if done_timeout:
                        r += draw_penalty  # timeout draw only

                    model.remember(actor_states[i], critic_states[i], logprobs[i], done_ppo, values[i], int(actions[i]), r)
                    period_reward += r
                    transition_count += 1

                    # Goal: reset positions, keep scores, continue
                    if done_goal and not done_timeout:
                        venv.reset_after_goal(i)
                        dummy = np.zeros((n_envs, 1), dtype=np.int32)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, gvisions_tmp, _, _ = venv.step(dummy, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]

                    # Timeout: end of match, stats, full reset
                    if done_timeout:
                        gf = int(venv.scores[i, 0])
                        total_games += 1
                        period_games += 1
                        period_goals += gf
                        rolling_goals.append(gf)
                        venv.reset_env(i)
                        step_in_game[i] = 0
                        dummy = np.zeros((n_envs, 1), dtype=np.int32)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, gvisions_tmp, _, _ = venv.step(dummy, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]
            else:
                action, logprob = model.actor.act(state)
                critic_state = state
                with torch.no_grad():
                    value = model.critic.net(
                        torch.as_tensor(critic_state, dtype=torch.float32, device=model.device).unsqueeze(0)).item()

                if hasattr(env.engine, 'full_step'):
                    vis, gv, rews, done_flag = env.fast_step([action])
                    reward, next_state, done = float(rews[0]), vis[0], done_flag
                else:
                    env.playerAct(0, action)
                    reward, done = env.step()[0], env.isDone()
                    next_state = env.getState(0)

                step_in_game += 1
                done_timeout = step_in_game >= max_steps_per_game
                done_ppo = done or done_timeout

                if done_timeout:
                    reward += draw_penalty  # timeout draw only

                model.remember(state, critic_state, logprob, done_ppo, value, action, reward)
                period_reward += reward
                transition_count += 1
                state = next_state

                # Goal: reset positions, keep scores, continue
                if done and not done_timeout:
                    env.reset_after_goal()
                    state = env.getState(0)

                # Timeout: end of match, stats, full reset
                if done_timeout:
                    gf = int(env.score[0])
                    total_games += 1
                    period_games += 1
                    period_goals += gf
                    rolling_goals.append(gf)
                    env.reset()
                    state = env.getState(0)
                    step_in_game = 0

        model.reorder_memory_for_gae(n_envs)
        model.replay(0.0, True)
        model.init_memory()
        total_replays += 1

        if total_replays >= next_notify:
            if period_games > 0:
                elapsed = int(time.time() - start)
                avg_r = period_reward / period_games
                avg_gf = period_goals / period_games
                r_gf = np.mean(rolling_goals) if rolling_goals else 0

                eta = (elapsed / max(total_replays, 1)) * (num_episodes - total_replays)
                eta_str = f"{int(eta//3600)}h{int(eta%3600//60):02d}m" if eta > 3600 else f"{int(eta//60)}m{int(eta%60):02d}s"

                print(f"[{elapsed}s] Replay {total_replays}/{num_episodes} | Games: {total_games} | ETA: {eta_str}")
                print(f"  Period ({period_games}g): Goals/game: {avg_gf:.2f} | Reward: {avg_r:.4f}")
                print(f"  Rolling ({len(rolling_goals)}g): Goals/game: {r_gf:.2f}")

                period_reward = 0.0
                period_games = 0
                period_goals = 0
            next_notify += interval_notify

        if total_replays >= next_eval:
            print(">>> Evaluating vs random agent...")
            _evaluate_vs_random(model, (1, 1), max_steps_per_game)
            next_eval += eval_interval

    elapsed = int(time.time() - start)
    print(f"Solo training finished in {elapsed}s. Replays: {total_replays}, Games: {total_games}")
    model.save(os.path.join(save_path, "actor_solo.pt"))


# =========================
# Training: 1v1
# =========================

def one_v_one_training(
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
    load_existing: bool = False,
    load_path: str = None,
    rolling_size: int = 250,
):
    """
    Train agent in 1v1 self-play.
    """
    players_number = (1, 1)
    n_envs = max(1, Settings.N_WORKERS) if Settings.PHYSICS_ENGINE == "numba" else 1
    use_vec = n_envs > 1

    os.makedirs(save_path, exist_ok=True)

    opponent_pool = [clone_opponent(model)]
    if load_existing and load_path:
        _load_opponent_pool(opponent_pool, load_path, model, max_pool_size)

    if use_vec:
        from Engine.Physics.VectorizedEnv import VectorizedEnv
        venv = VectorizedEnv(n_envs, players_number, model.reward_coeff_dict)
        venv.reset_all()
        dummy_actions = np.zeros((n_envs, 2), dtype=np.int32)
        visions, gvisions, _, _ = venv.step(dummy_actions)
    else:
        env = LearningEnvironment(players_number=players_number,
            scoring_function=model.scoring_function, reward_coeff_dict=model.reward_coeff_dict, human=False)

    print(f"Starting 1v1 training for {max_duration}s ({num_episodes} replays x {model.rollout_size} transitions) | {n_envs} envs")
    start = time.time()

    # Stats (reset each reporting period)
    period_reward = 0.0
    period_games = 0
    period_goals_for = 0
    period_goals_against = 0
    wins = losses = draws = 0

    # Rolling averages
    rolling_goals = deque(maxlen=rolling_size)  # (goals_for, goals_against) per match
    rolling_wins = deque(maxlen=rolling_size)  # 1=win, 0=draw, -1=loss

    # Global counters
    total_replays = 0
    total_games = 0
    total_models = len(opponent_pool)
    next_opponent_save = opponent_save_interval
    next_eval = eval_interval
    next_notify = interval_notify

    # Per-env step tracking (persistent across replays — envs are NOT reset between replays)
    step_in_game = np.zeros(n_envs, dtype=np.int32) if use_vec else 0

    # Select first opponent
    opponent = _select_opponent(opponent_pool, 1, 1, model)

    # Sequential: init state
    if not use_vec:
        env.reset()
        state = env.getState(0)

    while total_replays < num_episodes:
        if time.time() - start > max_duration:
            print(f"Time limit reached at {total_replays} replays")
            break

        # ---- Collect exactly rollout_size transitions (across all envs) ----
        transition_count = 0

        while transition_count < model.rollout_size:

            if use_vec:
                actor_states = np.ascontiguousarray(visions[:, 0])
                critic_states = actor_states  # 1v1: critic = actor state
                actions_0, logprobs = model.actor.batch_act_train(actor_states)
                values = model.critic.batch_value(critic_states)

                opp_states = visions[:, 1]
                opp_actions = opponent.batch_act(opp_states)

                all_actions = np.stack([actions_0, opp_actions], axis=1)
                visions, gvisions, rewards, dones = venv.step(all_actions)
                step_in_game += 1

                for i in range(n_envs):
                    done_goal = bool(dones[i])
                    done_timeout = step_in_game[i] >= max_steps_per_game
                    done_ppo = done_goal or done_timeout
                    r = float(rewards[i, 0])

                    if done_timeout:
                        r += draw_penalty  # timeout draw only

                    model.remember(actor_states[i], critic_states[i], logprobs[i],
                                   done_ppo, values[i], int(actions_0[i]), r)
                    period_reward += r
                    transition_count += 1

                    # Goal scored: reset positions, keep scores, continue match
                    if done_goal and not done_timeout:
                        venv.reset_after_goal(i)
                        reset_actions = np.zeros((n_envs, 2), dtype=np.int32)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, gvisions_tmp, _, _ = venv.step(reset_actions, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]
                        gvisions[i] = gvisions_tmp[i]

                    # Timeout: end of match, record stats, full reset
                    if done_timeout:
                        gf = int(venv.scores[i, 0])
                        ga = int(venv.scores[i, 1])
                        total_games += 1
                        period_games += 1
                        period_goals_for += gf
                        period_goals_against += ga
                        rolling_goals.append((gf, ga))
                        if gf > ga:
                            wins += 1
                            rolling_wins.append(1)
                        elif ga > gf:
                            losses += 1
                            rolling_wins.append(-1)
                        else:
                            draws += 1
                            rolling_wins.append(0)

                        venv.reset_env(i)
                        step_in_game[i] = 0
                        reset_actions = np.zeros((n_envs, 2), dtype=np.int32)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, gvisions_tmp, _, _ = venv.step(reset_actions, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]
                        gvisions[i] = gvisions_tmp[i]

            else:
                action, logprob = model.actor.act(state)
                critic_state = state  # 1v1: critic = actor state
                with torch.no_grad():
                    value = model.critic.net(
                        torch.as_tensor(critic_state, dtype=torch.float32, device=model.device).unsqueeze(0)).item()

                opp_state = env.getState(1)
                opp_action = opponent.act(opp_state, train=False)

                if hasattr(env.engine, 'full_step'):
                    vis, gv, rews, done_flag = env.fast_step([action, opp_action])
                    reward, next_state, done = float(rews[0]), vis[0], done_flag
                else:
                    env.playerAct(0, action)
                    env.playerAct(1, opp_action)
                    sr = env.step()
                    reward, done = sr[0], env.isDone()
                    next_state = env.getState(0)

                step_in_game += 1
                done_timeout = step_in_game >= max_steps_per_game
                done_ppo = done or done_timeout
                score = env.engine._score if hasattr(env.engine, '_score') else env.score

                if done_timeout:
                    reward += draw_penalty  # timeout draw only

                model.remember(state, critic_state, logprob, done_ppo, value, action, reward)
                period_reward += reward
                transition_count += 1
                state = next_state

                # Goal: reset positions, keep scores, continue match
                if done and not done_timeout:
                    env.reset_after_goal()
                    state = env.getState(0)

                # Timeout: end of match, record stats, full reset
                if done_timeout:
                    gf = int(score[0])
                    ga = int(score[1])
                    total_games += 1
                    period_games += 1
                    period_goals_for += gf
                    period_goals_against += ga
                    rolling_goals.append((gf, ga))
                    if gf > ga:
                        wins += 1
                        rolling_wins.append(1)
                    elif ga > gf:
                        losses += 1
                        rolling_wins.append(-1)
                    else:
                        draws += 1
                        rolling_wins.append(0)
                    env.reset()
                    state = env.getState(0)
                    step_in_game = 0

        # ---- Replay (backprop) on exactly rollout_size transitions ----
        model.reorder_memory_for_gae(n_envs)
        model.replay(0.0, True)
        model.init_memory()
        total_replays += 1

        # ---- Opponent pool update ----
        if total_replays >= next_opponent_save:
            opponent_pool.append(clone_opponent(model))
            if len(opponent_pool) > max_pool_size:
                opponent_pool.pop(0)
            total_models += 1
            model.save(os.path.join(save_path, f"actor_{total_models}.pt"),
                       critic=True, critic_path=os.path.join(save_path, f"critic_{total_models}.pt"))
            # Keep only last 100 critic files
            old_critic = os.path.join(save_path, f"critic_{total_models - 100}.pt")
            if os.path.exists(old_critic):
                os.remove(old_critic)
            opponent = _select_opponent(opponent_pool, 1, 1, model)
            next_opponent_save += opponent_save_interval

        # ---- Diagnostics ----
        if total_replays >= next_notify:
            if period_games > 0:
                elapsed = int(time.time() - start)
                avg_r = period_reward / period_games
                avg_gf = period_goals_for / period_games
                avg_ga = period_goals_against / period_games
                win_pct = wins / period_games * 100
                loss_pct = losses / period_games * 100
                draw_pct = draws / period_games * 100

                # Rolling averages
                r_win = sum(1 for w in rolling_wins if w == 1) / max(len(rolling_wins), 1) * 100
                r_gf = np.mean([g[0] for g in rolling_goals]) if rolling_goals else 0
                r_ga = np.mean([g[1] for g in rolling_goals]) if rolling_goals else 0

                eta = (elapsed / max(total_replays, 1)) * (num_episodes - total_replays)
                eta_str = f"{int(eta//3600)}h{int(eta%3600//60):02d}m" if eta > 3600 else f"{int(eta//60)}m{int(eta%60):02d}s"

                print(f"[{elapsed}s] Replay {total_replays}/{num_episodes} | Games: {total_games} | Gen {total_models} | ETA: {eta_str}")
                print(f"  Period ({period_games}g): W/D/L {wins}/{draws}/{losses} ({win_pct:.0f}%/{draw_pct:.0f}%/{loss_pct:.0f}%) | Goals: {avg_gf:.1f}-{avg_ga:.1f} | Reward: {avg_r:.4f}")
                print(f"  Rolling ({len(rolling_wins)}g): Win%: {r_win:.1f}% | Goals: {r_gf:.2f}-{r_ga:.2f}")

                period_reward = 0.0
                period_games = 0
                period_goals_for = 0
                period_goals_against = 0
                wins = losses = draws = 0
            next_notify += interval_notify

        # ---- Evaluation ----
        if total_replays >= next_eval:
            print(">>> Evaluating vs random agent...")
            _evaluate_vs_random(model, players_number, max_steps_per_game)
            next_eval += eval_interval

    elapsed = int(time.time() - start)
    print(f"1v1 training finished in {elapsed}s. Replays: {total_replays}, Games: {total_games}")
    model.save(os.path.join(save_path, "actor_1v1.pt"), critic=True,
               critic_path=os.path.join(save_path, "critic_1v1.pt"))


# =========================
# Training: Team (XvX)
# =========================

def team_training(
    model: PPOAgent,
    players_number: tuple[int, int],
    max_duration: int,
    num_episodes: int,
    save_path: str,
    interval_notify: int = 10,
    opponent_save_interval: int = 50,
    max_pool_size: int = 10,
    draw_penalty: float = -0.5,
    max_steps_per_game: int = 2048,
    eval_interval: int = 500,
    load_existing: bool = False,
    load_path: str = None,
    rolling_size: int = 250,
):
    """
    Train agents in XvX self-play. All left players share same model.
    """
    n_left, n_right = players_number
    n_players = n_left + n_right
    n_envs = max(1, Settings.N_WORKERS) if Settings.PHYSICS_ENGINE == "numba" else 1
    use_vec = n_envs > 1

    os.makedirs(save_path, exist_ok=True)

    opponent_pool = [clone_opponent(model)]
    if load_existing and load_path:
        _load_opponent_pool(opponent_pool, load_path, model, max_pool_size)

    if use_vec:
        from Engine.Physics.VectorizedEnv import VectorizedEnv
        venv = VectorizedEnv(n_envs, players_number, model.reward_coeff_dict)
        venv.reset_all()
        dummy_actions = np.zeros((n_envs, n_players), dtype=np.int32)
        visions, gvisions, _, _ = venv.step(dummy_actions)
    else:
        env = LearningEnvironment(players_number=players_number,
            scoring_function=model.scoring_function, reward_coeff_dict=model.reward_coeff_dict, human=False)

    print(f"Starting {n_left}v{n_right} training for {max_duration}s ({num_episodes} replays x {model.rollout_size} transitions) | {n_envs} envs")
    start = time.time()

    period_reward = 0.0
    period_games = 0
    period_goals_for = 0
    period_goals_against = 0
    wins = losses = draws = 0

    rolling_goals = deque(maxlen=rolling_size)
    rolling_wins = deque(maxlen=rolling_size)

    total_replays = 0
    total_games = 0
    total_models = len(opponent_pool)
    next_opponent_save = opponent_save_interval
    next_eval = eval_interval
    next_notify = interval_notify

    step_in_game = np.zeros(n_envs, dtype=np.int32) if use_vec else 0
    opponent = _select_opponent(opponent_pool, n_left, n_right, model)

    if not use_vec:
        env.reset()
        actor_states = [env.getState(pid) for pid in range(n_left)]
        opp_states = [env.getState(pid) for pid in range(n_left, n_players)]

    while total_replays < num_episodes:
        if time.time() - start > max_duration:
            print(f"Time limit reached at {total_replays} replays")
            break

        transition_count = 0

        while transition_count < model.rollout_size:

            if use_vec:
                critic_states = np.ascontiguousarray(gvisions)
                values = model.critic.batch_value(critic_states)

                all_actions = np.zeros((n_envs, n_players), dtype=np.int32)
                all_logprobs = np.zeros((n_envs, n_left), dtype=np.float64)

                for pid in range(n_left):
                    a_states = np.ascontiguousarray(visions[:, pid])
                    acts, lps = model.actor.batch_act_train(a_states)
                    all_actions[:, pid] = acts
                    all_logprobs[:, pid] = lps

                for pid in range(n_right):
                    opp_states_batch = visions[:, n_left + pid]
                    all_actions[:, n_left + pid] = opponent.batch_act(opp_states_batch)

                visions, gvisions, rewards, dones = venv.step(all_actions)
                step_in_game += 1

                for i in range(n_envs):
                    done_goal = bool(dones[i])
                    done_timeout = step_in_game[i] >= max_steps_per_game
                    done_ppo = done_goal or done_timeout

                    # Team-shared reward: average across left players
                    team_reward = float(np.mean(rewards[i, :n_left]))
                    if done_timeout:
                        team_reward += draw_penalty

                    for pid in range(n_left):
                        model.remember(visions[i, pid], critic_states[i], all_logprobs[i, pid],
                                       done_ppo, values[i], int(all_actions[i, pid]), team_reward)
                        period_reward += team_reward
                        transition_count += 1

                    # Goal: reset positions, keep scores, continue match
                    if done_goal and not done_timeout:
                        venv.reset_after_goal(i)
                        reset_actions = np.zeros((n_envs, n_players), dtype=np.int32)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, gvisions_tmp, _, _ = venv.step(reset_actions, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]
                        gvisions[i] = gvisions_tmp[i]

                    # Timeout: end of match, record stats, full reset
                    if done_timeout:
                        gf = int(venv.scores[i, 0])
                        ga = int(venv.scores[i, 1])
                        total_games += 1
                        period_games += 1
                        period_goals_for += gf
                        period_goals_against += ga
                        rolling_goals.append((gf, ga))
                        if gf > ga:
                            wins += 1
                            rolling_wins.append(1)
                        elif ga > gf:
                            losses += 1
                            rolling_wins.append(-1)
                        else:
                            draws += 1
                            rolling_wins.append(0)
                        venv.reset_env(i)
                        step_in_game[i] = 0
                        reset_actions = np.zeros((n_envs, n_players), dtype=np.int32)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, gvisions_tmp, _, _ = venv.step(reset_actions, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]
                        gvisions[i] = gvisions_tmp[i]

            else:
                critic_state = env.getGlobalState()
                with torch.no_grad():
                    value = model.critic.net(
                        torch.as_tensor(critic_state, dtype=torch.float32, device=model.device).unsqueeze(0)).item()

                actions_left, logprobs_left = [], []
                for pid in range(n_left):
                    a, lp = model.actor.act(actor_states[pid])
                    actions_left.append(a)
                    logprobs_left.append(lp)

                actions_right = [opponent.act(opp_states_seq, train=False) for opp_states_seq in opp_states]
                all_actions_list = actions_left + actions_right

                if hasattr(env.engine, 'full_step'):
                    vis, gv, rews, done_flag = env.fast_step(all_actions_list)
                    done = done_flag
                else:
                    for pid, a in enumerate(all_actions_list):
                        env.playerAct(pid, a)
                    step_rewards = env.step()
                    done = env.isDone()

                step_in_game += 1
                done_timeout = step_in_game >= max_steps_per_game
                done_ppo = done or done_timeout

                # Team-shared reward: average across left players
                left_rewards = [float(rews[pid]) if hasattr(env.engine, 'full_step') else step_rewards[pid] for pid in range(n_left)]
                team_reward = sum(left_rewards) / n_left
                if done_timeout:
                    team_reward += draw_penalty

                for pid in range(n_left):
                    model.remember(actor_states[pid], critic_state, logprobs_left[pid],
                                   done_ppo, value, actions_left[pid], team_reward)
                    period_reward += team_reward
                    transition_count += 1

                if hasattr(env.engine, 'full_step'):
                    actor_states = [vis[pid] for pid in range(n_left)]
                    opp_states = [vis[pid] for pid in range(n_left, n_players)]
                else:
                    actor_states = [env.getState(pid) for pid in range(n_left)]
                    opp_states = [env.getState(pid) for pid in range(n_left, n_players)]

                # Goal: reset positions, keep scores, continue match
                if done and not done_timeout:
                    env.reset_after_goal()
                    actor_states = [env.getState(pid) for pid in range(n_left)]
                    opp_states = [env.getState(pid) for pid in range(n_left, n_players)]

                # Timeout: end of match, record stats, full reset
                if done_timeout:
                    score = env.engine._score if hasattr(env.engine, '_score') else env.score
                    gf = int(score[0])
                    ga = int(score[1])
                    total_games += 1
                    period_games += 1
                    period_goals_for += gf
                    period_goals_against += ga
                    rolling_goals.append((gf, ga))
                    if gf > ga:
                        wins += 1
                        rolling_wins.append(1)
                    elif ga > gf:
                        losses += 1
                        rolling_wins.append(-1)
                    else:
                        draws += 1
                        rolling_wins.append(0)
                    env.reset()
                    actor_states = [env.getState(pid) for pid in range(n_left)]
                    opp_states = [env.getState(pid) for pid in range(n_left, n_players)]
                    step_in_game = 0

        model.reorder_memory_for_gae(n_envs * n_left)
        model.replay(0.0, True)
        model.init_memory()
        total_replays += 1

        if total_replays >= next_opponent_save:
            opponent_pool.append(clone_opponent(model))
            if len(opponent_pool) > max_pool_size:
                opponent_pool.pop(0)
            total_models += 1
            model.save(os.path.join(save_path, f"actor_{total_models}.pt"),
                       critic=True, critic_path=os.path.join(save_path, f"critic_{total_models}.pt"))
            old_critic = os.path.join(save_path, f"critic_{total_models - 100}.pt")
            if os.path.exists(old_critic):
                os.remove(old_critic)
            opponent = _select_opponent(opponent_pool, n_left, n_right, model)
            next_opponent_save += opponent_save_interval

        if total_replays >= next_notify:
            if period_games > 0:
                elapsed = int(time.time() - start)
                avg_r = period_reward / period_games
                avg_gf = period_goals_for / period_games
                avg_ga = period_goals_against / period_games
                win_pct = wins / period_games * 100
                loss_pct = losses / period_games * 100
                draw_pct = draws / period_games * 100

                r_win = sum(1 for w in rolling_wins if w == 1) / max(len(rolling_wins), 1) * 100
                r_gf = np.mean([g[0] for g in rolling_goals]) if rolling_goals else 0
                r_ga = np.mean([g[1] for g in rolling_goals]) if rolling_goals else 0

                eta = (elapsed / max(total_replays, 1)) * (num_episodes - total_replays)
                eta_str = f"{int(eta//3600)}h{int(eta%3600//60):02d}m" if eta > 3600 else f"{int(eta//60)}m{int(eta%60):02d}s"

                print(f"[{elapsed}s] Replay {total_replays}/{num_episodes} | Games: {total_games} | Gen {total_models} | ETA: {eta_str}")
                print(f"  Period ({period_games}g): W/D/L {wins}/{draws}/{losses} ({win_pct:.0f}%/{draw_pct:.0f}%/{loss_pct:.0f}%) | Goals: {avg_gf:.1f}-{avg_ga:.1f} | Reward/p: {avg_r / n_left:.4f}")
                print(f"  Rolling ({len(rolling_wins)}g): Win%: {r_win:.1f}% | Goals: {r_gf:.2f}-{r_ga:.2f}")

                period_reward = 0.0
                period_games = 0
                period_goals_for = 0
                period_goals_against = 0
                wins = losses = draws = 0
            next_notify += interval_notify

        if total_replays >= next_eval:
            print(f">>> Evaluating vs random agent ({n_left}v{n_right})...")
            _evaluate_vs_random(model, players_number, max_steps_per_game)
            next_eval += eval_interval

    elapsed = int(time.time() - start)
    print(f"{n_left}v{n_right} training finished in {elapsed}s. Replays: {total_replays}, Games: {total_games}")
    model.save(os.path.join(save_path, f"actor_{n_left}v{n_right}.pt"), critic=True,
               critic_path=os.path.join(save_path, f"critic_{n_left}v{n_right}.pt"))
