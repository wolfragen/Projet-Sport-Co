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
        self.to(device)
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
        self.to(device)
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
        self._actor_in_dim = dimensions[0][0]
        self._critic_in_dim = dimensions[1][0]
        self._buf_actor = None
        self._buf_critic = None
        self._buf_logprobs = None
        self._buf_actions = None
        self._buf_values = None
        self._buf_rewards = None
        self._buf_dones = None
        self._mem_ptr = 0

        self.actor = ActorNetwork(dimensions[0], self.device, lr_actor, lr_decay)
        self.critic = CriticNetwork(dimensions[1], self.device, lr_critic, lr_decay)

    def init_memory(self):
        self._mem_ptr = 0

    def _ensure_buffers(self):
        if self._buf_actor is None:
            n, d = self.rollout_size, self.device
            self._buf_actor   = torch.zeros(n, self._actor_in_dim,  dtype=torch.float32, device=d)
            self._buf_critic  = torch.zeros(n, self._critic_in_dim, dtype=torch.float32, device=d)
            self._buf_logprobs = torch.zeros(n, dtype=torch.float32, device=d)
            self._buf_actions = torch.zeros(n, dtype=torch.long,    device=d)
            self._buf_values  = torch.zeros(n, dtype=torch.float32, device=d)
            self._buf_rewards = torch.zeros(n, dtype=torch.float32, device=d)
            self._buf_dones   = torch.zeros(n, dtype=torch.bool,    device=d)

    def remember(self, actor_states, critic_states, log_probs, dones, vals, actions, rewards):
        """Single-transition write (sequential path)."""
        self._ensure_buffers()
        ptr = self._mem_ptr
        d = self.device
        self._buf_actor[ptr]   = torch.as_tensor(actor_states,  dtype=torch.float32, device=d)
        self._buf_critic[ptr]  = torch.as_tensor(critic_states, dtype=torch.float32, device=d)
        self._buf_logprobs[ptr] = float(log_probs)
        self._buf_actions[ptr] = int(actions)
        self._buf_values[ptr]  = float(vals)
        self._buf_rewards[ptr] = float(rewards)
        self._buf_dones[ptr]   = bool(dones)
        self._mem_ptr += 1

    def remember_batch(self, actor_t, critic_t, logprobs_t, actions_t, values_t, rewards_np, dones_np):
        """Write n_envs transitions at once. Tensor args must already be on self.device."""
        self._ensure_buffers()
        n   = actor_t.shape[0]
        ptr = self._mem_ptr
        self._buf_actor[ptr:ptr+n]   = actor_t
        self._buf_critic[ptr:ptr+n]  = critic_t
        self._buf_logprobs[ptr:ptr+n] = logprobs_t
        self._buf_actions[ptr:ptr+n] = actions_t
        self._buf_values[ptr:ptr+n]  = values_t
        self._buf_rewards[ptr:ptr+n] = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        self._buf_dones[ptr:ptr+n]   = torch.as_tensor(dones_np,   dtype=torch.bool,    device=self.device)
        self._mem_ptr = ptr + n

    @torch.no_grad()
    def batch_act_and_value(self, actor_t, critic_t):
        """Takes GPU tensors, returns GPU tensors (actions, logprobs, values).
        Avoids separate H2D transfers for actor and critic inference."""
        probs  = self.actor.net(actor_t)
        dist   = torch.distributions.Categorical(probs)
        actions  = dist.sample()
        logprobs = dist.log_prob(actions)
        values   = self.critic.net(critic_t).squeeze(-1)
        return actions, logprobs, values

    def evaluate(self, actor_states, critic_states, action):
        actor_states = actor_states.to(self.device)
        critic_states = critic_states.to(self.device)
        action = action.to(self.device)

        action_probs = self.actor.net(actor_states)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic.net(critic_states).squeeze(-1)

        return action_logprobs, state_values, dist_entropy

    def reorder_memory_for_gae(self, n_envs):
        """
        Reorder interleaved transitions from [e0s1,e1s1,...,e(N-1)s1,e0s2,...]
        to [e0s1,...,e0sT,e1s1,...,e1sT,...] so GAE runs per trajectory.
        All 7 GPU buffers are reordered in-place with a single index tensor.
        """
        self._gae_n_envs = n_envs
        if n_envs <= 1:
            return
        n_total = self._mem_ptr
        n_steps = n_total // n_envs
        idx = torch.arange(n_total, device=self.device).reshape(n_steps, n_envs).T.reshape(-1)
        for buf in (self._buf_actor, self._buf_critic, self._buf_logprobs,
                    self._buf_actions, self._buf_values, self._buf_rewards, self._buf_dones):
            buf[:n_total] = buf[:n_total][idx]

    def compute_gae(self, last_value: float, last_done: bool):
        """Vectorized GAE fully on GPU. Runs n_steps iterations (typically 4-8)."""
        n_envs  = getattr(self, '_gae_n_envs', 1)
        n_total = self._mem_ptr
        n_steps = n_total // n_envs

        # Each row is one env's contiguous trajectory — no cross-env interaction possible
        rewards = self._buf_rewards[:n_total].contiguous().view(n_envs, n_steps)
        values  = self._buf_values[:n_total].contiguous().view(n_envs, n_steps)
        dones   = self._buf_dones[:n_total].float().contiguous().view(n_envs, n_steps)

        advantages = torch.zeros(n_envs, n_steps, device=self.device)
        last_val   = (torch.zeros(n_envs, device=self.device) if last_done
                      else values[:, -1].clone())
        gae = torch.zeros(n_envs, device=self.device)

        for t in reversed(range(n_steps)):
            mask  = 1.0 - dones[:, t]
            delta = rewards[:, t] + self.gamma * last_val * mask - values[:, t]
            gae   = delta + self.gamma * self.lmbda * mask * gae
            advantages[:, t] = gae
            last_val = values[:, t].clone()  # clone: defensive copy, no aliasing risk

        advantages = advantages.reshape(-1)
        returns    = advantages + self._buf_values[:n_total]

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def replay(self, last_value: float, last_done: bool):
        n = self._mem_ptr
        old_actor_states  = self._buf_actor[:n]
        old_critic_states = self._buf_critic[:n]
        old_actions       = self._buf_actions[:n]
        old_logprobs      = self._buf_logprobs[:n]

        advantages, returns = self.compute_gae(last_value, last_done)

        for _ in range(self.n_epoch):
            logprobs, state_values, dist_entropy = self.evaluate(old_actor_states, old_critic_states, old_actions)

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

    def save_training_state(self, folder: str, actor_filename: str):
        """Save actor (to actor_filename), critic.pt, and optimizer.pt in folder."""
        os.makedirs(folder, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(folder, actor_filename))
        torch.save(self.critic.state_dict(), os.path.join(folder, "critic.pt"))
        torch.save({
            "actor_opt":  self.actor.optimizer.state_dict(),
            "critic_opt": self.critic.optimizer.state_dict(),
        }, os.path.join(folder, "optimizer.pt"))

    def load_training_state(self, folder: str, actor_filename: str) -> dict:
        """Load actor/critic/optimizer from folder. Graceful fallback if files missing.
        Returns a dict describing what was loaded: {actor, critic, optimizer} booleans."""
        loaded = {"actor": False, "critic": False, "optimizer": False}
        actor_p = os.path.join(folder, actor_filename)
        if os.path.exists(actor_p):
            self.actor.load_state_dict(torch.load(actor_p, map_location=self.device))
            loaded["actor"] = True
        else:
            print(f"Warning: {actor_p} not found, actor not loaded")
            return loaded
        critic_p = os.path.join(folder, "critic.pt")
        if os.path.exists(critic_p):
            self.critic.load_state_dict(torch.load(critic_p, map_location=self.device))
            loaded["critic"] = True
        else:
            print(f"Warning: {critic_p} not found, critic starts fresh")
        opt_p = os.path.join(folder, "optimizer.pt")
        if os.path.exists(opt_p):
            state = torch.load(opt_p, map_location=self.device)
            self.actor.optimizer.load_state_dict(state["actor_opt"])
            self.critic.optimizer.load_state_dict(state["critic_opt"])
            loaded["optimizer"] = True
        else:
            print(f"Warning: {opt_p} not found, Adam moments start fresh")
        return loaded


# =========================
# Helpers
# =========================

def clone_opponent(source: PPOAgent) -> PPOAgent:
    opp = copy.deepcopy(source)
    opp.actor.eval()
    for p in opp.actor.parameters():
        p.requires_grad = False
    # Free rollout buffers — opponents only do inference, never collect transitions
    opp._buf_actor = opp._buf_critic = opp._buf_logprobs = None
    opp._buf_actions = opp._buf_values = opp._buf_rewards = opp._buf_dones = None
    opp._mem_ptr = 0
    return opp


def _select_opponent(opponent_pool, n_left, n_right, model, random_pick):
    # Exactly 2 distinct opponents may be active at once (perf): the latest
    # clone (`opponent_pool[-1]`) and a single `random_pick` chosen by the
    # caller. Refresh `random_pick` periodically to keep diversity over time.
    if n_left == 1 and n_right == 1:
        if random.random() < 0.85:
            return opponent_pool[-1]   # 85% latest clone
        return random_pick             # 15% random from pool
    else:
        if random.random() < 0.85:
            return opponent_pool[-1]   # 85% latest
        return random_pick             # 15% random from pool


def _multi_opponent_batch_act(env_opponents, states_np):
    """Run batch_act per unique opponent. Each env has its own opponent (per-game).
    Groups envs by opponent identity to do one forward pass per distinct opponent.
    states_np: (n_envs, state_dim) numpy array. Returns (n_envs,) int32 action array.
    """
    n = len(env_opponents)
    actions = np.zeros(n, dtype=np.int32)
    groups: dict[int, tuple[object, list[int]]] = {}
    for i, opp in enumerate(env_opponents):
        key = id(opp)
        if key not in groups:
            groups[key] = (opp, [])
        groups[key][1].append(i)
    for opp, idxs in groups.values():
        idx_arr = np.asarray(idxs, dtype=np.int64)
        actions[idx_arr] = opp.batch_act(states_np[idx_arr])
    return actions


def _evaluate_vs_random(model, players_number, max_steps, nb_tests=1024):
    n_left, n_right = players_number
    agents = [model] * n_left + [RandomAgent(action_dim=4)] * n_right
    return runTests(
        players_number=players_number, agents=agents,
        scoring_function=model.scoring_function, reward_coeff_dict=model.reward_coeff_dict,
        max_steps=max_steps, training_progression=1.0, nb_tests=nb_tests, should_print=True,
    )


def _evaluate_vs_agent(model, eval_agent, players_number, max_steps, nb_tests=1024):
    n_left, n_right = players_number
    agents = [model] * n_left + [eval_agent] * n_right
    return runTests(
        players_number=players_number, agents=agents,
        scoring_function=model.scoring_function, reward_coeff_dict=model.reward_coeff_dict,
        max_steps=max_steps, training_progression=1.0, nb_tests=nb_tests, should_print=True,
    )


def _find_latest_actor(folder: str):
    """Return (N, filename) of the highest-N actor_{N}.pt in folder, or None."""
    import re
    if not os.path.isdir(folder):
        return None
    pat = re.compile(r"actor_(\d+)\.pt$")
    best = None
    for f in os.listdir(folder):
        m = pat.match(f)
        if m:
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, f)
    return best


def _save_metadata(folder: str, meta: dict):
    import json, datetime
    os.makedirs(folder, exist_ok=True)
    payload = {**meta, "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(folder, "checkpoint.json"), "w") as fp:
        json.dump(payload, fp, indent=2)


def _load_metadata(folder: str):
    import json
    p = os.path.join(folder, "checkpoint.json")
    if not os.path.exists(p):
        return None
    with open(p) as fp:
        return json.load(fp)


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
    eval_agent=None,
    resume_from: str = None,
    save_interval: int = 500,
):
    """
    Train a single agent against the environment (1v0).
    1 episode = 1 rollout of rollout_size transitions → replay().
    If resume_from is given, reloads actor/critic/optimizer + checkpoint.json from that folder.
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
    use_eval_agent = False

    if resume_from:
        loaded = model.load_training_state(resume_from, "actor.pt")
        meta = _load_metadata(resume_from)
        if loaded["actor"] and meta is not None:
            total_replays = meta.get("total_replays", 0)
            total_games   = meta.get("total_games", 0)
            print(f"Resumed solo from {resume_from}: replays={total_replays} games={total_games}")
        else:
            print(f"Warning: resume_from={resume_from} incomplete, starting fresh")

    next_notify = total_replays + interval_notify
    next_eval   = total_replays + eval_interval
    next_save   = total_replays + save_interval

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
                actor_t = torch.as_tensor(np.ascontiguousarray(visions[:, 0]), dtype=torch.float32, device=model.device)
                critic_t = actor_t  # solo: same state

                actions_t, logprobs_t, values_t = model.batch_act_and_value(actor_t, critic_t)

                actions_np = actions_t.cpu().numpy().astype(np.int32)
                visions, gvisions, rewards, dones = venv.step(actions_np.reshape(n_envs, 1))
                step_in_game += 1

                # Vectorized done/reward computation
                done_timeout = step_in_game >= max_steps_per_game
                done_ppo     = dones | done_timeout
                rewards_adj  = rewards[:, 0].copy()
                rewards_adj[done_timeout] += draw_penalty

                model.remember_batch(actor_t, critic_t, logprobs_t, actions_t, values_t, rewards_adj, done_ppo)
                period_reward  += float(rewards_adj.sum())
                transition_count += n_envs

                dummy = np.zeros((n_envs, 1), dtype=np.int32)
                for i in np.where(dones | done_timeout)[0]:
                    done_goal_i    = bool(dones[i])
                    done_timeout_i = bool(done_timeout[i])

                    if done_goal_i and not done_timeout_i:
                        venv.reset_after_goal(i)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, _, _, _ = venv.step(dummy, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]

                    if done_timeout_i:
                        gf = int(venv.scores[i, 0])
                        total_games += 1
                        period_games += 1
                        period_goals += gf
                        rolling_goals.append(gf)
                        venv.reset_env(i)
                        step_in_game[i] = 0
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, _, _, _ = venv.step(dummy, active_mask=reset_mask)
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
            if eval_agent is not None and use_eval_agent:
                print(">>> Evaluating vs eval agent...")
                _evaluate_vs_agent(model, eval_agent, (1, 1), max_steps_per_game)
            else:
                print(">>> Evaluating vs random agent...")
                result = _evaluate_vs_random(model, (1, 1), max_steps_per_game)
                if eval_agent is not None and result[2] > 0.5:
                    print(f">>> Win rate vs random = {result[2]:.1%} — switching to eval agent!")
                    use_eval_agent = True
            next_eval += eval_interval

        if total_replays >= next_save:
            model.save_training_state(save_path, "actor.pt")
            _save_metadata(save_path, {
                "version": 1, "mode": "solo",
                "total_replays": total_replays, "total_games": total_games,
            })
            next_save += save_interval

    elapsed = int(time.time() - start)
    print(f"Solo training finished in {elapsed}s. Replays: {total_replays}, Games: {total_games}")
    model.save_training_state(save_path, "actor.pt")
    _save_metadata(save_path, {
        "version": 1, "mode": "solo",
        "total_replays": total_replays, "total_games": total_games,
    })


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
    resume_from: str = None,
    rolling_size: int = 250,
    eval_agent=None,
):
    """
    Train agent in 1v1 self-play.
    If resume_from is given, reloads the latest actor_{N}.pt + critic.pt + optimizer.pt
    + checkpoint.json + opponent pool from that folder.
    """
    players_number = (1, 1)
    n_envs = max(1, Settings.N_WORKERS) if Settings.PHYSICS_ENGINE == "numba" else 1
    use_vec = n_envs > 1

    os.makedirs(save_path, exist_ok=True)

    opponent_pool = [clone_opponent(model)]
    total_replays = 0
    total_games = 0
    total_models = 0

    if resume_from:
        latest = _find_latest_actor(resume_from)
        meta = _load_metadata(resume_from)
        if latest is None:
            print(f"Warning: resume_from={resume_from} has no actor_N.pt, starting fresh")
        else:
            total_models = latest[0]
            model.load_training_state(resume_from, latest[1])
            if meta is not None:
                total_replays = meta.get("total_replays", 0)
                total_games   = meta.get("total_games", 0)
            _load_opponent_pool(opponent_pool, resume_from, model, max_pool_size)
            print(f"Resumed 1v1 from {resume_from}: replays={total_replays} games={total_games} models={total_models}")

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

    # Global counters (total_replays/total_games/total_models already initialized or loaded above)
    next_opponent_save = total_replays + opponent_save_interval
    next_eval = total_replays + eval_interval
    next_notify = total_replays + interval_notify
    use_eval_agent = False

    # Per-env step tracking (persistent across replays — envs are NOT reset between replays)
    step_in_game = np.zeros(n_envs, dtype=np.int32) if use_vec else 0

    # Per-env opponent: one opponent per environment, resampled only when that
    # env's current match ends (timeout). Avoids the "1024 rollouts vs a single
    # weak clone" collapse, while keeping a coherent opponent within each game.
    # `random_pick` is refreshed each time a new clone is added to the pool, so
    # at any moment only 2 distinct opponents are active across all envs.
    random_pick = random.choice(opponent_pool)
    if use_vec:
        env_opponents = [_select_opponent(opponent_pool, 1, 1, model, random_pick) for _ in range(n_envs)]
    else:
        env_opponent = _select_opponent(opponent_pool, 1, 1, model, random_pick)

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
                actor_t = torch.as_tensor(np.ascontiguousarray(visions[:, 0]), dtype=torch.float32, device=model.device)
                critic_t = actor_t  # 1v1: critic = actor state

                actions_t, logprobs_t, values_t = model.batch_act_and_value(actor_t, critic_t)

                opp_actions = _multi_opponent_batch_act(env_opponents, visions[:, 1])

                actions_np = actions_t.cpu().numpy().astype(np.int32)
                all_actions = np.stack([actions_np, opp_actions], axis=1)
                visions, gvisions, rewards, dones = venv.step(all_actions)
                step_in_game += 1

                # Vectorized done/reward computation — no per-env Python loop
                done_timeout = step_in_game >= max_steps_per_game
                done_ppo     = dones | done_timeout
                rewards_adj  = rewards[:, 0].copy()
                rewards_adj[done_timeout] += draw_penalty

                model.remember_batch(actor_t, critic_t, logprobs_t, actions_t, values_t, rewards_adj, done_ppo)
                period_reward  += float(rewards_adj.sum())
                transition_count += n_envs

                # Per-env loop only for events (goals/timeouts) — O(1) per step on average
                reset_actions = np.zeros((n_envs, 2), dtype=np.int32)
                for i in np.where(dones | done_timeout)[0]:
                    done_goal_i    = bool(dones[i])
                    done_timeout_i = bool(done_timeout[i])

                    if done_goal_i and not done_timeout_i:
                        venv.reset_after_goal(i)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, gvisions_tmp, _, _ = venv.step(reset_actions, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]
                        gvisions[i] = gvisions_tmp[i]

                    if done_timeout_i:
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
                        env_opponents[i] = _select_opponent(opponent_pool, 1, 1, model, random_pick)
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
                opp_action = env_opponent.act(opp_state, train=False)

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
                    env_opponent = _select_opponent(opponent_pool, 1, 1, model, random_pick)

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
            model.save_training_state(save_path, f"actor_{total_models}.pt")
            _save_metadata(save_path, {
                "version": 1, "mode": "1v1",
                "total_replays": total_replays, "total_games": total_games,
                "total_models": total_models,
            })
            random_pick = random.choice(opponent_pool)
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
            if eval_agent is not None and use_eval_agent:
                print(">>> Evaluating vs eval agent...")
                _evaluate_vs_agent(model, eval_agent, players_number, max_steps_per_game)
            else:
                print(">>> Evaluating vs random agent...")
                result = _evaluate_vs_random(model, players_number, max_steps_per_game)
                if eval_agent is not None and result[2] > 0.5:
                    print(f">>> Win rate vs random = {result[2]:.1%} — switching to eval agent!")
                    use_eval_agent = True
            next_eval += eval_interval

    elapsed = int(time.time() - start)
    print(f"1v1 training finished in {elapsed}s. Replays: {total_replays}, Games: {total_games}")
    model.save_training_state(save_path, "actor_1v1.pt")
    _save_metadata(save_path, {
        "version": 1, "mode": "1v1",
        "total_replays": total_replays, "total_games": total_games,
        "total_models": total_models,
    })


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
    resume_from: str = None,
    rolling_size: int = 250,
    eval_agent=None,
    team_reward_ratio: float = 0.5,
):
    """
    Train agents in XvX self-play. All left players share same model.
    team_reward_ratio: mix between individual and team reward.
        0.0 = pure individual, 1.0 = pure team mean (free-riding).
    If resume_from is given, reloads the latest actor_{N}.pt + critic.pt + optimizer.pt
    + checkpoint.json + opponent pool from that folder.
    """
    n_left, n_right = players_number
    n_players = n_left + n_right
    n_envs = max(1, Settings.N_WORKERS) if Settings.PHYSICS_ENGINE == "numba" else 1
    use_vec = n_envs > 1

    os.makedirs(save_path, exist_ok=True)

    opponent_pool = [clone_opponent(model)]
    total_replays = 0
    total_games = 0
    total_models = 0

    if resume_from:
        latest = _find_latest_actor(resume_from)
        meta = _load_metadata(resume_from)
        if latest is None:
            print(f"Warning: resume_from={resume_from} has no actor_N.pt, starting fresh")
        else:
            total_models = latest[0]
            model.load_training_state(resume_from, latest[1])
            if meta is not None:
                total_replays = meta.get("total_replays", 0)
                total_games   = meta.get("total_games", 0)
            _load_opponent_pool(opponent_pool, resume_from, model, max_pool_size)
            print(f"Resumed team from {resume_from}: replays={total_replays} games={total_games} models={total_models}")

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

    # total_replays/total_games/total_models already initialized or loaded above
    next_opponent_save = total_replays + opponent_save_interval
    next_eval = total_replays + eval_interval
    next_notify = total_replays + interval_notify
    use_eval_agent = False

    step_in_game = np.zeros(n_envs, dtype=np.int32) if use_vec else 0

    # Per-env opponent: one opponent per environment, resampled only when that
    # env's match ends (timeout). Avoids long streaks against a single weak
    # clone while keeping a coherent opponent within each game.
    # `random_pick` is refreshed each time a new clone is added to the pool, so
    # at any moment only 2 distinct opponents are active across all envs.
    random_pick = random.choice(opponent_pool)
    if use_vec:
        env_opponents = [_select_opponent(opponent_pool, n_left, n_right, model, random_pick) for _ in range(n_envs)]
    else:
        env_opponent = _select_opponent(opponent_pool, n_left, n_right, model, random_pick)

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
                # Critic: one forward pass per env (global state)
                critic_t = torch.as_tensor(np.ascontiguousarray(gvisions), dtype=torch.float32, device=model.device)
                with torch.no_grad():
                    values_t = model.critic.net(critic_t).squeeze(-1)  # (n_envs,)

                # Actor: one forward pass for all n_left players stacked
                # Layout: [e0p0, e0p1, e1p0, e1p1, ...] — matches interleaved memory order
                left_states_np = np.ascontiguousarray(visions[:, :n_left].reshape(n_envs * n_left, -1))
                left_t = torch.as_tensor(left_states_np, dtype=torch.float32, device=model.device)
                with torch.no_grad():
                    probs    = model.actor.net(left_t)
                    dist     = torch.distributions.Categorical(probs)
                    actions_t  = dist.sample()    # (n_envs*n_left,)
                    logprobs_t = dist.log_prob(actions_t)

                all_actions = np.zeros((n_envs, n_players), dtype=np.int32)
                all_actions[:, :n_left] = actions_t.cpu().numpy().astype(np.int32).reshape(n_envs, n_left)
                for pid in range(n_right):
                    all_actions[:, n_left + pid] = _multi_opponent_batch_act(env_opponents, visions[:, n_left + pid])

                visions, gvisions, rewards, dones = venv.step(all_actions)
                step_in_game += 1

                # Vectorized done/reward computation
                done_timeout  = step_in_game >= max_steps_per_game
                done_ppo      = dones | done_timeout
                # Mixed reward: (1-ratio)*individual + ratio*team_mean
                indiv_rewards = rewards[:, :n_left]  # (n_envs, n_left)
                team_mean     = indiv_rewards.mean(axis=1, keepdims=True)  # (n_envs, 1)
                mixed_rewards = (1.0 - team_reward_ratio) * indiv_rewards + team_reward_ratio * team_mean
                mixed_rewards[done_timeout] += draw_penalty
                # Flatten to interleaved layout [e0p0, e0p1, e1p0, e1p1, ...]
                team_rewards_rep = mixed_rewards.reshape(-1)
                done_ppo_rep     = np.repeat(done_ppo, n_left)
                # Repeat critic states and values for remember_batch
                critic_rep_t = critic_t.repeat_interleave(n_left, dim=0)  # (n_envs*n_left, critic_dim)
                values_rep_t = values_t.repeat_interleave(n_left)          # (n_envs*n_left,)

                model.remember_batch(left_t, critic_rep_t, logprobs_t, actions_t, values_rep_t,
                                     team_rewards_rep, done_ppo_rep)
                period_reward  += float(team_rewards_rep.sum())
                transition_count += n_envs * n_left

                # Per-env loop only for events
                reset_actions = np.zeros((n_envs, n_players), dtype=np.int32)
                for i in np.where(dones | done_timeout)[0]:
                    done_goal_i    = bool(dones[i])
                    done_timeout_i = bool(done_timeout[i])

                    if done_goal_i and not done_timeout_i:
                        venv.reset_after_goal(i)
                        reset_mask = np.zeros(n_envs, dtype=np.bool_)
                        reset_mask[i] = True
                        visions_tmp, gvisions_tmp, _, _ = venv.step(reset_actions, active_mask=reset_mask)
                        visions[i] = visions_tmp[i]
                        gvisions[i] = gvisions_tmp[i]

                    if done_timeout_i:
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
                        env_opponents[i] = _select_opponent(opponent_pool, n_left, n_right, model, random_pick)
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

                actions_right = [env_opponent.act(opp_states_seq, train=False) for opp_states_seq in opp_states]
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

                # Mixed reward: (1-ratio)*individual + ratio*team_mean
                left_rewards = [float(rews[pid]) if hasattr(env.engine, 'full_step') else step_rewards[pid] for pid in range(n_left)]
                team_mean = sum(left_rewards) / n_left
                mixed_rewards = [
                    (1.0 - team_reward_ratio) * left_rewards[pid] + team_reward_ratio * team_mean
                    for pid in range(n_left)
                ]
                if done_timeout:
                    mixed_rewards = [r + draw_penalty for r in mixed_rewards]

                for pid in range(n_left):
                    model.remember(actor_states[pid], critic_state, logprobs_left[pid],
                                   done_ppo, value, actions_left[pid], mixed_rewards[pid])
                    period_reward += mixed_rewards[pid]
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
                    env_opponent = _select_opponent(opponent_pool, n_left, n_right, model, random_pick)

        model.reorder_memory_for_gae(n_envs * n_left)
        model.replay(0.0, True)
        model.init_memory()
        total_replays += 1

        if total_replays >= next_opponent_save:
            opponent_pool.append(clone_opponent(model))
            if len(opponent_pool) > max_pool_size:
                opponent_pool.pop(0)
            total_models += 1
            model.save_training_state(save_path, f"actor_{total_models}.pt")
            _save_metadata(save_path, {
                "version": 1, "mode": f"team_{n_left}v{n_right}",
                "total_replays": total_replays, "total_games": total_games,
                "total_models": total_models,
            })
            random_pick = random.choice(opponent_pool)
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
            if eval_agent is not None and use_eval_agent:
                print(f">>> Evaluating vs eval agent ({n_left}v{n_right})...")
                _evaluate_vs_agent(model, eval_agent, players_number, max_steps_per_game)
            else:
                print(f">>> Evaluating vs random agent ({n_left}v{n_right})...")
                result = _evaluate_vs_random(model, players_number, max_steps_per_game)
                if eval_agent is not None and result[2] > 0.5:
                    print(f">>> Win rate vs random = {result[2]:.1%} — switching to eval agent!")
                    use_eval_agent = True
            next_eval += eval_interval

    elapsed = int(time.time() - start)
    print(f"{n_left}v{n_right} training finished in {elapsed}s. Replays: {total_replays}, Games: {total_games}")
    model.save_training_state(save_path, f"actor_{n_left}v{n_right}.pt")
    _save_metadata(save_path, {
        "version": 1, "mode": f"team_{n_left}v{n_right}",
        "total_replays": total_replays, "total_games": total_games,
        "total_models": total_models,
    })
