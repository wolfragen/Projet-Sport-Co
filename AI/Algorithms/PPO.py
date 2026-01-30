import torch
import numpy as np
from torch import nn
import time
from AI.Network import DeepRLNetwork
from Engine.Environment import LearningEnvironment
from AI.Algorithms.DQN import runTests
from AI.Algorithms.RANDOM import RandomAgent

import copy
import random
import os
from collections import deque


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
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        probs = self.net(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        if not train:
            return action.item()

        logprob = dist.log_prob(action)

        if log_prob_only:
            return logprob.item()

        return action.item(), logprob.item()


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
        assert dimensions[0][0] == dimensions[1][0]

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

        self.actor = ActorNetwork(dimensions[0], self.device, lr_actor, lr_decay)
        self.critic = CriticNetwork(dimensions[1], self.device, lr_critic, lr_decay)

    def init_memory(self):
        self.memory = {
            "states": [],
            "log_probs": [],
            "dones": [],
            "vals": [],
            "actions": [],
            "rewards": [],
        }

    # -------- CPU-safe memory --------
    def remember(self, states, log_probs, dones, vals, actions, rewards):
        self.memory["states"].append(np.asarray(states, dtype=np.float32))
        self.memory["log_probs"].append(float(log_probs))
        self.memory["dones"].append(bool(dones))
        self.memory["vals"].append(float(vals))
        self.memory["actions"].append(int(actions))
        self.memory["rewards"].append(float(rewards))

    # -------- Device-safe evaluation --------
    def evaluate(self, states, action):
        states = states.to(self.device)
        action = action.to(self.device)

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
        old_states = torch.as_tensor(
            np.array(self.memory["states"]),
            dtype=torch.float32,
            device=self.device,
        )
        old_actions = torch.as_tensor(
            self.memory["actions"], dtype=torch.long, device=self.device
        )
        old_logprobs = torch.as_tensor(
            self.memory["log_probs"], dtype=torch.float32, device=self.device
        )

        advantages, returns = self.compute_gae(last_value, last_done)

        for _ in range(self.n_epoch):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            state_values = state_values.squeeze(-1)

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

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

def train_PPO_model(
    model : PPOAgent,
    max_duration : int,
    num_episodes: int,
    save_path : str,
    interval_notify : int = 20,
    max_steps_per_game: int = 2048,
    draw_penalty = -0.5):
    """ Train a PPO model

    Parameters
    ----------
    model : PPOAgent
        The PPOAgent to train
    max_duration : int
        Max duration of the training process in seconds.
    num_episodes : int
        Number of episodes for the training. Can stop earlier if max_duration is reached. 
    save_path : str
        Where the model should be saved 
    interval_notify : int
        Number of episode until printing information about the current progress in the console"""

    reward_sums = {}  # cumul des rewards par type

    env = LearningEnvironment(players_number=(1,0), 
                              scoring_function=model.scoring_function, 
                              reward_coeff_dict=model.reward_coeff_dict,
                              human=False)
    print(f"Starting training for {max_duration} seconds ({num_episodes} episodes)")
    start = time.time()
    current_reward = 0
    num_game = 0
    score_history_1, score_history_2 = 0,0
    
    done = False
    done_ppo = False
    state = env.getState(0)
    step_in_game = 0
    total_steps = 0
    avg_total_reward = 0.0

    for i_episode in range(1, num_episodes + 1):
        if time.time() - start > max_duration:
            print("Reached max time for training, interrupting")
            break
        loss_hist = {"clip":0, "val":0, "entropy":0}

        for _ in range(model.rollout_size):
            step_in_game += 1
            state_t = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)

            with torch.no_grad():
                action, logprob = model.actor.act(state)
                value = model.critic.net(state_t).squeeze(-1).item()
           
            env.playerAct(0, action)

            rewards = env.step(debug=True)
            reward = rewards[0]

            reward_dict = env.last_reward_components[0]


            # Initialisation dynamique des clés
            if not reward_sums:
                reward_sums = {k: 0.0 for k in reward_dict.keys()}

            for k, v in reward_dict.items():
                reward_sums[k] += v

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
            
        loss = model.replay(last_value=last_value, last_done=done_ppo)

        loss_hist["clip"] += loss["clip"]
        loss_hist["val"] += loss["val"]
        loss_hist["entropy"] += loss["entropy"]

        model.init_memory()
        if i_episode%interval_notify == 0:
            avg_reward_components = {
                k: v / max(num_game, 1)
                for k, v in reward_sums.items()
            }


            print(f"[{int(time.time()-start)}s] Episode {i_episode} | ", end="")
            print(" | ".join([f"{k}: {v:.4f}" for k, v in avg_reward_components.items()]), end=" | ")
            print(f"avg_steps_per_game: {total_steps / max(num_game,1):.1f} | ", end="")
            print(f"number_of_games: {num_game} | ", end="")
            print(f"Loss_clip : {loss_hist['clip']} | ", end="")
            print(f"Loss_val : {loss_hist['val']} | ", end="")
            print(f"Loss_entropy : {loss_hist['entropy']} | ", end="")
            print(f"Score: {score_history_1/num_game if num_game != 0 else 0:.2f} - {score_history_2/num_game if num_game != 0 else 0:.2f}")

            last_total_reward = avg_reward_components.get("total_reward", 0)
            reward_sums = {k: 0.0 for k in reward_sums.keys()}
            num_game = 0
            current_reward = 0
            total_steps = 0
            score_history_1, score_history_2 = 0,0
            loss_hist = {"clip":0, "val":0, "entropy":0}

            if last_total_reward >= avg_total_reward and i_episode >= num_episodes // 4:
                avg_total_reward = last_total_reward
                print("Saving network...")
                model.save(save_path + "best_model.pt")
                print("Network saved.")

    print("Saving network...")
    model.save(save_path + "last_model.pt")
    print("Training finished")
    
def clone_opponent(source: PPOAgent) -> PPOAgent:
    opp = copy.deepcopy(source)
    opp.actor.eval()
    for p in opp.actor.parameters():
        p.requires_grad = False
    return opp

def runTests(
    players_number,
    agents,
    scoring_function,
    reward_coeff_dict,
    max_steps,
    training_progression,
    nb_tests=100,
    should_print=True,
):
    env = LearningEnvironment(
        players_number=players_number,
        scoring_function=scoring_function,
        reward_coeff_dict=reward_coeff_dict,
        training_progression=training_progression,
        human=False,
    )

    wins = losses = draws = 0
    total_steps = 0
    score_0 = score_1 = 0

    for test in range(nb_tests):

        env.reset()
        state_0 = env.getState(0)
        state_1 = env.getState(1)
        step_in_game = 0

        while True:
            # -------- Agent 0
            with torch.no_grad():
                if hasattr(agents[0], "actor"):
                    action_0, _ = agents[0].actor.act(state_0)
                else:
                    action_0 = agents[0].act(state_0, train=False)

            # -------- Agent 1
            with torch.no_grad():
                if hasattr(agents[1], "actor"):
                    action_1, _ = agents[1].actor.act(state_1)
                else:
                    action_1 = agents[1].act(state_1, train=False)

            env.playerAct(0, action_0)
            env.playerAct(1, action_1)

            env.step()
            step_in_game += 1

            state_0 = env.getState(0)
            state_1 = env.getState(1)

            done = env.isDone()
            timeout = step_in_game >= max_steps

            if done or timeout:
                total_steps += step_in_game
                score_0 += env.score[0]
                score_1 += env.score[1]

                if env.score[0] > env.score[1]:
                    wins += 1
                elif env.score[0] < env.score[1]:
                    losses += 1
                else:
                    draws += 1

                break

    if should_print:
        print(
            f"Evaluation over {nb_tests} games | "
            f"W/D/L: {wins}/{draws}/{losses} | "
            f"Win rate: {wins / nb_tests:.2f} | "
            f"Avg steps: {total_steps / nb_tests:.1f} | "
            f"Score: {score_0 / nb_tests:.2f} - {score_1 / nb_tests:.2f}"
        )

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / nb_tests,
        "avg_steps": total_steps / nb_tests,
        "avg_score_0": score_0 / nb_tests,
        "avg_score_1": score_1 / nb_tests,
    }

def train_PPO_competitive(
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
):
    """
    Train a PPO agent using competitive self-play with an opponent pool.
    Includes full training diagnostics and evaluation vs a random agent.
    """

    env = LearningEnvironment(
        players_number=(1, 1),
        scoring_function=model.scoring_function,
        reward_coeff_dict=model.reward_coeff_dict,
        human=False
    )

    # -------------------------
    # Training statistics
    # -------------------------
    current_reward = 0.0
    score_0 = score_1 = 0
    games_played = 0
    wins = losses = draws = 0
    total_steps = 0
    reward_sums = {}  
    num_game = 0

    state = env.getState(0)
    step_in_game = 0
    mean_steps = deque(maxlen=100)
    mean_steps.append(max_steps_per_game)
    total_steps_for_mean = 0
    games_played_for_mean = 0

    opponent_pool = []

    # bootstrap pool with initial snapshot
    opponent_pool.append(clone_opponent(model))

    # Random agent for evaluation
    random_agent = RandomAgent(action_dim=4)

    print(f"Starting PPO self-play with opponent pool ({num_episodes} episodes)")
    start_time = time.time()

    # =========================
    # Training loop
    # =========================
    for episode in range(1, num_episodes + 1):

        if time.time() - start_time > max_duration:
            print("Max training time reached.")
            break

        # ---- select opponent (frozen)
        if len(opponent_pool) == 0:
            opponent = clone_opponent(model)
        else:
            opponent = random.choice(opponent_pool[-3:])

        if opponent is None : 
            env = LearningEnvironment(
                players_number=(1, 0),
                scoring_function=model.scoring_function,
                reward_coeff_dict=model.reward_coeff_dict,
                mean_steps = sum(mean_steps)/len(mean_steps),
                human=False
            )
        else:
            env = LearningEnvironment(
                players_number=(1, 1),
                scoring_function=model.scoring_function,
                reward_coeff_dict=model.reward_coeff_dict,
                mean_steps = sum(mean_steps)/len(mean_steps),
                human=False
            )

        # ---- rollout
        for rollout in range(model.rollout_size):

            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=model.device
            ).unsqueeze(0)

            with torch.no_grad():
                action_0, logprob = model.actor.act(state)
                value = model.critic.net(state_t).squeeze(-1).item()
                if opponent is not None : action_1 = opponent.act(env.getState(1))

            env.playerAct(0, action_0)
            if opponent is not None : env.playerAct(1, action_1)

            rewards = env.step(debug=True)
            reward = rewards[0]

            reward_dict = env.last_reward_components[0]

            # Initialisation dynamique des clés
            if not reward_sums:
                reward_sums = {k: 0.0 for k in reward_dict.keys()}

            for k, v in reward_dict.items():
                reward_sums[k] += v

            current_reward += reward
            step_in_game += 1

            goal_scored = env.isDone()
            timeout = step_in_game >= max_steps_per_game
            rollout_timeout = (rollout == model.rollout_size-1) and not timeout
            done_ppo = timeout or rollout_timeout
            
            if goal_scored:
                env.reset_after_goal()

            if timeout:
                reward += draw_penalty
                current_reward += draw_penalty

            model.remember(state, logprob, done_ppo, value, action_0, reward)

            if done_ppo:
                num_game += 1
                if timeout:
                    total_steps += step_in_game
                    total_steps_for_mean += step_in_game
                    games_played_for_mean += 1
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
                state_t = torch.as_tensor(
                    state, dtype=torch.float32, device=model.device
                ).unsqueeze(0)
                last_value = model.critic.net(state_t).squeeze(-1).item()

        model.replay(last_value=last_value, last_done=done_ppo)
        model.init_memory()

        if games_played_for_mean > 0:
            mean_steps.append(total_steps_for_mean / games_played_for_mean)
        total_steps_for_mean = 0
        games_played_for_mean = 0

        # ---- save opponent snapshot
        if episode % opponent_save_interval == 0:
            opponent_pool.append(clone_opponent(model))

            if len(opponent_pool) > max_pool_size:
                opponent_pool.pop(0)


        # -------------------------
        # Training diagnostics
        # -------------------------
        if episode % interval_notify == 0:
            avg_reward_components = {
                k: v / max(num_game, 1)
                for k, v in reward_sums.items()
            }
            win_rate = wins / games_played if games_played > 0 else 0

            avg_reward_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_reward_components.items()])
            print(
                f"[{int(time.time()-start_time)}s] Ep {episode} | "
                f"Games {games_played} | "
                f"W/D/L {wins}/{draws}/{losses} | "
                f"({win_rate:.2f}) | "
                f"Score {score_0/games_played:.2f}-{score_1/games_played:.2f} | "
                f"{avg_reward_str} | "
                f"Pool {len(opponent_pool)}"
            )


            # reset stats
            current_reward = 0.0
            score_0 = score_1 = 0
            games_played = 0
            wins = losses = draws = 0
            total_steps = 0
            num_game = 0
            reward_sums = {k: 0.0 for k in reward_sums.keys()}

        # -------------------------
        # Evaluation vs random agent
        # -------------------------
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
                should_print=True
            )

    print("Saving final model...")
    model.save(os.path.join(save_path, "model.pt"))
    print("Self-play training finished.")


