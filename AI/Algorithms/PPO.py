import torch
import numpy as np
from tensordict import TensorDict
from torch import nn
import time
from AI.Network import DeepRLNetwork
from Engine.Environment import LearningEnvironment
from Graphics.GraphicEngine import startDisplay
from dataclasses import dataclass

class ActorNetwork(DeepRLNetwork):
    """
    Actor network for PPO.
    The last layer uses Softmax() since the network outputs a probability distribution that must sum to 1.

    Parameters
    ----------
    dimensions : List[int]
        List of integers representing the number of neurons in each layer.
    device : torch.device
        Where the neural network should be cast.
    lr : float
        Learning rate for the network.
    lr_decay : bool
        If True, the learning rate of the network will decrease after each epoch.
    """
    def __init__(self, dimensions: list[int], device : torch.device, lr : float, lr_decay : bool):
        super().__init__(dimensions=dimensions, last_layer=nn.Softmax(dim=1))
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = lr, weight_decay=0)
        if lr_decay:
            self.lr_decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer)
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

    Parameters
    ----------
    dimensions : List[int]
        List of integers representing the number of neurons in each layer.
    device : torch.device
        Where the neural network should be cast.
    lr : float
        Learning rate for the network.
    lr_decay : bool
        If True, the learning rate of the network will decrease after each epoch.
    """
    def __init__(self, dimensions, device, lr, lr_decay):
        super().__init__(dimensions)
        self.device = device
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = lr, weight_decay=0)
        if lr_decay:
            self.lr_decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer)
        else:
            self.lr_decay_scheduler = None
    

class PPOAgent:
    """
    PPO agent. For more information about the algorithm and the implementation:
    https://www.youtube.com/watch?v=5VHLd9eCZ-w
    https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html
    https://arxiv.org/abs/1707.06347
    
    Parameters
    ----------
    dimensions : tuple(List[int])
        2 lists of integers representing the number of neurons in the actor and critic networks.
    scoring_function : callable
        Reward function.
    reward_coeff_dict : dict[float]
        Reward coefficients for the scoring function.
    rollout_size : int
        rollout size before replay
    lr_actor : float
        Learning rate of the actor network.
    lr_critic : float
        Learning rate of the critic network.
    n_epoch : int
        Number of epoch per batch
    lr_decay : bool
        If True, the learning rate of the networks will decrease after each epoch. Defaults to True.
    clip_eps : float
        Epsilon for the ClipPPOLoss. Defaults to 0.2.
    gamma : float
        Discount factor. Used for advantage/rewards-to-go calculations. Defaults to 0.99
    lmbda : float
        Extra factor for the Generalized Advantage Estimate. Defaults to 0.95.
    critic_loss_coeff : float
        Factor for the critic loss. Defaults to 0.5.
    entropy_loss_coeff : float
        Factor for the entropy loss. Defaults to 0.01.
    normalize_advantage : bool
        Whether to normalize the advantage. Defaults to True.
    max_grad_norm : float
        Maximum gradient norm value for gradient clipping. Defaults to 1.0. If the value is 0.0, gradient clipping is disabled.
    cuda : bool
        Whether to use cuda (if available). Defaults to False
    """
    def __init__(self, dimensions: tuple[list[int]], scoring_function: callable, reward_coeff_dict : dict[float], 
                 rollout_size : int, lr_actor: float, lr_critic : float, n_epoch: int,
                 lr_decay: bool=True, clip_eps: float=0.2, gamma: float=0.99, lmbda: float=0.95, 
                 critic_loss_coeff: float=0.5, entropy_loss_coeff: float=0.01, normalize_advantage: bool=True, 
                 max_grad_norm: float=1.0, cuda: bool=False):
        
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
        self.actor = ActorNetwork(dimensions=dimensions[0], device=self.device, 
                                  lr=lr_actor,lr_decay=lr_decay)
        self.critic = CriticNetwork(dimensions=dimensions[1], device=self.device, lr=lr_critic, lr_decay=lr_decay)
    
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
        """Compute Generalized Advantage Estimation"""
        rewards = self.memory["rewards"]
        values  = self.memory["vals"]
        dones   = self.memory["dones"]
    
        advantages = []
        returns = []
    
        gae = 0.0
        next_value = 0.0 if last_done else float(last_value)
    
        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[i])  # 0 if terminal else 1
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            gae = delta + self.gamma * self.lmbda * mask * gae
    
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
    
            next_value = values[i]
    
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns    = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
    
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
        # NOT FOR TRAINING
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
    interval_notify : int = 10):
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

    env = LearningEnvironment(players_number=(1,0), 
                              scoring_function=model.scoring_function, 
                              reward_coeff_dict=model.reward_coeff_dict,
                              human=False)
    print(f"Starting training for maximum {max_duration} seconds (maximum {num_episodes} episodes)")
    start = time.time()
    current_reward = 0
    num_game = 0
    score_history_1, score_history_2 = 0,0
    
    done = False
    state = env.getState(0)

    for i_episode in range(1, num_episodes + 1):
        if time.time() - start > max_duration:
            print("Reached max time for training, interrupting")
            break
        loss_hist = {"clip":0, "val":0, "entropy":0}

        for _ in range(model.rollout_size):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)

            with torch.no_grad():
                action, logprob = model.actor.act(state)
                value = model.critic.net(state_t).squeeze(-1).item()
           
            env.playerAct(0, action)

            reward = env.step()[0]
            current_reward += reward
            done = env.isDone()
            
            model.remember(state, logprob, done, value, action, reward)
            
            if done:
                score_history_1 += env.score[0]
                score_history_2 += env.score[1]
                num_game += 1
                env.reset()
            
            state = env.getState(0)
                
        with torch.no_grad():
            if done:
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

            print(f"[{int(time.time()-start)}s] Episode {i_episode} | ", end="")
            print(f"Rewards: {current_reward/(model.rollout_size*interval_notify):.4f} | Loss_clip : {loss_hist['clip']/interval_notify} | ", end="")
            print(f"Loss_val : {loss_hist['val']/interval_notify} | Loss_entropy : {loss_hist['entropy']/interval_notify} | ", end="")
            print(f"Score: {score_history_1/num_game if num_game != 0 else 0:.2f} - {score_history_2/num_game if num_game != 0 else 0:.2f}")
            
            num_game = 0
            current_reward = 0
            score_history_1, score_history_2 = 0,0
            loss_hist = {"clip":0, "val":0, "entropy":0}

    print("Saving network...")
    model.save(save_path + "model.pt")
    print("Training finished")