"""
Based on 
https://github.com/automl/TempoRL
https://github.com/oh-lab/UTE-Uncertainty-aware-Temporal-Extension-/
"""

import os
import json
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
from collections import namedtuple, deque
import time
import numpy as np

import wandb
#from utils import experiments

#from envs.grid_envs import GridCore
import gymnasium as gym
#from envs.grid_envs import Bridge6x10Env, Pit6x10Env, ZigZag6x10, ZigZag6x10H
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

device = 'cpu'

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, feature_dim: int = 512, normalized_image: bool = False) ->None:
        super().__init__(observation_space, feature_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            #print(observation_space)
            #print(torch.as_tensor(observation_space.sample()[None]).float().size())
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    return Variable(torch.from_numpy(ndarray).float().to(device), requires_grad=False)
    
def tt_long(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    return Variable(torch.from_numpy(ndarray).long().to(device), requires_grad=False)
    

def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def get_decay_schedule(start_val: float, decay_start: int, num_steps: int, type_: str):
    """
    Create epsilon decay schedule
    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_steps: Total number of steps to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == 'const':
        return np.array([start_val for _ in range(num_steps)])
    elif type_ == 'log':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_steps - decay_start))])
    elif type_ == 'linear':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_steps - decay_start), endpoint=True)])
    else:
        raise NotImplementedError
    


class Q(nn.Module):
    def __init__(self, observation_space, action_dim, feature_dim=128, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self._feature_extractor = MinigridFeaturesExtractor(observation_space=observation_space, feature_dim=feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._feature_extractor(x)
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)

class BoostrappedDQN(nn.Module):
    def __init__(self, observation_space, action_dim, nheads, feature_dim=128, hidden_dim=50):
        super(BoostrappedDQN, self).__init__()
        self.nheads = nheads
        self._feature_extractor = MinigridFeaturesExtractor(observation_space=observation_space, feature_dim=feature_dim)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim+1, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, action_dim)) for _ in range(self.nheads)])

    def forward_single_head(self, x, a, k):
        x = self._feature_extractor(x)
        x = torch.cat((x, a))
        x = self.heads[k](x)
        return x

    def forward(self, x, a):
        out = []
        x = self._feature_extractor(x)
        x = torch.cat((x, a), dim=1)

        for head in self.heads:
            out.append(head(x))
        return out

class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)

class SkipReplayBuffer:
    """
    Replay Buffer for training the skip-Q.
    Expects "concatenated states" which already contain the behaviour-action for the skip-Q.
    Stores transitions as usual but with additional skip-length. The skip-length is used to properly discount.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states",
                                                 "rewards", "terminal_flags", "lengths"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[], lengths=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done, length):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._data.lengths.append(length)  # Observed skip-length of the transition
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)
            self._data.lengths.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i][0] for i in batch_indices])
        batch_action = np.array([self._data.states[i][1] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        batch_lengths = np.array([self._data.lengths[i] for i in batch_indices])
        return tt(batch_states), tt(batch_action), tt_long(batch_actions), tt(batch_next_states),\
               tt(batch_rewards), tt(batch_terminal_flags), tt(batch_lengths)

class UTE:
    def __init__(self, action_dim: int, skip_dim: int, uncertainty_factor:float, gamma: float,
                 env: VecEnv, out_dir='./runs'):

        self._q = Q(env.observation_space, action_dim).to(device)
        self._q_target = Q(env.observation_space, action_dim).to(device)
        self.n_heads = 10
        self._skip_q = BoostrappedDQN(env.observation_space, skip_dim, self.n_heads).to(device)

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._skip_loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.001)
        self._action_dim = action_dim
        #self._state_dim = state_dim
        self._skip_dim = skip_dim

        self.bernoulli_probability = 0.5
        self.uncertainty_factor = uncertainty_factor

        self._replay_buffer = ReplayBuffer(1e6)
        self._skip_replay_buffer = SkipReplayBuffer(1e6)
        self._env = env
        self.out_dir = out_dir

        self.ep_info_buffer = deque(maxlen=100)

    def get_action(self, x, epsilon: float) -> np.ndarray:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy(), axis=1)
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim, size=1)
        return u

    def get_skip(self, x, a: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Simple helper to get the skip epsilon-greedy based on observation x
        """
        current_outputs = self._skip_q(tt(x), tt(a[..., None]))
        #outputs = []
        #for k in range(self.n_heads):
        #    outputs.append(current_outputs[:, k].detach().cpu().numpy())
        #outputs = np.array(outputs)
        outputs = np.stack([out.detach().cpu().numpy() for out in current_outputs], axis=1) # B x n_head x action_dim
        
        mean_Q = np.mean(outputs , axis=1) # B x action_dim
        std_Q = np.std(outputs, axis=1) # B x action_dim
    
        Q_tilda = mean_Q + self.uncertainty_factor*std_Q
        u = np.argmax(Q_tilda, axis=1)
        return u

    def train(self, epsilon: float,
        epsilon_decay: str = "const", max_timesteps=250000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :return:
        """
        batch_size=32
        epsilon_schedule_action = get_decay_schedule(epsilon, 0, max_timesteps, epsilon_decay)
        start_time = time.time()
        s = self._env.reset()
        total_timesteps = 0
        i_episode = 0
        while total_timesteps < max_timesteps:
            #s = self._env.reset()
            ed, es, er = 0, 0, 0
            steps, rewards, decisions = [], [], []
            while True:
                #one_hot_s = np.eye(self._state_dim)[s]
                epsilon_action = epsilon_schedule_action[total_timesteps]
                a = self.get_action(s, epsilon_action)         
                #skip_state = np.hstack([one_hot_s, [a]])  # concatenate action to the state
                skip = self.get_skip(s, a, 0)
                d = False
                ed += 1
                skip_states, skip_rewards = [], []
                #print('###', skip[0])
                for curr_skip in range(skip[0] + 1):  # play the same action a "skip" times
                    ns, r, d, info = self._env.step(a)
                    ns = ns
                    r = r[0]
                    d = d[0]
                    er += r
                    es += 1
                    total_timesteps += 1
                    
                    #one_hot_s = np.eye(self._state_dim)[s]
                    #skip_states.append(np.hstack([one_hot_s, [a]]))  # keep track of all states that are visited inbetween
                    skip_states.append([s[0], a])
                    skip_rewards.append(r)

                    # Update the skip buffer with all observed transitions
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, curr_skip - skip_id, ns[0],
                                                                skip_reward, d, curr_skip - skip_id + 1) 
                        skip_id += 1

                   # Skip Q update based on double DQN where the target is the behaviour network
                    batch_states, batch_action, batch_actions, batch_next_states, batch_rewards, \
                    batch_terminal_flags, batch_lengths = self._skip_replay_buffer.random_next_batch(batch_size*2)
                    
                    #one_hot_batch_next_states = F.one_hot(batch_next_states, num_classes=self._state_dim).float()

                    #print(batch_rewards.shape, (1-batch_terminal_flags).shape, self._q_target(batch_next_states)[torch.arange(batch_size*2).long(), torch.argmax(
                    #             self._q(batch_next_states), dim=1)].shape)
                    
                    target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma, batch_lengths) * \
                             self._q_target(batch_next_states)[torch.arange(batch_size*2).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    
                    
                    current_outputs = self._skip_q(batch_states, batch_action)
                    masks = torch.bernoulli(torch.zeros((batch_size*2, self.n_heads), device=device) + self.bernoulli_probability )
                    cnt_losses = []
                    #print('***', len(current_outputs), current_outputs[0].shape)
                    for k in range(self.n_heads):
                        total_used = torch.sum(masks[:,k])
                        if total_used > 0.0:
                            current_prediction = current_outputs[k][torch.arange(batch_size*2).long(), batch_actions.long()]
                            l1loss = self._skip_loss_function(current_prediction, target.detach())
                            full_loss = masks[:,k]*l1loss
                            loss = torch.sum(full_loss/total_used)
                            cnt_losses.append(loss)

                    self._skip_q_optimizer.zero_grad()
                    skip_loss = sum(cnt_losses)/self.n_heads
                    skip_loss.backward()
                    self._skip_q_optimizer.step()


                    # Update replay buffer
                    self._replay_buffer.add_transition(s[0], a, ns[0], r, d)
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(batch_size)
    
                    ########### Begin double Q-learning update
                    #one_hot_batch_states = F.one_hot(batch_states, num_classes=self._state_dim).float()
                    #one_hot_batch_next_states = F.one_hot(batch_next_states, num_classes=self._state_dim).float()

                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                            self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                self._q(batch_next_states), dim=1)]
                    current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.squeeze().long()]

                    #print(batch_actions.size(), current_prediction.size(), target.size())
                    loss = self._loss_function(current_prediction, target.detach())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update

                    if d:
                        break
                    s = ns
                    
                if d:
                    break
            
            # logging
            self.ep_info_buffer.append(info[0]['episode'])
            wandb.log(
                {
                    'rollout/ep_rew_mean': np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    'rollout/ep_len_mean': np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    'Charts/global_step' : total_timesteps,
                },
                step=total_timesteps,
            )

            steps.append(es)
            rewards.append(er)
            decisions.append(ed)
            eval_stats = dict(
            elapsed_time=time.time() - start_time,
            training_eps=i_episode,
            avg_num_steps_per_ep=float(np.mean(steps)),
            avg_num_decs_per_ep=float(np.mean(decisions)),
            avg_rew_per_ep=float(np.mean(rewards)),
            std_rew_per_ep=float(np.std(rewards))
            )
            
            print('Done %4d episodes, %5d / %5d timesteps, rewards: %.4f' % (i_episode, total_timesteps, max_timesteps, float(np.mean(rewards))))
            #wandb.log(eval_stats)
            with open(os.path.join(self.out_dir, 'eval_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')
            i_episode += 1
  
    def save(self, filename):
        torch.save(self._q.state_dict(), filename + "_UTE")
        torch.save(self._q_optimizer.state_dict(), filename + "_UTE_optimizer")
        torch.save(self._skip_q.state_dict(), filename + "_UTE_skip")
        torch.save(self._skip_q_optimizer.state_dict(), filename + "_UTE_skip_optimizer")

    def load(self, filename):
        self._q.load_state_dict(torch.load(filename + "_UTE"))
        self._q_optimizer.load_state_dict(torch.load(filename + "_UTE_optimizer"))
        self._skip_q.load_state_dict(torch.load(filename + "_UTE_skip"))
        self._skip_q_optimizer.load_state_dict(torch.load(filename + "_UTE_skip_optimizer"))


