import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3 import DQN
from models import MinigridFeaturesExtractor
# from utils import make_env

import hashlib


SelfDQN = TypeVar("SelfDQN", bound="DQN")

def hash_dimension(index):
    """Generate a consistent hash for each dimension index."""
    return int(hashlib.md5(str(index).encode('utf-8')).hexdigest(), 16)

def normalize_tensor(tensor):
    """Normalize the tensor to [0, 1] range."""
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    if max_val - min_val == 0:
        return th.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)

def discretize_tensor(tensor, bins=100):
    """Discretize the normalized tensor into `bins` discrete values."""
    return (tensor * bins).long()

def simhash_tensor(tensor, bins=100):
    hash_vector = [0] * bins

    # Normalize and discretize the tensor
    normalized_tensor = normalize_tensor(tensor)
    discrete_tensor = discretize_tensor(normalized_tensor, bins=bins)

    for i, weight in enumerate(discrete_tensor[0]):
        hash_value = hash_dimension(i)
        for j in range(bins):
            bitmask = 1 << j
            if hash_value & bitmask:
                hash_vector[j] += weight.item()
            else:
                hash_vector[j] -= weight.item()

    simhash_value = 0
    for i in range(bins):
        if hash_vector[i] > 0:
            simhash_value |= 1 << i

    return simhash_value % bins

    
class UCB_simhash(DQN):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy


    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.n = 0
        self.action = None
        self.state = None
        self.bins = 100
        self.n_count = np.zeros((self.bins, self.action_space.n))
        self.c = 2
        
        if _init_setup_model:
            self._setup_model()


    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())

            th_obs = th.as_tensor(observation).to(self.device)
            feat = self.policy.extract_features(th_obs,self.policy.q_net.features_extractor)
            feat = feat.cpu()
            state = simhash_tensor(feat, bins=100) # let's say we have 100 states
            self.n_count[state][action] += 1 
        else:
            th_obs = th.as_tensor(observation).to(self.device)
            feat = self.policy.extract_features(th_obs,self.policy.q_net.features_extractor)
            feat = feat.cpu()
            state = simhash_tensor(feat, bins=100) # let's say we have 100 states
            
            with th.no_grad():
                q_values = self.policy.q_net(th_obs)

            count_values = [self.n_count[state][i] for i in range(self.action_space.n)]
            count_values = th.tensor(count_values).to(self.device)
            count_values = count_values.to(q_values.dtype)

            # make tensor that has a value of 5, and shape as q_values
            constant_tensor = th.ones_like(q_values) * 5

            # add a small epsilon to avoid division by zero
            q_values = q_values + self.c * th.sqrt(th.log(constant_tensor)/ (count_values + 1e-5))

            action = q_values.argmax(dim=1).reshape(-1)
            action = action.to("cpu")
            self.n_count[state][action] += 1                

        return action, state

if __name__ == "__main__":
    policy_kwargs = {
    "features_extractor_class": MinigridFeaturesExtractor,
    "features_extractor_kwargs": dict(feature_dim=128),
}
    config ={'env_id':'MiniGrid-Empty-Random-6x6-v0'}
    model_class = UCB_simhash
    from utils import make_env
    env = make_env(config)
    model = model_class("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(10000)