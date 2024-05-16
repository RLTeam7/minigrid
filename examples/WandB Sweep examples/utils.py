from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym

from minigrid.wrappers import ImgObsWrapper

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback

from functools import partial

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
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def make_env(config):
    env = gym.make(config["env_id"], render_mode='rgb_array')
    env = ImgObsWrapper(env)
    env = Monitor(env)  # record stats such as returns
    return env

def make_model(config):
    if config['model_type'] == 'PPO':
        return PPO
    elif config['model_type'] == 'DQN':
        return DQN
    elif config['model_type'] == 'A2C':
        return A2C
    elif config['model_type'] == 'TRPO':
        return TRPO
    
    return None

if __name__ == '__main__':
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 250000,
        #"env_id": "MiniGrid-DistShift1-v0",
        "env_id": "MiniGrid-MultiRoom-N6-v0",
    }

    policy_kwargs = {
        "features_extractor_class": MinigridFeaturesExtractor,
        "features_extractor_kwargs": dict(feature_dim=128),
    }

    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    #env = gym.make(config["env_id"], render_mode="rgb_array")
    #env = ImgObsWrapper(env)
    
    env = DummyVecEnv([partial(make_env, config=config)])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 50000 == 0,
        video_length=200,
    )

    model = PPO(
        config["policy_type"], 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()