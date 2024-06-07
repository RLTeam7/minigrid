import torch
import torch.nn as nn
import gymnasium as gym

from minigrid.wrappers import ImgObsWrapper

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from models import UTE, MinigridFeaturesExtractor
from ezDQN import ezDQN
from ucb_simhash import UCB_simhash

import wandb
from wandb.integration.sb3 import WandbCallback

from functools import partial
from typing import Callable

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
    # elif config['model_type'] == 'TDQN':
    #     return TDQN
    elif config['model_type'] == 'UTE':
        return UTE
    elif config['model_type'] == 'ezDQN':
        return ezDQN
    elif config['model_type'] == "UCB_simhash":
        return UCB_simhash
    
    return None

class WandBVecVideoRecorder(VecVideoRecorder):
    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
    ):
        super().__init__(venv, video_folder, record_video_trigger, video_length, name_prefix)

    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
            wandb.log({"video": wandb.Video(self.video_recorder.path, fps=4, format="mp4")}, step=self.step_id)
        self.recording = False
        self.recorded_frames = 1

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

