from utils import MinigridFeaturesExtractor, make_env, make_model

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecTransposeImage

from models import UTE
from utils import WandBVecVideoRecorder

import wandb

from functools import partial
import numpy as np

from ucb_simhash import UCB_simhash

parameters_dict = {
    'env_id': {
        'values': [
            'MiniGrid-Empty-Random-6x6-v0', 
            'MiniGrid-Empty-16x16-v0',
            'MiniGrid-DistShift1-v0',
            'MiniGrid-MultiRoom-N2-S4-v0',
            'MiniGrid-MultiRoom-N4-S5-v0',
            'MiniGrid-MultiRoom-N6-v0',
        ],
    },
    'model_type': {
        'value': "UTE"
    },
    "total_timesteps": {'value': 250000},
    "n_env": {'value': 1},

    "seed" : {'values': [42, 43, 44]},
    "agent-eps-decay": {'value': "const"}, #linear / log / const
    "agent-eps": {'value': 0.1}, # start epsilon val
    "max-skips": {'value': 7}, #max skip size
    "uncertainty-factor": {"values": [-1.5, 1.5]}, #for uncertainty-sensitive model
}

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'rollout/ep_rew_mean',
        'goal': 'maximize',
    },
    'parameters': parameters_dict,
}

def train(config=None):
    run = wandb.init(
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
    )
    config = wandb.config
   
    np.random.seed(config.seed)  # seed nump

    env = DummyVecEnv([
        partial(make_env, config=config) for _ in range(config['n_env']) 
        ])
    
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}_{config['env_id']}_{config['model_type']}",
        record_video_trigger=lambda x: x % ((config['total_timesteps']//config['n_env'])//5)== 0,
        video_length=200,
    )
    
    env = VecTransposeImage(env)

    if config["model_type"] != "UTE":
        raise NotImplementedError
    
    model = UTE(
        env.action_space.n,
        config['max-skips'],
        config['uncertainty-factor'],
        gamma=0.99,
        env=env
    )

    model.train(
        config['agent-eps'],
        config['agent-eps-decay'],
        config['total_timesteps'],
    )

    model.save(f"result_models/UTE_{config.env_id}_{config.seed}")

    run.finish()

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='sb3')
    wandb.agent(sweep_id=sweep_id, function=train)


