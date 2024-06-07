from utils import make_env, make_model
from models import MinigridFeaturesExtractor

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from functools import partial

parameters_dict = {
    'env_id': {
        'values': [
            'MiniGrid-Empty-Random-6x6-v0', 
            'MiniGrid-Empty-16x16-v0',
            'MiniGrid-DistShift1-v0',
            'MiniGrid-LavaGapS6-v0',
            'MiniGrid-MultiRoom-N2-S4-v0',
            'MiniGrid-FourRooms-v0',
        ],
    },
    'model_type': {
        'values': [
            # 'PPO',
            'DQN',
            'ezDQN',
            # 'TRPO',
            # 'A2C',
            'UCB_simhash',
            'UCB'
        ]
    },
    "seed" :{'values': [777]},
    "policy_type": {'value': "CnnPolicy"},
    "total_timesteps": {'value': 100000},
    "n_env": {'value': 1},
}

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'rollout/ep_rew_mean',
        'goal': 'maximize',
    },
    'parameters': parameters_dict,
}

policy_kwargs = {
    "features_extractor_class": MinigridFeaturesExtractor,
    "features_extractor_kwargs": dict(feature_dim=128),
}

def train(config=None):
    run = wandb.init(
        entity='RL_Team7',
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
    )
    config = wandb.config
   
    env = DummyVecEnv([
        partial(make_env, config=config) for _ in range(config['n_env']) 
        ])
    
    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % ((config['total_timesteps']//config['n_env'])//5)== 0,
    #     video_length=200,
    # )

    model_class = make_model(config)
    if model_class is not None:
        if config['model_type'] in ["PPO", "TRPO"]:
            model = model_class(
                config["policy_type"], 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1,
                tensorboard_log=f"runs/{run.id}",
                n_steps=128,
            )
        else:    
            model = model_class(
                config["policy_type"], 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1,
                tensorboard_log=f"runs/{run.id}",
                #device='mps',
            )

        eval_env = DummyVecEnv([
        partial(make_env, config=config) for _ in range(config['n_env']) 
        ])
        eval_env = VecTransposeImage(eval_env)

        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
                ),
                EvalCallback(
                        eval_env,
                        best_model_save_path=f'./best_models/{run.id}',
                        log_path='./logs/',
                        eval_freq=500,
                        deterministic=True, 
                        render=True,
                ),
                ]
            
        )
        test_env = DummyVecEnv([
        partial(make_env, config=config) for _ in range(config['n_env']) 
        ])
        test_env = VecVideoRecorder(
            test_env,
            f"eval_videos/{run.id}",
            record_video_trigger=lambda x: x % (1000)== 0,
            video_length=200,
        )

        if config['model_type'] in ["PPO", "TRPO"]:
            model = model_class(
                config["policy_type"], 
                test_env, 
                policy_kwargs=policy_kwargs, 
                verbose=1,
                tensorboard_log=f"runs/{run.id}",
                #n_steps=128,
            )
        else:    
            model = model_class(
                config["policy_type"], 
                test_env, 
                policy_kwargs=policy_kwargs, 
                verbose=1,
                tensorboard_log=f"runs/{run.id}",
                #device='mps',
            )
        model.load(f'best_models/{run.id}/best_model')

        
        model.learn(1000, WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
                ),)
    
        

    run.finish()

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, entity='RL_Team7', project='Term Project')
    wandb.agent(sweep_id=sweep_id, function=train)