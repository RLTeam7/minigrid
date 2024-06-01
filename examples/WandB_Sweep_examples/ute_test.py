from utils import  make_env, make_model, WandBVecVideoRecorder

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecTransposeImage

from models import UTE

import wandb
from wandb.integration.sb3 import WandbCallback

from functools import partial
import numpy as np

if __name__ == '__main__':
    import argparse

    outdir_suffix_dict = {'none': '', 'empty': '', 'time': '%Y%m%dT%H%M%S.%f',
                          'seed': '{:d}', 'params': '{:d}_{:d}_{:d}',
                          'paramsseed': '{:d}_{:d}_{:d}_{:d}'}
    parser = argparse.ArgumentParser('Skip-MDP Tabular-Q')
    parser.add_argument('--episodes', '-e',
                        default=1000,
                        type=int,
                        help='Number of training episodes')
    parser.add_argument('--out-dir',
                        default='experiments/tabular/',
                        type=str,
                        help='Directory to save results. Defaults to tmp dir.')
    parser.add_argument('--out-dir-suffix',
                        default='paramsseed',
                        type=str,
                        choices=list(outdir_suffix_dict.keys()),
                        help='Created suffix of directory to save results.')
    parser.add_argument('--seed', '-s',
                        default=42,
                        type=int,
                        help='Seed')
    parser.add_argument('--env-max-steps',
                        default=50,
                        type=int,
                        help='Maximal steps in environment before termination.',
                        dest='env_ms')
    parser.add_argument('--agent-eps-decay',
                        default='log',
                        choices={'linear', 'log', 'const'},
                        help='Epsilon decay schedule',
                        dest='agent_eps_d')
    parser.add_argument('--agent-eps',
                        default=1.0,
                        type=float,
                        help='Epsilon value. Used as start value when decay linear or log. Otherwise constant value.',
                        dest='agent_eps')
    parser.add_argument('--agent',
                        default='ute',
                        choices={'sq', 'q', 'ute'},
                        type=str.lower,
                        help='Agent type to train')
    parser.add_argument('--env',
                        default='lava',
                        choices={},
                        type=str.lower,
                        help='Enironment to use')
    parser.add_argument('--eval-eps',
                        default=10,
                        type=int,
                        help='After how many episodes to evaluate')
    parser.add_argument('--stochasticity',
                        default=0.0,
                        type=float,
                        help='probability of the selected action failing and instead executing any of the remaining 3')
    parser.add_argument('--no-render',
                        action='store_true',
                        help='Deactivate rendering of environment evaluation')
    parser.add_argument('--max-skips',
                        type=int,
                        default=7,
                        help='Max skip size for tempoRL')
    parser.add_argument('--uncertainty-factor',
                        type=float,
                        default=-1.5,
                        help='for uncertainty-sensitive model')    
    
    # setup output dir
    args = parser.parse_args()
    outdir_suffix_dict['seed'] = outdir_suffix_dict['seed'].format(args.seed)
    outdir_suffix_dict['params'] = outdir_suffix_dict['params'].format(
        args.episodes, args.max_skips, args.env_ms)
    outdir_suffix_dict['paramsseed'] = outdir_suffix_dict['paramsseed'].format(
        args.episodes, args.max_skips, args.env_ms, args.seed)

    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 250000,
        "env_id": "MiniGrid-Empty-16x16-v0",
        "n_env": 1
    }
    
    if args.agent == 'ute':
        out_dir = args.out_dir+'/'+args.env+f'_{args.stochasticity}'+'/'+args.agent+'/'+args.agent_eps_d+'/uncertainty_factor'+str(args.uncertainty_factor)
    else:
        out_dir = args.out_dir+'/'+args.env+f'_{args.stochasticity}'+'/'+args.agent+'/'+args.agent_eps_d
    
    
    np.random.seed(args.seed)  # seed nump
    wandb.init(project='test')
   
    env = DummyVecEnv([
        partial(make_env, config=config) for _ in range(config['n_env']) 
        ])
    
    env = WandBVecVideoRecorder(
        env,
        f"videos/{'test'}",
        record_video_trigger=lambda x: x % ((config['total_timesteps']//config['n_env'])//5)== 0,
        video_length=200,
    )

    env = VecTransposeImage(env)

    if args.agent == 'ute':
        agent = UTE(env.action_space.n, args.max_skips, args.uncertainty_factor, gamma=0.99, env=env)   
    else:
        raise NotImplemented
    
    episodes = args.episodes
    max_env_time_steps = args.env_ms
    epsilon = 0.1

    agent.train(epsilon, epsilon_decay=args.agent_eps_d, max_timesteps=config['total_timesteps'])
    file_name = f"{args.agent}_{args.env}_{args.seed}"
    #agent.save(f"{out_dir}/{file_name}")