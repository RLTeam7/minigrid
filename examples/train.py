import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from PIL import Image
import argparse
from custom_model import MinigridFeaturesExtractor, CustomCNN, CNN_DQN, DQN_UCB
from scratch_dqn import preprocess, ReplayBuffer


parser = argparse.ArgumentParser()

parser.add_argument("--env", default="MiniGrid-Empty-8x8-v0",
                    help="name of the environment to train on")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--max-memory", type=int, default=100000,
                    help="Maximum experiences stored (default: 100000)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.0001)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--use_cpu", default=True,
                    help="use cpu only (default true)")
parser.add_argument("--algo", default="stable_ucb",
                    help="algorithm to use (default dqn)")
parser.add_argument("--total-timesteps", type=int, default=10000,
                    help="total timesteps to train (default: 10000)")
args = parser.parse_args()


if args.algo == "stable_ppo":
    print("using PPO algorithm")
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env = gym.make(args.env, render_mode="rgb_array")
    env = ImgObsWrapper(env)

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(args.total_timesteps)
    model.save(f"ppo_stable_baselines_{args.env}")
    obs, info = env.reset()
    frames = []
    done = False
    while not done:
        frames.append(Image.fromarray(env.unwrapped.get_frame()))
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
            obs, info = env.reset()
    # save the frames into a gif file
    print("Visited terminal state: ",terminated)
    frames[0].save(f"ppo_stable_baselines_{args.env}.gif", save_all=True, append_images=frames[1:], loop=0)
    
elif args.algo == "stable_dqn":
    print("using stable baselines DQN algorithm")

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env = gym.make(args.env, render_mode="rgb_array")
    env = ImgObsWrapper(env)

    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=args.total_timesteps)

    model.save(f"dqn_stable_baselines_{args.env}")
    obs, info = env.reset()
    frames = []
    done = False
    while not done:
        frames.append(Image.fromarray(env.unwrapped.get_frame()))
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
            obs, info = env.reset()
    # save the frames into a gif file
    print("Visited terminal state: ",terminated)
    frames[0].save(f"dqn_stable_baselines_{args.env}.gif", save_all=True, append_images=frames[1:], loop=0)

elif args.algo == "stable_ucb":
    print("UCB with stable DQN")
    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=128))

    env = gym.make(args.env, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    # model = DQN_UCB('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, ucb_c=1.5, learning_rate=args.lr)
    model = DQN_UCB('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, ucb_c=1.5, learning_rate=args.lr)
    model.learn(total_timesteps=args.total_timesteps)

    model.save(f"dqn_stable_baselines_ucb_{args.env}")
    obs, info = env.reset()
    frames = []
    done = False
    while not done:
        frames.append(Image.fromarray(env.unwrapped.get_frame()))
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
            obs, info = env.reset()
    # save the frames into a gif file
    print(terminated)
    frames[0].save(f"dqn_stable_baselines_{args.env}.gif", save_all=True, append_images=frames[1:], loop=0)

elif args.algo == "scratch_dqn":
    print("using dqn scratch algorithm")

    # use only gpu 1
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if args.use_cpu:
        device = "cpu"    
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the environment
    env = gym.make(args.env)
    obs = env.reset()
    obs = obs[0]

    input_dim = obs['image'].shape[0] * obs['image'].shape[1] * obs['image'].shape[2]
    output_dim = env.action_space.n
    input_channels = obs['image'].shape[2]

    model = CNN_DQN(input_channels, output_dim, obs['image'].shape[0]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    buffer = ReplayBuffer(args.max_memory)

    # Training parameters
    num_episodes = 100
    batch_size = 128
    capture_interval = 5
    episode_capture_interval = 10 
    epsilon = 0.2
    for episode in range(num_episodes):
        obs = env.reset()
        obs = obs[0]
        done = False
        total_reward = 0
        step_count = 0
        frames = []

        while not done:

            if step_count % capture_interval == 0:
                frames.append(Image.fromarray(env.unwrapped.get_frame()))
            step_count += 1

            state = preprocess(obs, device)
            q_values = model(state)
            action = torch.argmax(q_values).item()

            # Epsilon-greedy action selection for exploration
            if np.random.rand() < epsilon:
                action = env.action_space.sample()

            next_obs, reward, done, info,_ = env.step(action)
            next_state = preprocess(next_obs, device)

            # Store the transition in replay buffer
            buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            # Sample random minibatch of transitions
            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32)
                states = states.squeeze().to(device)
                actions = torch.tensor(actions, dtype=torch.int64).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                next_states = next_states.squeeze().to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
                current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute Q(s_{t+1}) for all next states
                next_q_values = model(next_states).max(1)[0]
                # Compute the expected Q values
                expected_q_values = rewards + args.discount * next_q_values * (1 - dones)  # Mask out terminal states

                # Compute loss
                loss = criterion(current_q_values, expected_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            obs = next_obs
            total_reward += reward
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # Save frames as a GIF at specified interval
        if (episode + 1) % episode_capture_interval == 0 and frames:
            gif_filename = f'training_progress_episode_{episode + 1}.gif'
            frames[0].save(gif_filename, format='GIF',
                        append_images=frames[1:],
                        save_all=True,
                        duration=300, loop=0)
            print(f"Saved GIF for episode {episode + 1} as {gif_filename}")

    env.close()