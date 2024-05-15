import random
from collections import deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from PIL import Image
# use only gpu 1
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, sample_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)



# CNN-based DQN Model
class CNN_DQN(nn.Module):
    def __init__(self, input_channels, output_dim, image_size):
        super(CNN_DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * image_size * image_size, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# Initialize the environment
env = gym.make("MiniGrid-Empty-8x8-v0")
obs = env.reset()
obs = obs[0]

input_dim = obs['image'].shape[0] * obs['image'].shape[1] * obs['image'].shape[2]
output_dim = env.action_space.n
input_channels = obs['image'].shape[2]

model = CNN_DQN(input_channels, output_dim, obs['image'].shape[0]).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

buffer = ReplayBuffer(10000)  # Replay buffer for 10,000 experiences

def preprocess(obs):
    # Normalize pixel values and add a batch dimension (BCHW)
    return torch.tensor(obs['image'].transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)


# Training parameters
num_episodes = 100
batch_size = 32
capture_interval = 10
episode_capture_interval = 10 

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

        state = preprocess(obs)
        q_values = model(state)
        action = torch.argmax(q_values).item()

        # Epsilon-greedy action selection for exploration
        if np.random.rand() < 0.1:
            action = env.action_space.sample()

        next_obs, reward, done, info,_ = env.step(action)
        next_state = preprocess(next_obs)

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
            expected_q_values = rewards + 0.99 * next_q_values * (1 - dones)  # Mask out terminal states

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
